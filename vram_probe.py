# -*- coding: utf-8 -*-
"""Measure peak training VRAM of the V2 design (extractor + TCN + linear head).

One full step per config: forward, BCE, backward, Adam step. Peak is
torch.cuda.max_memory_allocated, i.e. excluding the CUDA context (~0.3-0.5 GB).
"""
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self, s1):
        super().__init__()
        k, p = (3, 5, 5), (1, 0, 0)
        self.convs = nn.ModuleList([
            nn.Conv3d(3, 16, k, stride=(1, s1, s1), padding=p),
            nn.Conv3d(16, 24, k, padding=p),
            nn.Conv3d(24, 32, k, padding=p),
        ])
        self.bns = nn.ModuleList([nn.BatchNorm3d(c) for c in (16, 24, 32)])
        self.pool = nn.AvgPool3d((1, 4, 4), stride=(1, 2, 2))
        self.spatial = nn.AdaptiveAvgPool3d((None, 4, 4))   # keep T
        self.out_bn = nn.BatchNorm3d(32)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(self.pool(conv(x))))           # V1 order: norm(pool(conv)) -> act
        x = self.out_bn(self.spatial(x))                  # [B,32,T,4,4]
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b, t, c * h * w)   # [B,T,512]


class TCNBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(512, 512, 3, dilation=d, padding=d)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        return x + F.gelu(self.bn(self.conv(x)))


class Net(nn.Module):
    def __init__(self, s1):
        super().__init__()
        self.ext = Extractor(s1)
        self.tcn = nn.Sequential(TCNBlock(1), TCNBlock(2))
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        z = self.ext(x).transpose(1, 2)        # [B,512,T]
        z = self.tcn(z).mean(dim=2)            # temporal GAP
        return self.head(z).squeeze(-1)


def probe(B, T, H, W, s1, amp):
    torch.cuda.empty_cache()
    net = Net(s1).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    x = torch.randn(B, 3, T, H, W, device='cuda')
    y = torch.randint(0, 2, (B,), device='cuda').float()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        with torch.cuda.amp.autocast(enabled=amp):
            loss = F.binary_cross_entropy_with_logits(net(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        res = round((peak - base) / 2**30, 3)
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            raise
        res = 'OOM'
    del net, opt, x, y
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    configs = [
        # name,               B, T,  H,   W,  s1, amp
        ("celeb s1",          1, 8,  256, 256, 1, False),
        ("celeb s1",          2, 8,  256, 256, 1, False),
        ("celeb s1",          1, 16, 256, 256, 1, False),
        ("celeb s1",          1, 32, 256, 256, 1, False),
        ("celeb s1",          2, 32, 256, 256, 1, False),
        ("celeb s1 amp",      1, 32, 256, 256, 1, True),
        ("celeb s1 amp",      2, 32, 256, 256, 1, True),
        ("celeb s2",          1, 32, 256, 256, 2, False),
        ("celeb s2",          2, 32, 256, 256, 2, False),
        ("dfd s1",            1, 8,  540, 960, 1, False),
        ("dfd s1",            1, 16, 540, 960, 1, False),
        ("dfd s1",            1, 32, 540, 960, 1, False),
        ("dfd s1 amp",        1, 16, 540, 960, 1, True),
        ("dfd s1 amp",        1, 32, 540, 960, 1, True),
        ("dfd s2",            1, 32, 540, 960, 2, False),
        ("dfd s2 amp",        1, 32, 540, 960, 2, True),
    ]
    out = []
    for name, B, T, H, W, s1, amp in configs:
        r = probe(B, T, H, W, s1, amp)
        out.append(dict(name=name, B=B, T=T, H=H, W=W, s1=s1, amp=amp, peak_gb=r))
        print("{:14s} B={} T={:2d} {}x{} -> {}".format(name, B, T, W, H, r), flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
