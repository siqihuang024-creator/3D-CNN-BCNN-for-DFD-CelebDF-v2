# -*- coding: utf-8 -*-
"""Inline-SVG figures for 实验方案_V2.html.

Each figure is drawn on a fixed grid with native SVG shapes. Colours come
from page-level CSS classes (see build.py), so the drawings follow the
page's light/dark theme; the one accent hue marks "what changed".
"""
import html

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def esc(s):
    return html.escape(str(s), quote=False)


def tw(s, size=12, mono=False):
    """Rough rendered width, used only to warn about overflowing labels."""
    if mono:
        return sum(size if ord(c) >= 0x2E80 else size * 0.62 for c in s)
    w = 0.0
    for c in s:
        if ord(c) >= 0x2E80 or c in "，。、：；（）·→←≤×…−":
            w += size
        elif c in "il.,:;|!'[]()1 ":
            w += size * 0.34
        elif c.isupper() or c in "mw@%":
            w += size * 0.68
        else:
            w += size * 0.56
    return w


def is_mono(s):
    s = s.strip()
    return s.startswith("[") or s.startswith("→ [")


class Fig:
    def __init__(self, fid, w, h, aria):
        self.fid, self.w, self.h, self.aria = fid, w, h, aria
        self.el, self.warn = [], []

    def add(self, s):
        self.el.append(s)

    # -- primitives --------------------------------------------------------
    def rect(self, x, y, w, h, cls="box", rx=6):
        self.add('<rect class="{}" x="{}" y="{}" width="{}" height="{}" rx="{}"/>'
                 .format(cls, x, y, w, h, rx))

    def text(self, x, y, s, cls="t", anchor="middle"):
        self.add('<text class="{}" x="{}" y="{}" text-anchor="{}">{}</text>'
                 .format(cls, x, y, anchor, esc(s)))

    def dot(self, x, y, r=5, cls="dot"):
        self.add('<circle class="{}" cx="{}" cy="{}" r="{}"/>'.format(cls, x, y, r))

    def line(self, pts, cls="ln", head=True, hl=False, sw=None):
        p = " ".join("{},{}".format(a, b) for a, b in pts)
        mk = ""
        if head:
            mk = ' marker-end="url(#{}-{})"'.format(self.fid, "h" if hl else "a")
        c = cls + (" hl" if hl and "hl" not in cls else "")
        st = ' style="stroke-width:{}"'.format(sw) if sw else ""
        self.add('<polyline class="{}" points="{}"{}{}/>'.format(c, p, st, mk))

    # -- composite ---------------------------------------------------------
    def box(self, x, y, w, h, title, subs=(), cls="box", tcls="t"):
        self.rect(x, y, w, h, cls)
        lines = [(title, tcls)]
        for s in subs:
            if isinstance(s, tuple):
                lines.append(s)
            else:
                lines.append((s, "tm" if is_mono(s) else "ts"))
        total = 15 + 14 * (len(lines) - 1)
        base = y + (h - total) / 2.0 + 11.5
        cx = x + w / 2.0
        for i, (s, c) in enumerate(lines):
            yy = base if i == 0 else base + 15 + 14 * (i - 1)
            size = 12 if i == 0 else 11
            if tw(s, size, mono=(c == "tm")) > w - 8:
                self.warn.append("{}: '{}' ~{:.0f}px in box {}px".format(
                    self.fid, s, tw(s, size, c == "tm"), w))
            self.text(cx, round(yy, 1), s, c)
        if total > h - 4:
            self.warn.append("{}: box '{}' too short".format(self.fid, title))

    def render(self):
        defs = (
            '<defs>'
            '<marker id="{f}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto"><path class="mk" d="M0,0 L10,5 L0,10 z"/></marker>'
            '<marker id="{f}-h" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            'markerHeight="6" orient="auto"><path class="mk-hl" d="M0,0 L10,5 L0,10 z"/></marker>'
            '</defs>'
        ).format(f=self.fid)
        return ('<svg class="dg" viewBox="0 0 {w} {h}" role="img" aria-label="{a}" '
                'xmlns="http://www.w3.org/2000/svg">{d}{b}</svg>').format(
            w=self.w, h=self.h, a=esc(self.aria), d=defs, b="".join(self.el))


# --------------------------------------------------------------------------
# F1 overview
# --------------------------------------------------------------------------


def fig_overview():
    f = Fig("ov", 980, 300,
            "总体流程：P0.5 划分后先量捷径地板，Phase A 训练提取器，冻结后分别做线性探针和只用真实视频的贝叶斯头")
    f.box(20, 50, 140, 46, "DFD", ["整帧 · 3,431 视频"])
    f.box(20, 126, 140, 46, "CelebDF++", ["人脸裁剪 · 54,086 视频"])
    f.box(200, 78, 150, 66, "P0.5 划分", ["源视频家族不相交", "70 / 15 / 15"])
    f.line([(160, 73), (180, 73), (180, 100), (200, 100)])
    f.line([(160, 149), (180, 149), (180, 122), (200, 122)])
    f.box(200, 236, 150, 52, "E0S · 捷径地板", ["码率 / 帧数 / 分辨率"])
    f.line([(275, 144), (275, 236)])
    f.text(283, 194, "同一划分", "tl", "start")
    f.box(420, 40, 180, 146, "Phase A · 表征学习",
          ["3D-CNN + TCN + Linear", "real + fake，BCE", "全部可训练 · AMP"])
    f.line([(350, 111), (420, 111)])
    f.text(385, 104, "clips", "tl")
    f.box(640, 86, 140, 54, "冻结的提取器", ["3D-CNN + TCN + 聚合"], cls="box frz")
    f.line([(600, 113), (640, 113)])
    f.text(620, 106, "best.pt", "tl")
    f.box(820, 28, 150, 62, "Phase B · 探针", ["ridge，按身份分折", "→ 探针 AUROC"])
    f.box(820, 118, 150, 76, "Phase C · 贝叶斯头", ["512→256→64→1", ("只用真实视频训练", "ts hlt")])
    f.line([(780, 113), (800, 113), (800, 59), (820, 59)])
    f.line([(800, 113), (800, 156), (820, 156)])
    f.text(806, 106, "512 维", "tl", "start")
    f.box(640, 236, 140, 52, "结果表", ["AUROC · AP · EER"])
    f.line([(895, 194), (895, 262), (780, 262)])
    f.text(838, 255, "异常分", "tl")
    f.line([(350, 262), (640, 262)], cls="ln dash")
    f.text(495, 255, "地板写进每张结果表", "tl")
    cap = ("总体流程。两个数据集按 P0.5 划分后，先量出元数据捷径能达到的分数（E0S）；"
           "Phase A 用有标签的 real + fake 训练提取器和 TCN；之后提取器冻结（虚线框），"
           "Phase B 在它的 512 维特征上拟合线性探针，判断表征本身好不好；"
           "Phase C 只用真实视频训练贝叶斯头，输出异常分。时间聚合从 GAP 起步，由 E4′ 选定。")
    return f, cap


# --------------------------------------------------------------------------
# F2 preprocessing
# --------------------------------------------------------------------------


def fig_preprocess():
    f = Fig("pp", 1000, 270,
            "预处理数据流：DFD 像素间隔采样，CelebDF++ 读缓存、对齐、裁剪，两者在时间采样处汇合")
    f.box(20, 24, 150, 52, "DFD 原始帧", ["1920×1080 · 24 fps"])
    f.line([(170, 50), (260, 50)])
    f.text(215, 42, "[::2, ::2]", "tm")
    f.box(260, 24, 160, 52, "整帧 960×540", ["不插值，保留全部场景"])
    f.box(20, 120, 150, 52, "CelebDF++ 原始帧", ["30 fps"])
    f.line([(170, 146), (200, 146)])
    f.box(200, 120, 130, 52, "读人脸缓存", ["dlib 框 + 81 点"])
    f.line([(330, 146), (355, 146)])
    f.box(355, 120, 110, 52, "眼线对齐", ["按关键点旋转"])
    f.line([(465, 146), (490, 146)])
    f.box(490, 120, 130, 52, "框 ×2.0 裁剪", ["缩放到 256×256"])
    f.box(670, 69, 150, 58, "时间采样", ["随机起点 · 步长 s", "取 32 帧"])
    f.line([(420, 50), (645, 50), (645, 88), (670, 88)])
    f.line([(620, 146), (645, 146), (645, 108), (670, 108)])
    f.line([(820, 98), (850, 98)])
    f.box(850, 69, 140, 58, "转张量 + 归一化", ["在 worker 内完成", "float32"])
    f.line([(920, 127), (920, 196)])
    f.text(928, 165, "collate", "tl", "start")
    f.box(850, 196, 140, 52, "送入 GPU", ["[B,3,32,H,W]"])
    f.text(830, 214, "每个 clip 在主机内存里：", "ts", "end")
    f.text(830, 232, "DFD 199 MB · CelebDF++ 25 MB", "ts hlt", "end")
    cap = ("预处理数据流。DFD 只做像素间隔采样、不做人脸检测；CelebDF++ 读离线人脸缓存后对齐、裁剪。"
           "两条路在时间采样处汇合，之后完全相同。张量在 DataLoader 的 worker 里就转成了 float32，"
           "所以 DFD 一个 32 帧 clip 在主机内存里占 199 MB，是 CelebDF++ 的 8 倍——"
           "这决定了远程机器上 worker 能开几个（第 1.2 节）。")
    return f, cap


# --------------------------------------------------------------------------
# F3 time axis
# --------------------------------------------------------------------------


def fig_timeaxis():
    f = Fig("ta", 980, 300,
            "时间轴上的三个杠杆：输入采样步长、3D-CNN 的 7 帧感受野、TCN 的空洞卷积")
    x1 = lambda i: 160 + i * 33          # original frames
    x2 = lambda k: 160 + k * 66          # sampled time steps
    y1, y2, y3, y4 = 46, 116, 186, 256
    # row labels
    for y, a, b in [(y1, "原视频", "s=2：隔一帧取"), (y2, "网络输入", "32 帧，画出前 12"),
                    (y3, "3D-CNN 输出", "每步看 7 个输入步"), (y4, "TCN 空洞卷积", "d=2：隔一步取")]:
        f.text(20, y - 3, a, "t", "start")
        f.text(20, y + 13, b, "ts", "start")
    # connectors original -> sampled
    for k in range(12):
        f.line([(x1(2 * k), y1 + 5), (x2(k), y2 - 6)], cls="ln faint", head=False)
    # fan: z5 <- inputs t=2..8
    for k in range(2, 9):
        f.line([(x2(k), y2 + 6), (x2(5), y3 - 7)], cls="ln hl thin", head=False)
    # taps: h5 <- z3, z5, z7
    for k in (3, 5, 7):
        f.line([(x2(k), y3 + 6), (x2(5), y4 - 7)], cls="ln hl thin", head=False)
    for i in range(24):
        f.dot(x1(i), y1, 4.5 if i % 2 == 0 else 3.2, "dot" if i % 2 == 0 else "dot o")
    for k in range(12):
        f.dot(x2(k), y2, 5, "dot hl" if 2 <= k <= 8 else "dot")
        f.dot(x2(k), y3, 5, "dot hl" if k in (3, 5, 7) else "dot")
        f.dot(x2(k), y4, 5, "dot hl" if k == 5 else "dot")
    f.text(945, y1 + 4, "…", "t")
    for y in (y2, y3, y4):
        f.text(930, y + 4, "…", "t")
    f.text(x2(7) + 30, y3 + 30, "z5 看输入 t=2…8（原视频 13 帧）", "ts hlt", "start")
    f.text(x2(5) + 110, y4 + 26, "h5 取 z3、z5、z7", "ts hlt")
    cap = ("时间轴上的三个杠杆。第 1→2 行：输入采样步长 s 决定 32 帧覆盖多长的真实时间"
           "（s=2 时覆盖 64 帧），网络内部仍是 32 个时间步。第 2→3 行：三层 3D 卷积让每个输出步看 7 个相邻的输入步，"
           "相邻输出步大量重叠，但相隔 7 步以上就互不相交。第 3→4 行：TCN 的空洞卷积隔步取值，"
           "不减少时间步就把视野拉远（这里只画 d=2 那一层的取法，两层如何叠加见图 7）。")
    return f, cap


# --------------------------------------------------------------------------
# F4 ladder
# --------------------------------------------------------------------------


def fig_ladder():
    f = Fig("ld", 980, 412,
            "实验阶梯 E0 到 E4：每行一次实验的数据流，橙色为相对上一行唯一改动的部件")
    xs = [90 + 150 * i for i in range(6)]
    w, h = 132, 48
    heads = ["输入", "3D-CNN ×3", "时间维处理", "空间池化", "时序建模", "输出 → 头"]
    for x, s in zip(xs, heads):
        f.text(x + w / 2, 26, s, "th")
    base = [("T=8 · B=1", "与 V1 相同"), ("stride 1", "感受野 7 帧"),
            ("mean(T)", "时间被平均掉"), ("2D 池化 22×22", "每 clip 一张图"),
            None, ("15488 维", "Linear → 1")]
    rows = [
        ("E0", "V1 基线", {}, set()),
        ("E1", "clip 长度", {0: ("T=32 · B=1", "clip 加长 4 倍")}, {0}),
        ("E2", "batch", {0: ("T=32 · B=8", "DFD 为 4×累积 2")}, {0}),
        ("E3", "特征维度", {0: ("T=32 · B=8", "DFD 为 4×累积 2"),
                          3: ("2D 池化 4×4", "特征 15488→512"),
                          5: ("512 维", "Linear → 1")}, {3}),
        ("E4", "时序建模", {0: ("T=32 · B=8", "DFD 为 4×累积 2"),
                          2: ("不做 mean", "T=32 保留"),
                          3: ("3D 池化 (T,4,4)", "每个时间步 512"),
                          4: ("TCN ×2 → GAP", "d = 1, 2"),
                          5: ("512 维", "Linear → 1")}, {2, 3, 4}),
        ("E4′", "时间聚合", {0: ("T=32 · B=8", "DFD 为 4×累积 2"),
                           2: ("不做 mean", "T=32 保留"),
                           3: ("3D 池化 (T,4,4)", "每个时间步 512"),
                           4: ("TCN → 换聚合", "Max/Attn/CLS/Flatten"),
                           5: ("512 维", "Flatten 先投影")}, {4}),
    ]
    for r, (name, sub, over, hl) in enumerate(rows):
        y = 40 + 62 * r
        f.text(20, y + 22, name, "th", "start")
        f.text(20, y + 38, sub, "ts", "start")
        for c, x in enumerate(xs):
            cell = over.get(c, base[c])
            if cell is None:
                f.box(x, y, w, h, "无", [], cls="box gh", tcls="ts")
            else:
                f.box(x, y, w, h, cell[0], [cell[1]], cls="box hl" if c in hl else "box")
            if c < 5:
                f.line([(x + w, y + h / 2), (x + 150, y + h / 2)])
    cap = ("实验阶梯 E0–E4。每一行是一次实验的完整数据流，橙色框是相对上一行唯一改动的部件"
           "（E4 的三个橙框是同一个改动：去掉提前的时间平均、改为保留时间步交给 TCN）。"
           "E0–E3 在 3D-CNN 之后立刻对时间求平均，之后的部件都看不到时间顺序。"
           "E4′ 是一组实验：只把 TCN 之后的 GAP 换成别的聚合方式（4.5 节）。")
    return f, cap


# --------------------------------------------------------------------------
# F5 tail comparison V1 vs V2
# --------------------------------------------------------------------------


def fig_tail():
    f = Fig("tl", 1000, 215,
            "V1 与 V2 在 stage 3 之后的区别：V1 先对时间求平均，V2 保留时间步交给 TCN 后才平均")
    f.box(20, 40, 120, 150, "Stage 1–3", ["两者完全相同", "3D 卷积 ×3"])
    # V1
    f.text(150, 34, "V1 · E0–E2", "th", "start")
    y = 52
    f.line([(140, 76), (240, 76)])
    f.text(190, 68, "[B,32,T,26,26]", "tm")
    f.box(240, y, 120, 48, "mean(T)", [("时间在此抹掉", "ts hlt")], cls="box hl")
    f.line([(360, 76), (450, 76)])
    f.text(405, 68, "[B,32,26,26]", "tm")
    f.box(450, y, 100, 48, "2D 池化", ["→ 22×22"])
    f.line([(550, 76), (640, 76)])
    f.text(595, 68, "[B,32,22,22]", "tm")
    f.box(640, y, 110, 48, "BN2d + 展平", ["15488 维"])
    f.line([(750, 76), (840, 76)])
    f.text(795, 68, "[B,15488]", "tm")
    f.box(840, y, 70, 48, "Linear")
    f.line([(910, 76), (970, 76)])
    f.text(940, 68, "logit", "tl")
    # V2
    f.text(150, 128, "V2 · E4", "th", "start")
    y = 142
    f.line([(140, 166), (240, 166)])
    f.text(190, 158, "[B,32,T,26,26]", "tm")
    f.box(240, y, 120, 48, "3D 池化 (T,4,4)", ["只压空间"])
    f.line([(360, 166), (450, 166)])
    f.text(405, 158, "[B,32,T,4,4]", "tm")
    f.box(450, y, 100, 48, "BN3d + 重排", ["每步 512 维"])
    f.line([(550, 166), (620, 166)])
    f.text(585, 158, "[B,T,512]", "tm")
    f.box(620, y, 90, 48, "TCN ×2", ["d = 1, 2"], cls="box hl")
    f.line([(710, 166), (780, 166)])
    f.text(745, 158, "[B,T,512]", "tm")
    f.box(780, y, 66, 48, "GAP(T)", ["最后才平均"])
    f.line([(846, 166), (910, 166)])
    f.text(878, 158, "[B,512]", "tm")
    f.box(910, y, 66, 48, "Linear")
    cap = ("V1 与 V2 的唯一结构区别在 stage 3 之后（箭头上是流过的张量形状）。"
           "V1 先对时间求平均，再把空间压到 22×22，得到 15488 维；"
           "V2 只把空间压到 4×4、保留 32 个时间步，每步恰好 32×4×4 = 512 维，交给 TCN 之后才对时间求平均。")
    return f, cap


# --------------------------------------------------------------------------
# F6 E4 full model, layer by layer
# --------------------------------------------------------------------------


def fig_e4():
    f = Fig("e4", 980, 530,
            "E4 主模型逐层张量形状，CelebDF++ 与 DFD 两列对照，自适应池化后形状相同")
    SAME = None
    rows = [
        ("输入 clip", "[B,3,32,256,256]", "[B,3,32,540,960]", "", ""),
        ("Stage 1 · Conv3d 3→16，k(3,5,5)", "[B,16,32,252,252]", "[B,16,32,536,956]", "3,616", ""),
        ("AvgPool (1,4,4)/(1,2,2) + BN + ReLU", "[B,16,32,125,125]", "[B,16,32,267,477]", "32", ""),
        ("Stage 2 · Conv3d 16→24", "[B,24,32,121,121]", "[B,24,32,263,473]", "28,824", ""),
        ("AvgPool + BN + ReLU", "[B,24,32,59,59]", "[B,24,32,130,235]", "48", ""),
        ("Stage 3 · Conv3d 24→32", "[B,32,32,55,55]", "[B,32,32,126,231]", "57,632", ""),
        ("AvgPool + BN + ReLU", "[B,32,32,26,26]", "[B,32,32,62,114]", "64", ""),
        ("AdaptiveAvgPool3d (T,4,4) + BN3d", "[B,32,32,4,4]", SAME, "64", "hl"),
        ("重排：每个时间步展平 32×4×4", "[B,32,512]", SAME, "", "hl"),
        ("TCN-1 · Conv1d k3 d1 + BN + GELU + 残差", "[B,32,512]", SAME, "787,968", "hl"),
        ("TCN-2 · Conv1d k3 d2 + BN + GELU + 残差", "[B,32,512]", SAME, "787,968", "hl"),
        ("GAP：沿 T 求平均", "[B,512]", SAME, "", "hl"),
        ("Linear 512→1（Phase A 头）", "[B]", SAME, "513", ""),
    ]
    f.text(175, 28, "层 / 运算", "th")
    f.text(470, 28, "CelebDF++ · 256×256", "th")
    f.text(730, 28, "DFD · 960×540", "th")
    f.text(965, 28, "本层参数量", "th", "end")
    for i, (op, c, d, p, hl) in enumerate(rows):
        y = 44 + 34 * i
        if i:
            f.line([(175, y - 8), (175, y)], head=False)
        f.box(20, y, 310, 26, op, [], cls="box hl" if hl else "box", tcls="tn")
        cy = y + 17
        if d is SAME:
            f.text(600, cy, c, "tm")
        else:
            f.text(470, cy, c, "tm")
            f.text(730, cy, d, "tm")
        if p:
            f.text(965, cy, p, "tm", "end")
    f.line([(350, 44 + 34 * 7 - 4), (850, 44 + 34 * 7 - 4)], cls="ln dash faint", head=False)
    f.text(20, 506, "提取器 90,280 · TCN 1,575,936 · Phase A 头 513 · 合计 1,666,729（参数列中 BN 已计入）",
           "ts", "start")
    cap = ("E4 主模型逐层张量形状，数据自上而下流动。两个数据集用同一套层和权重，只有输入的 H×W 不同；"
           "自适应池化把两边都压到 4×4，虚线以下形状完全相同（合并成中间一列）。"
           "橙色为相对 V1 新增或改动的层。张量写作 [B, C, T, H, W]——这里通道数 C 和时间步 T 恰好都是 32，别看混。"
           "最右列是这一层的参数量，不是特征维度：TCN 每层 787,968 个参数，进出都是每步 512 维。")
    return f, cap


# --------------------------------------------------------------------------
# F7 TCN block + dilation stacking
# --------------------------------------------------------------------------


def fig_tcn():
    f = Fig("tc", 980, 320,
            "TCN 残差块结构，以及两层空洞卷积叠加后输出 t=5 看到输入 t=2 到 8")
    f.text(160, 30, "输入 [B,512,32]", "tm")
    f.line([(160, 36), (160, 60)])
    f.box(80, 60, 160, 44, "Conv1d k=3，空洞 d", ["512 → 512 通道"])
    f.line([(160, 104), (160, 124)])
    f.box(80, 124, 160, 36, "BatchNorm1d")
    f.line([(160, 160), (160, 180)])
    f.box(80, 180, 160, 36, "GELU")
    f.line([(160, 216), (160, 239)])
    f.add('<circle class="box" cx="160" cy="252" r="13"/>')
    f.text(160, 257, "+", "t")
    f.line([(160, 265), (160, 286)])
    f.text(160, 302, "输出 [B,512,32]", "tm")
    f.line([(160, 46), (280, 46), (280, 252), (173, 252)], hl=True)
    f.text(288, 150, "残差：原样加回", "ts hlt", "start")
    # right panel
    X = lambda k: 440 + 42 * k
    f.text(440, 30, "两层叠加后的时间视野（以输出 t=5 为例）", "th", "start")
    rows = [(80, "输入：3D-CNN 输出 z", set(range(2, 9)), "t=2…8 共 7 步"),
            (160, "TCN-1（d=1）", {3, 5, 7}, "t=3、5、7"),
            (240, "TCN-2（d=2）", {5}, "t=5")]
    for y, lab, hot, note in rows:
        f.text(440, y - 16, lab, "ts", "start")
        f.text(872, y + 4, note, "ts hlt", "start")
    for k in (3, 5, 7):
        f.line([(X(k), 165), (X(5), 234)], cls="ln hl thin", head=False)
    for k in (3, 5, 7):
        for j in (k - 1, k, k + 1):
            f.line([(X(j), 85), (X(k), 154)], cls="ln hl thin", head=False)
    for y, lab, hot, note in rows:
        for k in range(11):
            f.dot(X(k), y, 5, "dot hl" if k in hot else "dot")
    cap = ("左：TCN 的一层 = 空洞 Conv1d → BatchNorm1d → GELU，再把输入原样加回（残差）。"
           "残差让 TCN 在时序建模帮不上忙时可以退化成恒等映射，不会破坏已有特征。"
           "右：第 2 层（d=2）的输出 t=5 取第 1 层的 t=3、5、7，而它们又各取输入的相邻 3 步，"
           "合起来看到输入 t=2…8 共 7 步——只占 32 步中的一小段，余量充足。")
    return f, cap


# --------------------------------------------------------------------------
# F8 Phase B / C
# --------------------------------------------------------------------------


def fig_phasebc():
    f = Fig("bc", 980, 310,
            "E5 与 E6：都读取选定模型冻结后的 512 维特征，E5 拟合线性探针，E6 只用真实视频训练贝叶斯头")
    f.box(20, 122, 110, 56, "一个视频", ["均匀取 8 个 clip"])
    f.line([(130, 150), (210, 150)])
    f.text(170, 142, "每次 ≤4 clip", "tl")
    f.box(210, 106, 165, 88, "选定模型（冻结）", ["3D-CNN + TCN + 聚合", "eval()，不回传梯度"], cls="box frz")
    f.line([(375, 150), (438, 150)])
    f.text(406, 142, "[8,512]", "tm")
    f.box(438, 126, 92, 48, "clip 均值", ["每视频 512 维"])
    f.line([(530, 150), (560, 150), (560, 66), (600, 66)])
    f.line([(560, 150), (560, 234), (600, 234)])
    f.text(552, 100, "带标签：real + fake", "tl", "end")
    f.text(552, 212, "训练时只用 real", "tl hlt", "end")
    f.box(600, 30, 200, 72, "E5 · 线性探针", ["ridge 闭式解，按身份 5 折", "对照：随机初始化 ×8"])
    f.box(600, 178, 200, 112, "E6 · 贝叶斯头",
          ["BayesLinear 512→256 · GELU", "BayesLinear 256→64 · GELU", "BayesLinear 64→1",
           "ELBO，Pyro SVI"])
    f.line([(800, 66), (840, 66)])
    f.line([(800, 234), (840, 234)])
    f.box(840, 40, 130, 52, "探针 AUROC", ["表征好不好"])
    f.box(840, 208, 130, 52, "异常分", ["= −后验均值"])
    f.line([(905, 92), (905, 208)], cls="ln dash", head=False)
    f.text(897, 154, "对照判读（5.3 节）", "tl", "end")
    cap = ("E5 与 E6 都读取选定模型（E4 或 E4′ 的胜者）冻结后的 512 维视频特征（每个视频 8 个 clip 的特征取均值；"
           "每次前向最多送 4 个 clip，否则 DFD 整帧评测会超显存）。E5 用带标签的特征拟合线性探针，"
           "回答“表征本身能不能分开真假”；E6 只用真实视频训练贝叶斯头，异常分取后验均值的相反数。"
           "两者对照就是第 5.3 节的判读表。")
    return f, cap


# --------------------------------------------------------------------------
# F9 E7 backbone: MC3-18 vs R3D-18
# --------------------------------------------------------------------------


def fig_e7():
    f = Fig("e7", 1000, 262,
            "E7 选 MC3-18：它只在空间上降采样，32 个时间步保留到输出；R3D-18 会把时间压到 4 步")
    f.text(20, 24, "输入 [B,3,32,256,256] · CelebDF++ · 用 Kinetics 的均值方差归一化", "ts", "start")
    xs = [20 + 200 * i for i in range(5)]
    mc3 = [("stem", ["Conv3d (3,7,7)，空间 s2", "→ [64,32,128,128]"], "box"),
           ("layer1 · 3D 卷积", ["3×3×3，不降采样", "→ [64,32,128,128]"], "box"),
           ("layer2–4 · (1,3,3)", ["只降空间，T 不动", "→ [512,32,16,16]"], "box hl"),
           ("池化 (T,1,1)", ["每个时间步 512 维", "→ [B,32,512]"], "box"),
           ("TCN → 选定聚合 → Linear", ["与选定模型相同"], "box")]
    r3d = [("stem", ["Conv3d (3,7,7)，空间 s2", "→ [64,32,128,128]"], "box gh"),
           ("layer1 · 3D 卷积", ["3×3×3，不降采样", "→ [64,32,128,128]"], "box gh"),
           ("layer2–4 · 3×3×3", ["时间和空间一起降", ("→ [512,4,16,16]", "tm hlt")], "box gh"),
           ("池化 (T,1,1)", [("→ [B,4,512]", "tm hlt")], "box gh"),
           ("TCN 只剩 4 步", ["时序建模无从谈起"], "box gh")]
    f.text(20, 54, "MC3-18（选用）· 1170 万参数 · Kinetics-400 预训练", "th", "start")
    f.text(20, 166, "R3D-18（不用）· 3340 万参数 · 同样是 Kinetics-400 预训练", "th", "start")
    for (y, row) in [(64, mc3), (176, r3d)]:
        for i, (x, (t, s, c)) in enumerate(zip(xs, row)):
            f.box(x, y, 170, 62, t, s, cls=c)
            if i < 4:
                f.line([(x + 170, y + 31), (x + 200, y + 31)])
    cap = ("E7 为什么用 MC3-18 而不用 R3D-18（两者都是 torchvision 自带的 Kinetics-400 预训练模型，形状为本地实测）。"
           "MC3-18 只在 stem 和 layer1 做 3D 卷积，layer2–4 的卷积核是 (1,3,3)、只在空间上降采样，"
           "所以 32 个时间步一直保留到输出，可以直接接和选定模型完全相同的 TCN 与时间聚合；R3D-18 在 layer2–4 同时降时间，输出只剩 4 步。")
    return f, cap


# --------------------------------------------------------------------------
# F10 run order
# --------------------------------------------------------------------------


def fig_runorder():
    f = Fig("ro", 980, 262,
            "运行顺序：环境检查、P0.5 划分、捷径地板，两个数据集各走 E0 到 E4 和聚合系列，E7 等 E4′ 选定后再跑，最后测试")
    f.box(20, 112, 100, 56, "① 远程环境", ["路径/显存/读速"])
    f.line([(120, 140), (140, 140)])
    f.box(140, 112, 100, 56, "② P0.5 划分", ["本地生成后提交"])
    f.line([(240, 140), (260, 140)])
    f.box(260, 112, 100, 56, "③ E0S", ["捷径地板"])
    f.text(400, 22, "④ 阶梯（两个数据集各跑一遍）", "th", "start")
    f.text(400, 44, "CelebDF++", "ts", "start")
    f.text(400, 168, "DFD", "ts", "start")
    f.line([(360, 140), (380, 140), (380, 74), (400, 74)])
    f.line([(380, 140), (380, 198), (400, 198)])
    xs = [400 + 56 * i for i in range(5)]
    for y, skip in [(52, set()), (176, {2, 3})]:
        f.line([(400, y + 22), (780, y + 22)])
        for i, x in enumerate(xs):
            f.box(x, y, 46, 44, "E{}".format(i), [], cls="box gh" if i in skip else "box")
        f.box(684, y, 64, 44, "E4′", ["聚合 ×5"], cls="box gh" if y > 100 else "box")
    f.box(539, 124, 104, 34, "DFD 池化筛查", ["不训练"], cls="box frz")
    f.line([(591, 158), (591, 176)])
    f.text(560, 244, "E2、E3、E4′ 视时长可跳过", "ts", "middle")
    f.box(780, 4, 196, 38, "E7 · MC3-18（仅 CelebDF++）")
    f.line([(716, 52), (716, 23), (780, 23)])
    f.text(708, 38, "选定聚合后", "tl", "end")
    f.box(780, 50, 84, 48, "⑤ E5 · E6", ["探针 / BCNN"])
    f.box(780, 174, 84, 48, "⑤ E5 · E6", ["探针 / BCNN"])
    f.line([(864, 74), (882, 74), (882, 127), (900, 127)])
    f.line([(864, 198), (882, 198), (882, 153), (900, 153)])
    f.line([(938, 42), (938, 110)])
    f.box(900, 110, 76, 60, "⑥ 测试", ["只跑一次"])
    cap = ("运行顺序。①–③ 只做一次；④ 两个数据集各自沿 E0→E4 爬阶梯，再跑 E4′ 时间聚合系列"
           "（两轮多 seed，6.2 节；DFD 若单次训练超过约 24 小时，虚线框的 E2、E3、E4′ 可以跳过，归因在 CelebDF++ 上完成）；"
           "DFD 在 E3 之前先对已有检查点做一次不训练的池化筛查（4.3 节）；"
           "E7 用 E4′ 选定的聚合方式，所以要等 CelebDF++ 的 E4′ 出结果；"
           "⑤ 用各自选定的模型做 E5 探针和 E6 贝叶斯头；⑥ 封存的测试集只在最后评估一次。")
    return f, cap


# --------------------------------------------------------------------------
# F11 temporal aggregation options
# --------------------------------------------------------------------------


def fig_agg():
    f = Fig("ag", 980, 400,
            "五种时间聚合方式：前四种对每个时间步用同一套运算，Flatten 给每个时间位置一套独立权重")
    X = lambda k: 190 + 34 * k
    rows = [(20, "GAP（主线）", "逐步平均", "平均，权重都是 1/32", "0"),
            (90, "Max", "逐通道取最大", "每个通道取最大值", "0"),
            (160, "Attention", "按内容打分加权", "Σ αt · ht", "65,793"),
            (230, "Transformer CLS", "FTCN 的做法", "1 层自注意力，取 CLS", "2,120,192"),
            (300, "Flatten", "直接拼接", "Linear 16384→512", "8,389,120")]
    f.text(965, 12, "聚合部分参数量", "th", "end")
    widths = [1.2, 2.8, 1.0, 1.4, 3.4, 1.0, 1.8, 1.0]
    for top, name, desc, op, par in rows:
        f.text(20, top + 16, name, "th", "start")
        f.text(20, top + 32, desc, "ts", "start")
        for k in range(8):
            f.rect(X(k), top + 4, 22, 22, "box", rx=3)
        f.text(X(8) + 6, top + 20, "…", "t")
        if name == "Flatten":
            yb = top + 50
            for k in range(8):
                f.line([(X(k) + 11, top + 26), (X(k) + 11, yb)], head=False)
                f.rect(X(k) - 6, yb, 34, 16, "box hl", rx=0)
            f.text(511, yb, "[B,16384]", "tm")
            f.line([(X(7) + 34, yb + 8), (560, yb + 8)])
            f.box(560, yb - 12, 170, 40, op)
            f.text(745, yb + 12, "→ [B,512]", "tm", "start")
            f.text(965, yb + 12, par, "tm", "end")
            f.text(X(0) - 6, top + 86, "每个时间位置一套独立权重（橙色块）", "ts hlt", "start")
            continue
        yc = top + 50
        mx = X(3) + 28
        if name == "Transformer CLS":
            f.rect(X(0) - 42, top + 4, 22, 22, "box hl", rx=3)
            f.text(X(0) - 31, top + 19, "C", "ts")
            f.rect(X(0) - 48, top, 34 * 7 + 22 + 54, 30, "box gh", rx=5)
            f.line([(X(0) - 31, top + 26), (X(0) - 31, yc), (560, yc)])
        else:
            for k in range(8):
                f.line([(X(k) + 11, top + 26), (mx, yc)], head=False,
                       sw=(widths[k] if name == "Attention" else None))
            f.line([(mx, yc), (560, yc)])
        f.box(560, yc - 18, 170, 36, op)
        f.text(745, yc + 4, "→ [B,512]", "tm", "start")
        f.text(965, yc + 4, par, "tm", "end")
    cap = ("TCN 输出的 32 个时间步（这里画 8 个）怎样变成一个 512 维向量。前四种对每个时间步用同一套运算："
           "伪造痕迹出现在第 5 步还是第 21 步，结果一样；Attention 的线粗细表示权重 αt，由每一步的内容决定；"
           "Transformer 加一个 CLS 标记（C），让所有步两两互看后取 CLS 的输出。"
           "Flatten 给每个时间位置分配一套独立权重——而 clip 的起点是随机采的，“第 t 步”在不同样本之间缺少稳定含义，"
           "所以它更容易学到特定位置的捷径，中间向量 16384 维，比 V1 的 15488 维还大。"
           "参数受控的 Flatten-小（先把每步降到 64 维，1,081,920 参数）见 4.5 节的表。")
    return f, cap


# --------------------------------------------------------------------------
# F12 DFD face size vs pooling grid (to scale)
# --------------------------------------------------------------------------


def fig_dfdpool():
    f = Fig("dp", 980, 262,
            "按比例绘制：DFD 960×540 帧中中位大小的人脸只占 4×4 池化一个格子的 28%")
    sc = 0.3
    FW, FH = 960 * sc, 540 * sc
    fx, fy, fs = 300 * sc, 160 * sc, 94 * sc
    panels = [(20, 4, 4, "4×4 平均（主线）", "每步 512 维", "人脸占所在格 28%"),
              (346, 4, 7, "4×7 平均（A10）", "每步 896 维", "人脸占所在格 48%"),
              (672, 8, 14, "8×14 平均", "每步 3584 维", "人脸跨 2×2 个格")]
    y0 = 40
    for x0, rows, cols, title, dim, note in panels:
        cw, ch = FW / cols, FH / rows
        c0, c1 = int(fx // cw), int((fx + fs) // cw)
        r0, r1 = int(fy // ch), int((fy + fs) // ch)
        f.rect(x0, y0, round(FW, 1), round(FH, 1), "box", rx=2)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                f.rect(round(x0 + c * cw, 1), round(y0 + r * ch, 1),
                       round(cw, 1), round(ch, 1), "cell", rx=0)
        for c in range(1, cols):
            xx = round(x0 + c * cw, 1)
            f.line([(xx, y0), (xx, y0 + FH)], cls="ln faint", head=False)
        for r in range(1, rows):
            yy = round(y0 + r * ch, 1)
            f.line([(x0, yy), (x0 + FW, yy)], cls="ln faint", head=False)
        f.rect(round(x0 + fx, 1), round(y0 + fy, 1), round(fs, 1), round(fs, 1), "face", rx=2)
        f.text(x0, y0 - 10, title, "th", "start")
        f.text(x0 + FW / 2, y0 + FH + 22, dim, "tm")
        f.text(x0 + FW / 2, y0 + FH + 40, note, "ts hlt")
    cap = ("DFD 整帧里人脸有多小（按比例绘制，橙框为中位大小的人脸）。V1 实验 2 缓存的 dlib 人脸框中位宽 189 px（1920×1080），"
           "在送入网络的 960×540 帧里是 94 px，只占画面的 1.7%。4×4 平均池化后一个格子对应 240×135 px，"
           "人脸只占其中 28%（最小的 10% 视频只占 13%），其余全是背景，伪造痕迹在求平均时被背景稀释。"
           "改成 4×7 也只提到 48%；不稀释的办法是取最大值（max 池化，维度不变）或用更细的网格。"
           "CelebDF++ 是人脸裁剪，不存在这个问题。")
    return f, cap


FIGURES = {
    "overview": fig_overview,
    "preprocess": fig_preprocess,
    "timeaxis": fig_timeaxis,
    "ladder": fig_ladder,
    "tail": fig_tail,
    "e4": fig_e4,
    "tcn": fig_tcn,
    "phasebc": fig_phasebc,
    "e7": fig_e7,
    "runorder": fig_runorder,
    "agg": fig_agg,
    "dfdpool": fig_dfdpool,
}
