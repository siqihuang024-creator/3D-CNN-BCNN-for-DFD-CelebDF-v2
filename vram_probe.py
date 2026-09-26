# -*- coding: utf-8 -*-
"""Measure peak training VRAM of the real V2 network, one full step per setting.

Why this was rewritten: the first version carried a hand-written copy of the
architecture. A copy is only as good as the day it was written, and the point
of the probe is to decide a batch size for the model that actually trains, so
it now builds the extractor and head through build_feature_extractor and
DeterministicHead from the run's own config.

What it is for. DFD feeds whole 960x540 frames -- 7.9 times the pixels of the
CelebDF++ 256x256 face crop -- while the batch composition is the largest
single effect measured on the ladder so far (E1 to E2, +0.119 AUROC, from a
batch of one to four real plus four fake). Gradient accumulation does not help:
it matches the optimizer's effective batch and leaves batch normalisation
looking at the physical batch. So DFD has to reach a physical batch of 8, and
the only question is which input setting leaves room for it on the actual card.

The lever is the input, not the architecture. Decimation keeps every Nth pixel
on both axes, so step 3 gives 640x360, 44% of the pixels of step 2, without
changing a single layer -- unlike conv1 stride 2, which would make DFD a
different model from CelebDF++ and spoil the cross-dataset comparison. Note
that the largest single tensor is the gradient of conv1's output, which
backward needs whatever the forward did, so activation checkpointing is
expected to reproduce it rather than avoid it. That is an analysis, not a
measurement -- test it if batch 8 only just misses.

Peak is torch.cuda.max_memory_allocated above the resting allocation, i.e.
excluding the CUDA context (0.3-0.5 GB) and allocator fragmentation, which is
why the verdict is taken against a safety line below the card's capacity.

Usage:
    python vram_probe.py --config configs/v2/phase_a_dfd.yaml \
        --variant e2 e3 e4 --batch-sizes 4 8 --decimate-steps 2 3 \
        --output artifacts/v2/vram_probe_dfd.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.model import DeterministicHead, build_feature_extractor  # noqa: E402
from video_bcnn.utils import load_config, save_json  # noqa: E402

# The three rungs differ in what the head has to hold, so they are not
# interchangeable for a memory measurement.
VARIANTS = {
    "e2": {"spatial_output_size": 22, "temporal_head": "mean", "feature_dim": 32 * 22 * 22},
    "e3": {"spatial_output_size": 4, "temporal_head": "mean", "feature_dim": 512},
    "e4": {"spatial_output_size": 4, "temporal_head": "tcn", "feature_dim": 512},
}


def frame_size(config, decimate_step, source_height, source_width):
    """The height and width the network actually sees, per data.py.

    Only the face path resizes. In decimate mode input_resize and center_crop
    are never applied, which is the whole reason DFD is expensive.
    """
    mode = config["data"].get("frame_mode", "face")
    if mode == "face":
        crop = int(config["data"].get("center_crop", config["model"].get("center_crop", 256)))
        return crop, crop
    if mode == "decimate":
        step = int(decimate_step or config["data"].get("decimate_step", 2))
        if step < 1:
            raise ValueError("decimate_step must be positive.")
        # frame[::step] keeps indices 0, step, 2*step, ..., which is ceil, not
        # floor: 1079 rows at step 2 leave 540 rows, not 539. It only differs
        # when the resolution is not a multiple of the step, but the probe is
        # supposed to report the tensor the network will really see.
        return -(-source_height // step), -(-source_width // step)
    raise ValueError("Probe covers 'face' and 'decimate'; got {!r}.".format(mode))


def build(config, variant):
    model_config = copy.deepcopy(config["model"])
    model_config.update(VARIANTS[variant])
    extractor = build_feature_extractor(model_config).cuda()
    head = DeterministicHead(extractor.feature_dim,
                             model_config.get("deterministic_hidden_dim", 0),
                             model_config.get("dropout", 0.2)).cuda()
    return extractor, head


def probe(config, variant, batch, steps, height, width, amp, measured_steps=3):
    """Several full training steps; peak GB, or 'OOM'.

    Not one step, for two reasons. Adam allocates its moment buffers during the
    first step() and the GradScaler may skip that step entirely while it
    calibrates, so a single measured step under-reports the steady state. And
    cuDNN picks its algorithm, and its workspace, on the first call. So the
    first step is a warm-up whose peak is discarded and the maximum over the
    following steps is reported.

    Three numbers. peak_gb is the absolute high-water allocation -- weights,
    Adam state, the input and every activation -- which is what has to fit.
    reserved_gb is what the caching allocator held from the driver, always
    larger, and the honest figure against the card's capacity because
    fragmentation is what actually raises OOM. activation_gb is the part that
    scales with the batch and the frame, which says whether a smaller input
    would buy enough room.

    Building the model is inside the guard too: a batch large enough to fail
    can fail while the weights are being placed, and that is still an answer.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    extractor = head = optimizer = None
    try:
        extractor, head = build(config, variant)
        optimizer = torch.optim.Adam(
            list(extractor.parameters()) + list(head.parameters()), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        clips = torch.randn(batch, 3, steps, height, width, device="cuda")
        labels = torch.randint(0, 2, (batch,), device="cuda").float()

        def training_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = F.binary_cross_entropy_with_logits(head(extractor(clips)), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()

        training_step()                       # warm-up, peak discarded
        torch.cuda.reset_peak_memory_stats()
        resting = torch.cuda.memory_allocated()
        for _ in range(max(1, int(measured_steps))):
            training_step()
        peak = torch.cuda.max_memory_allocated()
        result = {"peak_gb": round(peak / 2 ** 30, 3),
                  "reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
                  "activation_gb": round((peak - resting) / 2 ** 30, 3),
                  "resting_gb": round(resting / 2 ** 30, 3),
                  "measured_steps": int(max(1, measured_steps))}
        del clips, labels
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result = {"peak_gb": "OOM", "reserved_gb": None, "activation_gb": None,
                  "resting_gb": None, "measured_steps": 0}
    finally:
        del extractor, head, optimizer
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", nargs="+", default=["e4"], choices=sorted(VARIANTS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--decimate-steps", type=int, nargs="+", default=None,
                        help="Ignored for face-crop configs. Defaults to the config value.")
    parser.add_argument("--temporal-steps", type=int, nargs="+", default=None)
    parser.add_argument("--conv1-strides", type=int, nargs="+", default=[1],
                        help="Stride 2 is the A12 fallback. It changes the architecture, "
                             "so use it only if the input lever is not enough.")
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--measured-steps", type=int, default=3,
                        help="Steps measured after the discarded warm-up step.")
    parser.add_argument("--safety-gb", type=float, default=None,
                        help="Verdict line. Defaults to the card's capacity minus 2 GB, "
                             "leaving room for the CUDA context and fragmentation.")
    parser.add_argument("--output", default="artifacts/v2/vram_probe.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This probe measures CUDA memory and needs a GPU.")

    config = load_config(args.config)
    properties = torch.cuda.get_device_properties(0)
    capacity = properties.total_memory / 2 ** 30
    safety = args.safety_gb if args.safety_gb is not None else max(1.0, capacity - 2.0)
    decimate_steps = args.decimate_steps or [config["data"].get("decimate_step", 2)]
    temporal_steps = args.temporal_steps or [config["model"].get("temporal_steps", 32)]
    amp = not args.no_amp

    report = {"config": str(Path(args.config).resolve()),
              "device_name": properties.name,
              "device_total_gb": round(capacity, 3),
              "safety_gb": round(safety, 3),
              "frame_mode": config["data"].get("frame_mode", "face"),
              "amp": amp, "torch": torch.__version__, "rows": [],
              "note": ("The verdict uses reserved_gb, which includes allocator "
                       "fragmentation, against safety_gb; peak_gb is the allocated "
                       "high-water mark and neither figure includes the CUDA context "
                       "(0.3-0.5 GB). A pass here is a necessary condition, not a "
                       "guarantee: synthetic inputs exercise no data pipeline, so "
                       "confirm with a short run on real DFD video before committing. "
                       "Gradient accumulation does not change what batch normalisation "
                       "sees, so only the physical batch here is the BN batch.")}
    for variant in args.variant:
        for step in decimate_steps:
            height, width = frame_size(config, step, args.source_height, args.source_width)
            for length in temporal_steps:
                for stride in args.conv1_strides:
                    current = copy.deepcopy(config)
                    current["model"]["temporal_steps"] = int(length)
                    current["model"]["conv1_spatial_stride"] = int(stride)
                    for batch in args.batch_sizes:
                        measured = probe(current, variant, batch, int(length),
                                         height, width, amp, args.measured_steps)
                        peak = measured["peak_gb"]
                        reserved = measured["reserved_gb"]
                        # Judge on reserved, not allocated: fragmentation is what
                        # raises OOM in a real run, and it is always the larger.
                        decisive = reserved if isinstance(reserved, float) else peak
                        row = dict(measured, variant=variant, batch=batch,
                                   decimate_step=int(step), temporal_steps=int(length),
                                   conv1_spatial_stride=int(stride),
                                   height=height, width=width,
                                   fits=(isinstance(decisive, float)
                                         and decisive <= safety))
                        report["rows"].append(row)
                        print("%-3s B=%-2d step=%d %4dx%-4d T=%-2d s%d -> peak %s / "
                              "reserved %s GB (activations %s) %s" % (
                                  variant, batch, step, width, height, length, stride,
                                  peak, reserved, measured["activation_gb"],
                                  "OK" if row["fits"] else "OVER"), flush=True)
    save_json(args.output, report)
    print("\nwrote {}".format(args.output))
    print(json.dumps({"device": report["device_name"],
                      "total_gb": report["device_total_gb"],
                      "safety_gb": report["safety_gb"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
