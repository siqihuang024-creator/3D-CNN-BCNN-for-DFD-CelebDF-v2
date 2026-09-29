# E4 follow-up protocol (updated 2026-09-29)

## What is settled, and what runs next

Keep E2/E3/E4 seed42, 60-epoch results as the fixed-budget comparison. A single
training seed is not a replication.

Settled on CelebDFv3, from the order control and the three mechanism checks:

- The TCN restores what the 512-d compression cost and does not improve on the
  uncompressed baseline: E2 0.8058, E3 0.7819, E4 0.8064, so E4-E3 = +0.0245
  while E4-E2 = +0.0006. Do not quote the first figure without the second.
- Shuffling, reversing and block-shuffling the pre-TCN sequence move individual
  video scores substantially (Pearson 0.9976, MAE 13% of the score standard
  deviation, about 200x the numerical noise floor) and do not move AUROC. The
  representation responds to order; the response is not discriminative.
- No centre-tap dominance: shares 0.333/0.334 against a uniform 0.333. That
  rules out a trivial pointwise degeneration and nothing more.
- Per-position aggregation inside a clip: mean 0.8064 beats max 0.7951 and top2
  0.7974 with intervals excluding zero. No evidence that the cues are
  concentrated in a few temporal positions.
- The apparent class asymmetry in shuffle sensitivity was substantially
  attenuated after score matching: +62.7% uncontrolled (normalised 1.627)
  against +4.8% at matched scores (1.048, 95% CI 0.895 to 1.183), with the
  matched AUROC at 0.534 (0.450 to 0.594). No reliable class difference was
  detected once the score level was controlled. That is attenuation, not
  elimination -- the interval still admits about 18%, so this does not
  establish that the raw asymmetry was entirely a score-level effect. The
  headline figure of an earlier run, 2.374, came from a mean of per-real ratios
  and is a property of that estimator rather than of the data. The corrected
  run is results/v2/E4_shuffle_sensitivity_audit_v2.json.

  TODO on the remote, not yet done: rename the superseded report so its
  filename says so, because 2.374 is the number a reader would otherwise lift
  out of it months from now.

      mv results/v2/E4_shuffle_sensitivity_audit.json \
         results/v2/E4_shuffle_sensitivity_audit_SUPERSEDED_biased_estimator.json

With that, the CelebDFv3 E4 phase is closed by agreement. No further
CelebDF-side diagnostics are to be added before the DFD work below, whatever
this last result had shown.

Already done, kept here only so the record reads in order: the pre-TCN order
control, the three mechanism checks, and the score-matched sensitivity audit
including its real/fake split. Their commands are in the sections below, which
are history rather than instructions.

Order of work from here, starting at 1. Earlier drafts of this file led with
the raw-frame shuffle and E4_long, and that order is superseded.

1. DFD feasibility profiling: GPU, whether physical batch 8 fits, peak VRAM,
   train seconds per epoch separately from validation, and the loader wait
   fraction. One to two epochs, no long run. See Phase 2.
2. DFD main chain under one fixed protocol: E2-DFD, pool screen, E3-DFD,
   E4-DFD, then the same pre-TCN order control on E4-DFD.
3. E4-prime on CelebDFv3, reduced by agreement to attention and flatten, the
   two that ask different questions from GAP (content-adaptive weighting, and
   explicitly position-preserving). Flatten is Linear(32x512, 512), 8.39M
   parameters against GAP's zero, so it is a stress test biased against the
   null: losing is strong evidence, winning is confounded with capacity and
   would need flatten64 to attribute. Implemented choices are
   gap/max/attention/cls/flatten/flatten64; top-k is NOT implemented and should
   not be listed as an available CLI option. Max is not scheduled: the
   post-hoc screen gives no reason to spend the training budget on it, which is
   a statement about this checkpoint and this screen, not a proof that a model
   trained with max pooling from the start would be worse.
4. Raw-frame shuffle of E2/E3/E4: auxiliary whole-model diagnostic, not a
   TCN-only intervention. Run sequentially on one GPU.
5. E4_long: independent 90-epoch run in a NEW directory, not a continuation.
   Convergence and fixed-budget fairness appendix, not a temporal experiment.

Corrections to the discussion:

- The trunk sees 7 raw frames; the TCN sees 7 cached feature positions
  (kernel3, dilations1/2); the combined raw-frame receptive field is 13.
  Block8 can preserve some complete TCN windows; it is not shorter than the
  TCN's feature-space receptive field. Block16 at T32 always swaps the halves
  in this protocol (identity block permutations are excluded).
- E2's actual Phase-A config has deterministic_hidden_dim=0. Its classifier
  has 15489 parameters, NOT a 7.9M hidden FC head. E4 is not a parameter-count
  reduction over the actual E2 Phase-A model.
- Little shuffle drop/CI crossing zero does NOT prove order independence or
  equivalence. Cached features already encode trunk temporal context. Bypass
  is a fixed-checkpoint distribution-shift ablation, not a retrained capacity
  control; it cannot decompose E4-minus-E3 gains into causal percentages.
- The frozen nine-method shortcut-weak subset remains a sensitivity analysis,
  not an independent shortcut-free test set.

## History: the CelebDFv3 order control (completed, kept for reproduction)

This section and the Phase 1 mechanism checks below it record work that is
finished; the raw-frame control between them has not been run and is item 4.
They are here so the finished runs can be reproduced, not as the next thing to
launch -- the current first step is Phase 2, DFD feasibility profiling.

After pushing the reviewed code and pulling it on the server:

```bash
cd /root/3D-CNN-BCNN-for-DFD-CelebDF-v2
source remote_env.sh
mkdir -p logs
tmux new -s e4_order
```

Inside tmux, source the environment if it was not inherited, then:

```bash
set -o pipefail
python -u scripts/temporal_order_control.py \
  --config configs/v2/phase_a_celeb.yaml \
  --manifest artifacts/manifests/combined_manifest_p05.csv \
  --checkpoint artifacts/v2/celebdfv3_e4_gap_seed42/checkpoints/best.pt \
  --split val --expected-videos 8205 --shuffle-seeds 10 \
  --block-widths 4 8 16 --draws 2000 \
  --sequence-cache artifacts/v2/e4_val_sequences.pt \
  --output results/v2/E4_temporal_order_control.json \
  2>&1 | tee logs/E4_temporal_order_control.log
```

The ordered full_val_video_scores.csv in the E4 run's reports folder is required.
The script checks identical IDs/labels and every ordered score (absolute
tolerance 1e-4) before evaluating perturbations. If the check fails, investigate
precision/config/data differences; do not increase tolerance just to pass.

Artifacts: JSON report, matching *_video_scores.csv with every condition,
and terminal log. Individual deltas are condition-minus-ordered (negative means
a drop); shuffle_summary.mean_drop and its paired cluster interval are
ordered-minus-mean-of-per-seed metrics (positive means a drop). This is NOT AUC
of the averaged score vector. Bootstrap CIs describe cluster sampling with the
ten perturbation seeds fixed; shuffle std separately describes permutation
variability. With --sequence-cache the pre-TCN sequences are written to disk (about 2 GB
for full validation) after the ordered branch has matched full-val, so the
mechanism checks below reuse this trunk pass instead of repeating it;
without the flag the cache is RAM-only. Inference
preserves the AMP dtype and chunking rather than forcibly quantising to FP16.

## Raw-frame auxiliary control (item 4, not yet run)

```bash
for E in e2 e3 e4; do
  python -u evaluate_3d_bcnn.py \
    --config configs/v2/phase_a_celeb.yaml \
    --manifest artifacts/manifests/combined_manifest_p05.csv \
    --checkpoint artifacts/v2/celebdfv3_${E}_gap_seed42/checkpoints/best.pt \
    --split val --expected-videos 8205 --shuffle-frames \
    --bootstrap-draws 2000 --experiment ${E^^} \
    2>&1 | tee logs/${E}_raw_frame_shuffle.log || break
done
```

Use pipefail as above. Perturbed reports have distinct names and do not overwrite
the ordered legacy val_scores.csv. Compare ordered and perturbed video CSVs
with compare_experiments.py using identity-paired clustering.

## History: Phase 1 mechanism checks (completed, kept for reproduction)

The order control settled that permuting the pre-TCN sequence does not move
AUROC. It did not settle why, and three post-hoc checks separate the accounts
without retraining anything:

```bash
set -o pipefail
python -u scripts/e4_mechanism_checks.py \
  --checkpoint artifacts/v2/celebdfv3_e4_gap_seed42/checkpoints/best.pt \
  --config configs/v2/phase_a_celeb.yaml \
  --manifest artifacts/manifests/combined_manifest_p05.csv \
  --split val --expected-videos 8205 --draws 2000 \
  --order-control-scores results/v2/E4_temporal_order_control_video_scores.csv \
  --sequence-cache artifacts/v2/e4_val_sequences.pt \
  --output results/v2/E4_mechanism_checks.json \
  2>&1 | tee logs/E4_mechanism_checks.log
```

A. Score agreement, seconds, reads only the order control's CSV. A correlation
near one with a negligible MAE means the detector is close to permutation
invariant; a visibly lower correlation with unchanged AUROC means order moves
the score without moving the real/fake ranking. Only the first licenses the
sentence "the TCN does not use order", so this decides the wording.

B. Temporal tap norms, seconds, reads only the checkpoint. Reported beside a
fresh initialisation of the same architecture, because the uniform share is
1/3 by construction. Circumstantial in both directions, as agreed.

C. Per-position aggregation screen. Because the aggregator is GAP and the
classifier is a single Linear(512,1), the clip logit is exactly the mean of the
per-position logits, so max/top-k/min over the 32 positions can be scored on
the trained decision function without feeding the classifier features from a
distribution it never saw, which is the objection that makes tcn_bypass
uninterpretable. The statistic itself does shift -- a max sits above a mean by
construction -- but AUROC is rank-based, so the comparison is unaffected while
any thresholded reading would need recalibrating. The script enforces both
conditions rather than assuming them, measures the decomposition error on real
data, and requires the mean branch to reproduce full_val_video_scores.csv.

This replaces the earlier idea of re-evaluating at 16 or 32 clips per video.
Evaluation clips are evenly spaced starts and each spans 32 frames at stride 2,
i.e. 64 raw frames, so more clips raise sampling density without resolving
anything below that window. The per-position screen asks the sparsity question
at the scale it was meant for, and at a fraction of the cost.

Precision. Three separate checks, deliberately not merged, because two remote
attempts each failed on a different one of them:

  decomposition -- the screen's own float32 identity, per clip, which is the
  number the aggregation comparison rests on;
  linear_path_equivalence -- the published-precision TCN output widened to
  float32, both orderings of the linear map compared on that one tensor, which
  is what tells a rounding effect apart from an implementation error;
  numerics -- how far the screen's precision moves a video score against the
  published path, which is the floor under any AUROC difference read here.

The screen runs in float32 while the published evaluation ran under autocast,
and the two are deliberately separated. The first remote
attempt stopped here: the two paths disagreed by 4.06e-03 on the worst of
65,640 clips against a 1e-3 gate. That was the gate being wrong, not the
model. The paths differ only in the order of a linear map and a mean, so they
are equal in real arithmetic; this run's clip logits reach 13.7, where a single
fp16 step is already 1.3e-2, so the observed gap is about a third of one unit
in the last place. Measured on the same architecture locally: float32 1.9e-06,
autocast 3.2e-03 at a slightly smaller logit scale. Running the screen in
float32 also keeps fp16 noise from deciding which of 32 positions wins a max.
The branch that reproduces the published scores is temporal_order_control's
score() itself, called rather than reimplemented. The second remote attempt
failed on exactly that distinction: a rewrite that is identical in real
arithmetic still landed 2.44e-04 -- one fp16 ulp -- from
full_val_video_scores.csv, because the shape of a chunk decides which GEMM
kernel runs and therefore how it rounds. The same rewrite is bit-identical to
score() on an RTX 3060, so the effect is card-specific and cannot be found by
local testing; the standing rule is that a published number is reproduced by
calling the code that produced it.

It needs one trunk pass unless a cache exists. Pass --sequence-cache to the
order control to write it there, or to this script to have it written on the
first run; roughly 2 GB for CelebDF++ full validation. A cache carries no
guarantee of its own, which is why the mean branch is always re-verified
against the published scores.

Outcome: if max/top-k rank no better than mean, then for this checkpoint,
under this screen, the evidence is not concentrated in a few positions, and
E4-prime Max is not worth 5.5 hours. That is a statement about a model trained
and selected under the mean protocol, not about the sparse-anomaly account in
general -- training with max pooling could still find a different solution. If
they rank better, that run becomes justified.

## Phase 2 DFD feasibility -- THE CURRENT FIRST STEP (before any long DFD run)

DFD is 960x540 whole frames -- decimate mode never applies input_resize, so the
256 in the config is a red herring -- which is 7.9 times the pixels of the
CelebDF++ face crop. Two consequences, and neither may be guessed at:

Batch. DFD now trains at physical batch 8 (4 real + 4 fake), like CelebDF++,
not 4 with accumulation 2. Accumulation matches the optimizer's effective batch
and leaves batch normalisation looking at the physical batch, and BN
composition is the largest single effect on this ladder (E1 to E2, +0.119).
If 8 does not fit, raise decimate_step to 3 -- keeping every third pixel
instead of every second, so 640x360, 44% of the pixels, no layer changed --
rather than lowering the batch or enabling conv1 stride 2, which would make DFD
a different architecture from CelebDF++. Whichever step the probe selects is
then fixed for the whole DFD chain (E2, E3, E4 and the order control); a run at
a different input scale is not comparable with the others.

Activation checkpointing is expected not to help, because the largest single
tensor is the gradient of conv1's output, which backward needs whatever the
forward did, and recomputation reproduces it rather than avoiding it -- for
conv1 it should make the peak worse, by materialising that output again at the
moment its gradient is alive. An earlier version of this plan proposed
checkpointing stage 1. That proposal is withdrawn and should not be
reintroduced without a measurement showing the peak actually falls. It remains
an analysis of where the memory goes rather than a measurement, so if the probe
shows batch 8 only just missing at step 2, test it before dismissing it.

Resizing the whole frame to 256x256 while keeping whole-frame mode is not an
open option either. That is the letterbox path, and it is the configuration
that failed: cv2's INTER_AREA is a low-pass filter and averaged 4.6 input
pixels into each output pixel on the DFD run that failed, while manipulation
traces are high frequency, so the resize discarded the evidence before conv1
saw it (the comment in src/video_bcnn/data.py records this). Decimation is the
fix precisely because it slices rather than filters, which is why a larger
decimate_step stays inside that rationale and a resize does not.

Source the environment first. The configs carry this machine's Windows dataset
paths, and remote_env.sh is what replaces them with DFD_ROOT and
CELEBDFV3_ROOT. The memory probe runs on synthetic tensors and would not
notice, which is the trap: it would pass, and then the loader benchmark and the
timing run would fail on paths that do not exist.

```bash
cd /root/3D-CNN-BCNN-for-DFD-CelebDF-v2
source remote_env.sh
set -o pipefail

python -u vram_probe.py --config configs/v2/phase_a_dfd.yaml \
  --variant e2 e3 e4 --batch-sizes 8 --decimate-steps 2 3 \
  --output artifacts/v2/vram_probe_dfd.json
```

Runtime. The published estimate of 40 hours per DFD run assumed the GPU was the
bottleneck. The measured CelebDF++ E2 log is 250 iterations in 1:40, about
3 TFLOP/s including backward, which is under a tenth of the card's half
precision peak. That is consistent with an input-bound loop rather than a
compute-bound one, but it is indirect -- small kernels, Python overhead and
normalisation all depress the same number -- so it is a hypothesis for the
timing run to confirm, not a finding. Either way DFD's wall clock cannot be
extrapolated from FLOPs and has to be measured.

STOP HERE AND READ THE PROBE FIRST. The probe sweeps decimate_step 2 and 3, but
the two commands below do not: neither benchmark_loader.py nor
pretrain_extractor.py takes a decimate_step argument, so both read whatever
configs/v2/phase_a_dfd.yaml says, which is 2. If batch 8 only fits at step 3,
running them as written measures an input scale that will not be used, and the
timing is then a projection for the wrong experiment.

So if the probe selects step 3, edit the config before continuing:

```yaml
# configs/v2/phase_a_dfd.yaml
data:
  decimate_step: 3
```

There is deliberately no command-line override for it. The whole DFD chain --
E2, the pool screen, E3, E4 and the order control -- has to share one input
scale to be comparable, and a flag makes it easy for one run to diverge from
the rest. Keeping it in the config means one edit, recorded in git, that every
later command inherits.

```bash
python -u scripts/benchmark_loader.py --config configs/v2/phase_a_dfd.yaml \
  --manifest artifacts/manifests/combined_manifest_p05.csv
python -u pretrain_extractor.py --config configs/v2/phase_a_dfd.yaml \
  --manifest artifacts/manifests/combined_manifest_p05.csv \
  --experiment E2_dfd --seed 42 --max-epochs 2 --run-suffix timing \
  --profile-input-pipeline
```

Every epoch already records epoch_duration_seconds, and now train_seconds and
validation_seconds separately, alongside peak_vram_bytes and the GPU name.
--profile-input-pipeline adds the one that was missing:
input_pipeline.data_wait_fraction, the share of the TRAINING LOOP spent waiting
for the loader rather than computing. Its denominator is data_wait plus
compute, which is the training loop only -- validation is not in it, and it is
not a fraction of epoch_duration_seconds. Project a 60-epoch cost from
train_seconds and validation_seconds, and read the bottleneck from the
fraction. That is what decides the strategy, and FLOPs cannot answer it: a run
starved of data gains nothing from fewer pixels. Near zero means compute, and
the input scale is the lever. Near one means the training loop is waiting on
the input pipeline -- and that is as far as this number goes. Which part of the
pipeline is responsible, decode or disk or the preprocessing around them, it
does not say; pair it with benchmark_loader.py, whose worker sweep separates a
pipeline that scales with workers from one that does not. Prescribing "add
workers" from this fraction alone would be guessing. It costs a CUDA sync per
step, so it stays off for any run that will be reported.

DFD chain once the timing is known: E2-DFD, then the pooling screen S (it needs
a DFD-trained mean-head checkpoint and there is none yet, so E2 cannot be
skipped), then E3-DFD, E4-DFD, then the same pre-TCN order control on E4-DFD.
Note the power limit up front: DFD validation is 55 real and 472 fake videos in
25 identity clusters against CelebDF++'s 8205 in 94, so a DFD null excludes
only a large order effect and must be written as "no large effect was detected
in the available sample".

Face cropping DFD is not an open question: V1 experiment 2 measured validation
AUROC falling to 0.34, and dlib failed to detect a face on 50.3% of frames.
Whole-frame input stays, and the cross-dataset limitation -- aligned face crops
on CelebDF++ against whole frames on DFD, so absolute AUROCs are not comparable
-- is recorded rather than engineered away.

## Optional independent long-budget run (not resume)

```bash
python -u pretrain_extractor.py \
  --config configs/v2/phase_a_celeb.yaml \
  --manifest artifacts/manifests/combined_manifest_p05.csv \
  --experiment E4_gap --seed 42 --max-epochs 90 --run-suffix E4_long
```

This starts at epoch1 in artifacts/v2/celebdfv3_e4_long_seed42 and refuses to
overwrite existing checkpoints/history. Old E4 checkpoints contain model
weights and epoch but no optimizer/scheduler/scaler/RNG state, so strict
epoch60-to90 resume is unavailable. Keep this convergence analysis separate
from the original E4_60 result.
