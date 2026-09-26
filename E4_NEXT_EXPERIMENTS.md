# E4 follow-up protocol (2026-09-26)

## Interpretation and order

Keep E2/E3/E4 seed42, 60-epoch results as the fixed-budget comparison.
E4 improves over E3 in the identity-paired analysis but does not demonstrate
superiority or equivalence to E2. A single training seed is not a replication.
Do not attribute the improvement to order before the controls below.

1. Pre-TCN order control: cache the unchanged trunk sequences once; evaluate
   ordered, ten reproducible shuffles, reversed, non-identity block4/8/16, and
   TCN bypass with the same checkpoint and classifier.
2. Raw-frame shuffle of E2/E3/E4: auxiliary whole-model diagnostic, not a
   TCN-only intervention. Run sequentially on one GPU.
3. E4_long: independent 90-epoch run in a NEW directory, not exact continuation.
4. E4-prime aggregation: logically independent of order sensitivity. Reduced
   by agreement to attention and flatten, the two that ask different questions
   from GAP (content-adaptive weighting, and explicitly position-preserving).
   Flatten is Linear(32x512, 512), 8.39M parameters against GAP's zero, so it
   is a stress test biased against the null: losing is strong evidence, winning
   is confounded with capacity and would need flatten64 to attribute.
   Implemented choices are gap/max/attention/cls/flatten/flatten64; top-k is
   NOT implemented and should not be listed as an available CLI option.
5. DFD, raised in priority: the order-control null is only established on
   CelebDFv3 and may be a property of that dataset. See Phase 2 below.

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

## Remote first experiment (bash, py312 environment)

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

## Raw-frame auxiliary control

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

## Phase 1 mechanism checks (after the order control)

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

## Phase 2 DFD feasibility (before any long DFD run)

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
forward did, and recomputation reproduces it rather than avoiding it. That is
an analysis of where the memory goes, not a measurement; if the probe shows
batch 8 just missing at step 2, it is worth testing before being dismissed.

```bash
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
extrapolated from FLOPs and has to be measured:

```bash
python -u scripts/benchmark_loader.py --config configs/v2/phase_a_dfd.yaml \
  --manifest artifacts/manifests/combined_manifest_p05.csv
python -u pretrain_extractor.py --config configs/v2/phase_a_dfd.yaml \
  --manifest artifacts/manifests/combined_manifest_p05.csv \
  --experiment E2_dfd --seed 42 --max-epochs 2 --run-suffix timing
```

Watch nvidia-smi during the second command. High utilisation means compute and
the input lever applies; low utilisation with busy CPUs means decoding, where
smaller frames still help but conv1 stride would not.

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
