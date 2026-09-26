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
4. E4-prime aggregation: logically independent of order sensitivity. Finalize
   candidates/budgets/seeds before spending the larger training budget. Current
   implemented choices are gap/max/attention/cls/flatten/flatten64; top-k is NOT
   implemented, and should not be listed as an available CLI option.

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
variability. The cache is RAM-only, not persistent across restarts; inference
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
