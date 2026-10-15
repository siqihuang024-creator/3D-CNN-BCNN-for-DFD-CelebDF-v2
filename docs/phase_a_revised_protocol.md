# Revised CelebDFv3 Phase-A protocol

This is a new experiment protocol. The original P0.5 manifest, E2 configuration,
checkpoints and results remain unchanged. Both E2 and E2-interleave must be
retrained under this protocol before they are compared with each other.

## Data and sampling

- Start from `artifacts/manifests/combined_manifest_p05.csv` (seed 42).
- Keep all 623 real training videos. Select 1,246 fake training videos without
  replacement, preserving the original fake-method proportions and preferring
  distinct real source clips. The resulting 22 fake methods and counts are
  recorded in the adjacent manifest summary JSON.
- Merge the original validation and test source families into one development
  validation set: 267 real and 16,236 fake videos. No independent test set
  remains in this CelebDFv3 protocol. No video path or original source-video
  family appears in both training and validation. Identities may overlap.
- Training uses random clip start positions, with **32 consecutive frames** per
  clip. Model-selection and full-validation clips also use 32 consecutive
  frames, at fixed evenly spaced start positions. Strict clip preflight rejects
  source videos with fewer than 32 frames rather than repeating the last frame.
- Physical batch size is 8, plain BCE loss with no class weighting. Each
  epoch is **one pass over the whole training pool** (623 real + 1,246 fake =
  1,869 videos, 234 batches): the pool is shuffled and cut into batches of 8;
  the last batch is topped up with 1 real and 2 fake videos drawn at random
  from the rest of the pool, so only those 3 repeat within an epoch. Each
  epoch has 624 real and 1,248 fake clips. A new clip start is drawn every
  time a video is used.
- No data augmentation: horizontal flipping is off (`horizontal_flip_probability: 0.0`),
  so the random clip start is the only source of variation between epochs.
- A batch's class mix is whatever the shuffle gives, except that a batch with
  no real (or no fake) clip swaps one clip with a batch holding at least two of
  that class. Over 60 epochs at seed 42 the real clips per batch come out as
  1: 20.1%, 2: 27.9%, 3: 27.5%, 4: 16.4%, 5: 6.4%, 6: 1.5%, 7: 0.2%. Each
  epoch's histogram is written to `history` as `batch_real_count_histogram`;
  if 1-real batches turn out to disturb BatchNorm, raise the floor to two.
- Checkpoint selection uses all 267 real validation videos plus a fixed,
  method-stratified set of 1,000 fakes, eight clips per video -- the same as
  full validation, so the saved threshold applies to identically computed scores. Full validation
  scores all 16,503 videos using eight clips per video after training.

## Build and run on the GPU host

```bash
python scripts/build_phase_a_revised_manifest.py \
  --input artifacts/manifests/combined_manifest_p05.csv \
  --output artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
  --dataset CelebDFv3 --fake-per-real 2 --seed 42

# Run each model independently with the same revised config and manifest.
python pretrain_extractor.py \
  --config configs/v2/phase_a_celeb_revised.yaml \
  --manifest artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
  --experiment E2_celeb --seed 42

python pretrain_extractor.py \
  --config configs/v2/phase_a_celeb_revised.yaml \
  --manifest artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
  --experiment E2_interleave --seed 42

# After training: full development-validation scoring (all 16,503 videos).
python evaluate_3d_bcnn.py   --config configs/v2/phase_a_celeb_revised.yaml   --manifest artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv   --checkpoint artifacts/v2_revised/celebdfv3_e2_celeb_gap_seed42/checkpoints/best.pt   --split val --expected-videos 16503 --bootstrap-draws 2000 --experiment E2_celeb
```

Report accuracy, balanced accuracy, TPR, TNR and the confusion matrix at the
saved selection-set threshold (stating the observed FPR) next to AUROC and both
APs. The development set is 98.4% fake, so all-fake already scores 98.4%
accuracy; accuracy is never read on its own.

The model-selection runs use separate paths under `artifacts/v2_revised/`.
Before a full run, cache face boxes for the newly merged validation rows and
scan the revised manifest on the GPU host before a one-epoch feasibility/VRAM
run:

```bash
python scripts/scan_decode_health.py \
  --manifest artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
  --datasets CelebDFv3 --splits train val --min-frames 32 --workers 4 \
  --output artifacts/v2_revised/decode_health.csv
```

Inspect all non-`ok` rows and every `decoded_frames < 32` row. The scan
fully decodes each video, so it can take substantial time; it does not edit the
manifest. The pool-pass face-cache preflight also checks the entire merged
validation set, but cache `frame_count` is only container metadata and cannot
replace the full-decode scan. The local copy of
`REAL/Real_WEB/id27_0005.mp4` declares and decodes only one frame. Check the
GPU-host copy: if it is also short, repair it or replace this training video
with another real training video from the original P0.5 split and regenerate
the revised manifest and caches. Do not silently drop it, as that would change
the agreed 623-real / 1,246-fake pool. The old 8,205-video results use a
different data and clip protocol and are not directly comparable to these
future 16,503-video results.
