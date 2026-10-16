# Revised CelebDFv3 Phase-A protocol

This is a new experiment protocol. The original P0.5 manifest, E2 configuration,
checkpoints and results remain unchanged. Both E2 and E2-interleave must be
retrained under this protocol before they are compared with each other.

## Data and sampling

- Start from `artifacts/manifests/combined_manifest_p05.csv` (seed 42).
- The GPU-host decode scan found that the sole copy of
  `REAL/Real_WEB/id27_0005.mp4` contains only one decoded frame. Explicitly
  exclude this unusable training real video when building the manifest.
  Keep the remaining 622 real training videos and select 1,244 fake training
  videos without replacement (1:2), preserving the original fake-method
  proportions and preferring distinct real source clips. The excluded path,
  fake methods, and counts are recorded in the adjacent manifest summary JSON.
  The original P0.5 manifest and source video are not changed.
- Merge the original validation and test source families into one development
  validation set: 267 real and 16,236 fake videos. No independent test set
  remains in this CelebDFv3 protocol. No video path or original source-video
  family appears in both training and validation. Identities may overlap.
- Training uses random clip start positions, with **32 consecutive frames** per
  clip. Model-selection and full-validation clips also use 32 consecutive
  frames, at fixed evenly spaced start positions. Strict clip preflight rejects
  source videos with fewer than 32 frames rather than repeating the last frame.
- Physical batch size is 8, plain BCE loss with no class weighting. Each
  epoch is **one pass over the whole training pool** (622 real + 1,244 fake =
  1,866 videos, 234 batches): the pool is shuffled and cut into batches of 8;
  the last batch is topped up with 2 real and 4 fake videos drawn at random
  from the rest of the pool, so only those 6 repeat within an epoch. Each
  epoch has 624 real and 1,248 fake clips. A new clip start is drawn every
  time a video is used.
- No data augmentation: horizontal flipping is off (`horizontal_flip_probability: 0.0`),
  so the random clip start is the only source of variation between epochs.
- A batch's class mix is whatever the shuffle gives, except that a batch with
  no real (or no fake) clip swaps one clip with a batch holding at least two of
  that class. Each epoch's observed real-clip-count histogram is written to
  `history` as `batch_real_count_histogram`;
  if 1-real batches turn out to disturb BatchNorm, raise the floor to two.
- Checkpoint selection uses all 267 real validation videos plus a fixed,
  method-stratified set of 1,000 fakes, eight clips per video -- the same as
  full validation, so the saved threshold applies to identically computed scores. Full validation
  scores all 16,503 videos using eight clips per video after training.
- The full development set is **not** 1:2: it contains 267 real and 16,236
  fake videos (about 1:60.8). The evaluator also writes a fixed-ratio report
  using all 267 real and 534 method-stratified fakes (1:2), selected with seed
  43 from fakes outside the 1,000-video checkpoint-selection fake subset.
  Both model variants use the same manifest and seeds, so they use the same
  801 videos. This report is computed from the already scored full set; it
  requires no second GPU pass. The 267 real videos still overlap checkpoint
  selection and threshold calibration, so it remains a development result,
  not an independent test.

## Build and run on the GPU host

```bash
python scripts/build_phase_a_revised_manifest.py \
  --input artifacts/manifests/combined_manifest_p05.csv \
  --output artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
  --dataset CelebDFv3 --fake-per-real 2 --seed 42 \
  --exclude-train-real-path REAL/Real_WEB/id27_0005.mp4

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
python evaluate_3d_bcnn.py \
  --config configs/v2/phase_a_celeb_revised.yaml \
  --manifest artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
  --checkpoint artifacts/v2_revised/celebdfv3_e2_celeb_gap_seed42/checkpoints/best.pt \
  --split val --expected-videos 16503 --bootstrap-draws 2000 \
  --experiment E2_celeb

python evaluate_3d_bcnn.py \
  --config configs/v2/phase_a_celeb_revised.yaml \
  --manifest artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
  --checkpoint artifacts/v2_revised/celebdfv3_e2_interleave_gap_seed42/checkpoints/best.pt \
  --split val --expected-videos 16503 --bootstrap-draws 2000 \
  --experiment E2_interleave
```

The evaluator writes both `full_val.json` and `full_val_1to2.json`,
with corresponding video-score CSVs. **Both reports calculate accuracy and a
confusion matrix** at the same saved selection-set threshold, alongside AUROC,
both APs, balanced accuracy, TPR, TNR and observed FPR. The complete
16,503-video development set is the primary evaluation scope; its accuracy is
`(TP + TN) / 16,503` and must be reported. The fixed 1:2 subset is a
supplementary prevalence check, not a replacement for full-set accuracy.
The full set is 98.4% fake, so an all-fake classifier already scores 98.4%
raw accuracy there. On the 1:2 subset, an all-fake classifier scores 66.7%.
Interpret both accuracy values with their class proportions, TPR, TNR and
balanced accuracy. The fixed-ratio report is descriptive development
validation because its real videos were also used for model selection and
threshold calibration.

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
replace the full-decode scan. The local and GPU-host copies of
`REAL/Real_WEB/id27_0005.mp4` both decode only one frame, and the GPU host
has no second copy in `/root/Datasets`. The manifest command above explicitly
excludes it; verify the summary says 622 training real and 1,244 training
fake. Rerun the scan on this regenerated manifest before training, since
changing the fake quota can change which fake videos are selected. The old
8,205-video results use a
different data and clip protocol and are not directly comparable to these
future 16,503-video results.
