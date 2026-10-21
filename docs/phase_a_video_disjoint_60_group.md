# CelebDF++ Phase A: video-disjoint 60-group pilot protocol

This protocol keeps the original CelebDF train/validation/test assignments.
Its enforced leakage rule is exact video-path disjointness: a video cannot
appear in more than one split. The builder imposes no additional source-family
or identity constraint. An audit of this particular source manifest found zero
source-family overlap across the original splits, so the relaxed rule does not
create source overlap by reshuffling videos.

The lists are built once from `artifacts/manifests/combined_manifest_p05.csv`
using seed 42, selecting each video's role only within its original split.
The one training real video known to decode to fewer than 32 frames,
`REAL/Real_WEB/id27_0005.mp4`, is excluded.
The generated manifests and `summary.json` are checked into the repository so
all runs use exactly the same fixed holdouts.

| Role | Real | Fake | How used |
| --- | ---: | ---: | --- |
| Training pool (original train) | 622 | 36,960 | Pool from which the 60 independent runs are built |
| Each independent run | 616 | 616 | 22 fake methods represented; batch size 8, exactly 4+4 in shuffled order |
| Validation (original val) | 134 | 134 | Same complete 268-video validation set every epoch and run; checkpoint/epoch/run selection and threshold calibration |
| Test (original test) | 133 | 133 | Same 266 paths, evaluated once after a run is selected on validation |
| Unused fake videos | — | 15,969 | Remaining original val/test fakes; no role in this experiment |

The 36,960 fake training videos are partitioned into 60 **non-overlapping**
groups of 616, stratified by forgery method. Each run samples 616 of the 622
training real videos with a group-specific deterministic random order, so real
videos necessarily recur across runs. Each epoch sees its run's 616 real and
616 fake videos once. There are 154 physical batches and 154 optimizer updates
per epoch. The six real videos omitted from a run can be different in another
run; the fake group never changes during a run.

The checked-in audit records zero shared source families for train/val,
train/test and val/test. Exact video-path overlap is also zero in every group.
These are observed properties of the original split, not extra selection rules.

Pilot settings: the E2 architecture, Adam at 1e-4, exponential LR factor 0.95
per epoch, BCE, FP32, no gradient clipping, no augmentation and no early
stopping. Each epoch is followed by evaluation on **all 268 fixed validation
videos**, with no further model-selection subset. Training and validation
losses, validation AUROC and balanced accuracy are logged every epoch; the
best validation-AUROC checkpoint is saved. Run 2–3 chosen groups first and
inspect whether validation AUROC is
still rising when the learning rate becomes small. **Only then** freeze the
maximum epoch count, LR schedule and optional early-stopping settings for all
60 formal runs. The test split must not influence those choices.
The 5% FPR threshold is calibrated on validation real videos for thresholded
accuracy and later test reporting; it is not a training-stop condition.
Every epoch also records a separate fixed-threshold operating point:
`sigmoid(-mean_video_real_logit) >= 0.5` predicts fake, equivalently fake
anomaly logit >= 0. The validation and final test reports include accuracy,
balanced accuracy, TN/FP/FN/TP, fake precision, fake recall and fake F1 at
both operating points. The fixed 0.5 threshold does not select epochs or runs.
The final evaluation checks that its validation/test video-list hash matches
the list saved with the checkpoint; a 60-group test also refuses threshold
recalibration on test videos.
An online training-clip accuracy at 0.5 is logged as a rough diagnostic; it
uses predictions made while the weights are changing, so it is not directly
comparable with the frozen-model, video-level validation accuracy.

Rebuild or audit the manifests on another checkout:

```bash
python scripts/build_phase_a_video_disjoint_groups.py \
  --input artifacts/manifests/combined_manifest_p05.csv \
  --output-dir artifacts/manifests/celeb_video_disjoint_seed42
```

The training pool uses fake videos that the earlier 18,369-video experiment did not
need. Before the GPU pilot, complete the face-box cache for all 38,116 required
videos using the generated union manifest; the training script will otherwise
stop during preflight. The cache script resumes past files already present:

```bash
python scripts/cache_face_boxes.py \
  --manifest artifacts/manifests/celeb_video_disjoint_seed42/all_required.csv \
  --datasets CelebDFv3 --splits train val test \
  --output-dir artifacts/face_cache
```

Example GPU pilot after setting `CELEBDFV3_ROOT` and completing that cache:

```bash
python pretrain_extractor.py \
  --config configs/v2/phase_a_celeb_video_disjoint.yaml \
  --manifest artifacts/manifests/celeb_video_disjoint_seed42/group_01.csv \
  --experiment E2_celeb --max-epochs 100 \
  --run-dir artifacts/v2_video_disjoint/e2_group01_pilot
```

Group 02 or 03 requires its own manifest and run directory; pass `--seed 43`
or `--seed 44`, respectively. For all 60 runs, use `--seed 42 + group_index - 1`
(computed as an integer before invoking the command) and record each seed.
The validation and test path lists stay fixed. Run directories cannot be reused
for another independent training, preventing accidental checkpoint overwrite.
