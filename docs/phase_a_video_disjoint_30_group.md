# CelebDF++ Phase A: video-disjoint 30-run protocol

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
| Training pool (original train) | 622 | 36,960 | Source of the precomputed disjoint fake groups |
| Each independent run | 616 | 616 | 22 fake methods represented; batch size 8, exactly 4+4 in shuffled order |
| Union of the 30 training runs | Up to 622, reused | 18,480 distinct | Fixed groups 01–30; the other 18,480 training fakes are not used |
| Validation (original val) | 134 | 134 | Same complete 268-video validation set every epoch and run; checkpoint/epoch/run selection and threshold calibration |
| Test (original test) | 133 | 133 | Same 266 paths, evaluated once after a run is selected on validation |
| Other unused fake videos | — | 15,969 | Remaining original val/test fakes; no role in this experiment |

The checked-in manifests partition all 36,960 fake training videos into 60
**non-overlapping** groups of 616, stratified by forgery method. The formal
experiment uses only the fixed groups **01–30**, totaling 18,480 distinct
training fake videos (half of the available training-fake pool). Do not
regenerate a 30-group partition: that would change the pilot groups and
break comparability. The decision to use 30 groups was made after inspecting
pilot groups 01–03 to reduce the computation budget; their validation
results were already known. No further group-count change or selection of
group IDs based on validation results is allowed.

Each run samples 616 of the 622 training real videos with a group-specific
deterministic random order, so real
videos necessarily recur across runs. Each epoch sees its run's 616 real and
616 fake videos once. There are 154 physical batches and 154 optimizer updates
per epoch. The six real videos omitted from a run can be different in another
run; the fake group never changes during a run.

The checked-in audit records zero shared source families for train/val,
train/test and val/test. Exact video-path overlap is also zero in every group.
These are observed properties of the original split, not extra selection rules.

Training settings: the E2 architecture, Adam at 1e-4, exponential LR factor 0.95
per epoch, BCE, FP32, no gradient clipping, no augmentation and no early
stopping. Each epoch is followed by evaluation on **all 268 fixed validation
videos**, with no further model-selection subset. Training and validation
losses, validation AUROC and balanced accuracy are logged every epoch; the
best validation-AUROC checkpoint is saved. The three pilots used 100 epochs
without early stopping. Their validation results fixed the formal rule at
**100 epochs per group, no early stopping**, with the best validation-AUROC
checkpoint selected independently within each group. The same validation
AUROC then selects one of the 30 groups. The test split does not influence
either selection.

| Pilot group | Best epoch | Best validation AUROC | Best epoch within first 60 | Late best (epochs 61–100) |
| --- | ---: | ---: | ---: | ---: |
| 01 | 30 | 0.7807 | 30 (0.7807) | 88 (0.7787) |
| 02 | 93 | 0.7973 | 60 (0.7893) | 93 (0.7973) |
| 03 | 37 | 0.7787 | 37 (0.7787) | 82 (0.7770) |

These values come from the completed remote pilot histories. A 60-epoch cap
would miss group 02's best checkpoint. Retrospective patience-20 stopping
would stop group 03 at epoch 31, before its best checkpoint at epoch 37.
Fixed 100-epoch budgets give every group the same number of optimization
updates and validation opportunities. Pilot groups 01–03 may count toward the
30 formal runs only after verifying that their saved training-effective
settings, manifests and seeds match the formal runs and that the training
implementation has not changed. The documentation/configuration commit after
the pilots changes the Git revision without changing the training code or
effective settings. The run directory and inactive early-stopping fields need
not match. If any training-effective setting or training implementation differs,
rerun those groups from scratch.

The difference between group 02's AUROC at epoch 93 and its best result by
epoch 60 is 0.0080 on only 268 validation videos. It is a reason not to
truncate the prespecified 100-epoch budget, not evidence that learning was
still improving reliably at epoch 93. Selecting the maximum over 100 epochs
and then over 30 groups favors optimistic validation fluctuations. Report the
distribution of the 30 groups' best validation AUROCs (mean, standard
deviation, median and range), the selected group's validation AUROC and best
epoch, and the held-out test metrics for that single selected checkpoint.
Do not treat the 30 correlated group runs as 30 independent test estimates.
The union of formal groups 01–30 covers 18,480 training fakes, not all 36,960;
the selected checkpoint was trained on only its own 616 fake videos.
Do not evaluate multiple candidate groups on test and pick the best test score.

The 5% FPR threshold is calibrated on validation real videos for thresholded
accuracy and later test reporting; it is not a training-stop condition.
Every epoch also records a separate fixed-threshold operating point:
`sigmoid(-mean_video_real_logit) >= 0.5` predicts fake, equivalently fake
anomaly logit >= 0. The validation and final test reports include accuracy,
balanced accuracy, TN/FP/FN/TP, fake precision, fake recall and fake F1 at
both operating points. The fixed 0.5 threshold does not select epochs or runs.
The final evaluation checks that its validation/test video-list hash matches
the list saved with the checkpoint; this protocol's test also refuses threshold
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

After auditing and reusing pilot groups 01–03, start with group 04. Example
run after setting `CELEBDFV3_ROOT` and completing that cache:

```bash
python pretrain_extractor.py \
  --config configs/v2/phase_a_celeb_video_disjoint.yaml \
  --manifest artifacts/manifests/celeb_video_disjoint_seed42/group_04.csv \
  --experiment E2_celeb --seed 45 \
  --run-dir artifacts/v2_video_disjoint/e2_group04_formal
```

For the fixed 30 runs (groups 01–30), use the matching group manifest, a
distinct run directory and `--seed 42 + group_index - 1` (compute the integer
before invoking the command). Group 02 uses seed 43, group 03 uses seed 44,
and so on. Record each
seed and the Git commit used for the run. Do not use `--max-epochs 60` or
enable early stopping for one group while leaving the others at 100 epochs.
The validation and test path lists stay fixed. Run directories cannot be reused
for another independent training, preventing accidental checkpoint overwrite.

After group 04 is complete, run groups 05–30 sequentially in a tmux session:

```bash
source remote_env.sh
python scripts/run_phase_a_groups_05_30.py --dry-run
python scripts/run_phase_a_groups_05_30.py
```

The runner uses each group's checked-in manifest, seed `41 + group number`,
and a separate `e2_groupXX_formal` output directory. It waits for one group
to finish all 100 epochs before launching the next. Re-running the same
command skips groups with complete, matching output; a failed or partial
group stops the queue for inspection and is never overwritten. The runner
does not evaluate the test set or select the winning group.
