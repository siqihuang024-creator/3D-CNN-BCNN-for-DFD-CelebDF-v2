# Deepfake Video 3D-CNN + BCNN V2

本项目实现 `实验方案_V2.md` 的两阶段实验：Phase A 用真实/伪造视频监督训练特征器；Phase C 冻结特征器，只用真实视频训练 512→256→64→1 的贝叶斯异常头。主实验不是“纯单类表征学习”，完全不使用伪造训练标签的对照是 E7b。

## 环境与数据

先安装与 GPU 匹配的 PyTorch/torchvision，再运行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

数据根目录可写入 YAML，也可用 `DFD_ROOT`、`CELEBDFV3_ROOT` 或 `--dataset-root NAME=PATH` 覆盖。

CelebDF++ face 模式强制依赖离线 dlib 缓存。先复制 V1 缓存，再补齐 P0.5 新增视频：

```powershell
Copy-Item "E:/PhD/Deepfake Video 3D-CNN+BCNN/artifacts/face_cache" `
  "artifacts/face_cache" -Recurse
python scripts/cache_face_boxes.py `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --datasets CelebDFv3 --splits train val test --output-dir artifacts/face_cache
```

默认参数即 dlib、detect stride 4、最大检测边 640、upsample 1。还需将 `shape_predictor_81_face_landmarks.dat` 放入 `scripts/models/`。所有训练、缓存和评测入口都会逐视频预检；缺一个缓存也会立即报错，不会静默跳过。

## 1. P0.5 协议与捷径控制

```powershell
python scripts/build_manifest.py `
  --dfd-root "E:/path/to/DFD" --celeb-root "E:/path/to/CelebDFv3" `
  --output-dir artifacts/manifests --protocol p05 --seed 42

python scripts/shortcut_controls.py `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --video-sizes scripts/video_size_reports/video_sizes.csv `
  --box-cache artifacts/face_cache --splits val test --frame-mode face
```

P0.5 按 `(dataset, source_clip)` 家族做 70/15/15 划分，不丢视频，并检查三个 split 的家族交集为零。`--protocol p0` 和 `p1` 可生成控制协议。

## 2. Phase A、E4′ 与 E7

```powershell
python pretrain_extractor.py `
  --config configs/v2/phase_a_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --experiment E4_attention --seed 42
```

DFD 换用 `phase_a_dfd.yaml`。Phase A 固定 Adam 1e-4、每 epoch 乘 0.95、60 epoch、每类每 epoch 1000 clip；验证保留全部真实视频并按方法分层最多取 1000 个伪造视频，每视频 4 clip。最终测试仍使用全量视频、每视频 8 clip。

E4′ 第一轮运行五个非 GAP 变体的 seed 42；第二轮为 GAP 和前两名补齐 seed 42/43/44。E3 与 E4/GAP 在 CelebDF++ 上也运行三个 seed。各 run 完成 val 评测后执行：

```powershell
python scripts/select_aggregator.py `
  artifacts/v2/<gap-42> artifacts/v2/<gap-43> artifacts/v2/<gap-44> `
  artifacts/v2/<candidate1-42> artifacts/v2/<candidate1-43> artifacts/v2/<candidate1-44> `
  artifacts/v2/<candidate2-42> artifacts/v2/<candidate2-43> artifacts/v2/<candidate2-44>
```

胜者必须同时满足三 seed 平均 AUROC 高于 GAP、三个同 seed 配对全胜、身份聚类配对 bootstrap 的 95% 区间完全大于 0。选出胜者后再运行 E7，并显式传入聚合器：

```powershell
python pretrain_extractor.py ... --experiment E7_mc3 `
  --temporal-aggregation attention --seed 42
```

E7 是 MC3-18 + 两层 TCN + 入选聚合器；E7b 不带 TCN。DFD 在 E2 后运行只改变最终空间压缩的筛查 S：

```powershell
python scripts/pool_screen.py `
  --config configs/v2/phase_a_dfd.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --checkpoint artifacts/v2/<dfd-e2>/checkpoints/best.pt
```

## 3. Phase C

```powershell
python train_3d_bcnn.py `
  --config configs/v2/phase_c_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --init-extractor artifacts/v2/<phase-a-run>/checkpoints/best.pt `
  --experiment E4_attention --seed 42
```

Phase C 严格加载并冻结特征器和 BatchNorm、使用 FP32，并在验证 AUROC 连续 8 epoch 未提升至少 0.002 时早停。验证特征只计算一次；贝叶斯非线性头始终逐 clip 推断，再聚合成视频分数。

可先缓存每个真实训练视频的 16 个可复现随机 clip：

```powershell
python scripts/cache_video_features.py `
  --config configs/v2/phase_c_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --checkpoint artifacts/v2/<phase-a-run>/checkpoints/best.pt `
  --output artifacts/cache/celeb_train_real_16clips.npz
python train_3d_bcnn.py ... --feature-cache artifacts/cache/celeb_train_real_16clips.npz
```

## 4. 完整验证集与最终评测

训练时的 1,134-video `selection-val` 只用于选择 `best.pt`。每个实验完成后，
用该 checkpoint 在全部 8,205 个 CelebDFv3 validation 视频上统一评测；不要加
`--max-videos`，也不要用 test 做结构选择：

```powershell
python evaluate_3d_bcnn.py `
  --config configs/v2/phase_a_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --checkpoint artifacts/v2/<phase-a-run>/checkpoints/best.pt `
  --split val --expected-videos 8205 --bootstrap-draws 2000 --experiment E0
```

完整验证会写出 `full_val.json`、`full_val_video_scores.csv`、
`full_val_clip_scores.csv`，同时保留兼容旧脚本的 `val_scores.csv`。正式表统一使用
full-val 指标，AUROC 为 primary，`Macro-AP (real/fake)` 为 secondary。若实际评分
视频少于 manifest 请求数，正式评估会报错，不写出不完整的 full-val 报告。

E0/E1/E2 都完成 full-val 后做严格配对、聚类 bootstrap 和 manipulation 分解：

```powershell
python scripts/compare_experiments.py `
  artifacts/v2/<E0>/reports/full_val_video_scores.csv `
  artifacts/v2/<E1>/reports/full_val_video_scores.csv `
  artifacts/v2/<E2>/reports/full_val_video_scores.csv `
  --labels E0 E1 E2 --cluster-key identity --expected-videos 8205 `
  --draws 2000 --output results/v2/E0_E1_E2_full_val.json
```

比较报告中的 `paired_cluster_bootstrap` 是身份聚类主区间；
`sensitivity_paired_cluster_bootstrap` 同时给出 source-family 聚类的敏感性区间。

测试集仍只在最终模型确定后运行：

```powershell
python evaluate_3d_bcnn.py `
  --config configs/v2/phase_c_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --checkpoint artifacts/v2/<phase-c-run>/checkpoints/best.pt `
  --split test --bootstrap-draws 2000
```

报告包含 AUROC、真假两个方向 AP、Macro-AP (real/fake)、base rate、lift/gain、
balanced accuracy、EER、TPR@5%FPR，以及身份与 source-family 聚类区间。逐视频和
逐 clip 分数会同时保存，供配对比较和时间稳定性分析。

容器元数据关联诊断：

```powershell
python scripts/metadata_association.py `
  --scores artifacts/v2/<run>/reports/full_val_video_scores.csv `
  --video-sizes scripts/video_size_reports/video_sizes.csv `
  --box-cache artifacts/face_cache --dataset CelebDFv3
```

关联报告使用 Spearman rho，并分别输出 real-only、fake-only 和每种 manipulation
内部的 rho、p-value、n；不要把跨类别的总体相关解释为模型依赖。

捷径基线应在 E2 完整验证结果导出后，使用相同视频配对比较。例如人脸框占比：

```bash
python scripts/shortcut_baseline_scores.py \
  --manifest artifacts/manifests/combined_manifest_p05.csv \
  --video-sizes scripts/video_size_reports/video_sizes.csv \
  --box-cache artifacts/face_cache --dataset CelebDFv3 --split val \
  --control face_box_share_of_frame \
  --match artifacts/v2/celebdfv3_e2_gap_seed42/reports/full_val_video_scores.csv \
  --output results/v2/shortcut_facebox_scores.csv
python scripts/compare_experiments.py \
  artifacts/v2/celebdfv3_e2_gap_seed42/reports/full_val_video_scores.csv \
  results/v2/shortcut_facebox_scores.csv \
  --labels E2 FaceBoxShare --intersect --draws 2000 \
  --output results/v2/E2_vs_shortcut.json
```

CelebDF++ 的 22 种伪造方法里，有 13 种的人脸框占比单变量 AUROC 接近 1，
另 9 种该指标接近随机水平。这仅针对该单变量，不能排除其他捷径。
分组已冻结在 `configs/v2/shortcut_split_celeb_val.json`
（2026-09-23 由完整验证集的标签和人脸框占比分数导出，不使用模型分数），后续实验一律复用，
不再按新结果重新划分：

```bash
python scripts/shortcut_split_report.py \
  artifacts/v2/<E2>/reports/full_val_video_scores.csv \
  artifacts/v2/<E3>/reports/full_val_video_scores.csv \
  --labels E2 E3 --frozen-split configs/v2/shortcut_split_celeb_val.json \
  --draws 2000 --output results/v2/E2_E3_shortcut_split.json
```

该子集是事后定义、诊断用的敏感性分析，不是新的独立测试集；正式主指标仍是完整验证集上的 AUROC。

捷径脚本默认要求至少 99% 的模型视频有可用元数据，并写出
`*_provenance.json` 记录覆盖率、缺失数及分数方向。配对报告的 `alignment`
记录每个文件因交集而丢掉的真实/伪造视频数。E0/E1/E2 的模型比较仍按
完整相同的 video ID 严格配对，不使用 `--intersect`。方向自动校正使用了验证
标签，因此捷径对照和相关性分析属于诊断，不作为独立测试集性能。

## 验证

```powershell
python -m unittest discover -s tests -v
python smoke_test.py --config configs/v2/phase_a_celeb.yaml
```

每个 run 保存最终配置、运行环境、完整 history、best/last checkpoint 和报告。`runtime.json` 记录 clips/epoch、optimizer steps/epoch、优化器、学习率/衰减率、AMP、GPU/CUDA，以及 cuDNN deterministic/benchmark 状态。
