# Deepfake Video 3D-CNN + BCNN V2

本项目实现 `实验方案_V2.md` 的改进实验。实验被明确拆成两阶段：

1. **Phase A（监督表征学习）**：真实与伪造视频共同训练 3D-CNN/MC3-18 特征器；
2. **Phase C（真实样本贝叶斯异常建模）**：冻结特征器，仅用真实视频训练均值场贝叶斯头，测试时以负预测均值作为伪造异常分数。

它不是端到端的“纯单类学习”：主方案的特征器在 Phase A 使用了伪造标签。可选 E7b 才是完全不使用伪造训练标签的控制。

## 项目结构

- `src/video_bcnn/`：数据、模型、协议、指标、评估与日志模块；
- `configs/v2/`：CelebDF++/DFD 的 Phase A、Phase C 配置和 E0–E7 消融矩阵；
- `scripts/build_manifest.py`：P0、P0.5、P1 数据划分；
- `scripts/pool_screen.py`：DFD 的池化诊断 S；
- `pretrain_extractor.py`：Phase A；
- `train_3d_bcnn.py`：Phase C；
- `evaluate_3d_bcnn.py`：统一的视频级验证/测试；
- `tests/`：无需数据集即可运行的协议和模型契约测试。

## 安装

建议在目标 GPU 对应的官方源中先安装匹配的 PyTorch/torchvision，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

代码保持 Python 3.7 / PyTorch 1.13 兼容；E7 的 MC3-18 同时兼容 torchvision 新旧 weights API。

数据根目录可写在 YAML，也可在当前机器设置 `DFD_ROOT`、`CELEBDFV3_ROOT`，或在命令中重复传入 `--dataset-root NAME=PATH`。

## 1. 构建主协议 P0.5 manifest

```powershell
python scripts/build_manifest.py `
  --dfd-root "E:/path/to/DFD" `
  --celeb-root "E:/path/to/CelebDFv3" `
  --output-dir artifacts/manifests `
  --protocol p05 --seed 42
```

输出为 `combined_manifest_p05.csv` 和 `combined_summary_p05.json`。生成过程会自动检查：

- 每条记录都有 `source_clip`；
- 真实源视频及其所有派生伪造属于同一 split；
- train/val/test 的源视频家族交集为 0；
- P0.5 不丢弃任何视频。

把 `--protocol` 改成 `p0` 或 `p1` 可生成两种控制协议。

## 2. Phase A 和 E0–E4′/E7

CelebDF++ 示例：

```powershell
python pretrain_extractor.py `
  --config configs/v2/phase_a_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --experiment E4_attention --seed 42
```

DFD 只需换成 `configs/v2/phase_a_dfd.yaml`。`--experiment` 的可用名称在
`configs/v2/experiment_matrix.yaml` 中，包括 E0、E1、E2_celeb/E2_dfd、E3、六种 E4′ 聚合器和 E7_mc3。

对入选的 E4′ 方案应至少运行种子 42、43、44。E7 仅按方案在 CelebDF++ 上运行。

DFD 在 E2 后先运行池化诊断：

```powershell
python scripts/pool_screen.py `
  --config configs/v2/phase_a_dfd.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --checkpoint artifacts/v2/dfd_e2_dfd_seed42/checkpoints/best.pt
```

## 3. Phase C

Phase C 的配置必须与选中的 Phase A 特征器结构一致：

```powershell
python train_3d_bcnn.py `
  --config configs/v2/phase_c_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --init-extractor artifacts/v2/celebdfv3_e4_attention_seed42/checkpoints/best.pt `
  --experiment E4_attention --seed 42
```

加载采用 `strict=True`，因此不可能把错误结构的特征器悄悄接到贝叶斯头。Phase C 始终为 FP32，特征器参数和 BatchNorm 运行统计均被冻结。

可选 E7b 完全不使用伪造训练标签：在 Celeb 配置上指定
`--experiment E7b_mc3_frozen` 并省略 `--init-extractor`，程序会直接冻结
Kinetics 预训练 MC3-18，再仅用 P0.5 train 中的真实视频训练贝叶斯头。

Phase C 推荐先缓存稳定的视频级特征，避免 50 个 epoch 重复解码：

```powershell
python scripts/cache_video_features.py --config configs/v2/phase_c_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --checkpoint <phase-a-best.pt> --output artifacts/cache/celeb_train_real.npz
python train_3d_bcnn.py ... --feature-cache artifacts/cache/celeb_train_real.npz
```

## 4. 最终评估

```powershell
python evaluate_3d_bcnn.py `
  --config configs/v2/phase_c_celeb.yaml `
  --manifest artifacts/manifests/combined_manifest_p05.csv `
  --checkpoint artifacts/v2/celebdfv3_e6_e4_attention_seed42/checkpoints/best.pt `
  --split test --bootstrap-draws 2000
```

评估固定每个 DataLoader batch 一个视频、每次最多前向 4 个 clips。报告与 checkpoint 放在同一 run 下，包含视频级 AUROC、fake/real AP、两类 base rate 与 lift、balanced accuracy、EER、TPR@5%FPR、按 `source_clip` 聚类的 2000 次 bootstrap 区间，以及数据集/伪造方法分组结果。

## 验证代码

```powershell
python -m unittest discover -s tests -v
python smoke_test.py --config configs/v2/phase_a_celeb.yaml
```

所有 run 都保存 `config.json`、`runtime.json`、`logs/history.json`、`logs/history.csv`、`checkpoints/last.pt` 和 `checkpoints/best.pt`。运行时记录实际物理/有效 batch、精度、worker 数、Python/PyTorch/CUDA/GPU 与 git revision；每轮记录耗时、峰值显存、特征方差、三阶段饱和度、TCN 输出统计以及 Phase C 的 rho/sigma。
