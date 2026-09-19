# V2 代码修改清单（交给 Codex 执行）

依据：`实验方案_V2.md`（第 6 版）。审查对象：提交 `4df17d7`。
审查方式：逐文件阅读 + 单元测试 + 在本地真实视频上端到端运行（CelebDF++ Phase A → Phase C → 测试评测已跑通）。

**执行规则**

1. 按编号顺序改。P0 全部完成之前，不要启动任何正式训练。
2. 每一项都有"验收"条件，改完必须满足；能写成单元测试的，写进 `tests/test_core.py`。
3. 最后一节"不要改动的部分"是已核实正确的实现，不要顺手"优化"。
4. 改完后运行 `python -m unittest discover -s tests -v` 和 `python smoke_test.py --config configs/v2/phase_a_celeb.yaml`，全部通过。

---

## P0：开跑前必须修（不修会得出错误结论，或者跑不完）

### P0-1　CelebDF++ 有 14,605 个视频缺人脸框缓存，且被静默跳过

**现象**：P0.5 把 V1 中因 donor 规则排除的伪造视频加了回来，但人脸框缓存只覆盖 V1 的划分。
按 `combined_manifest_p05.csv` 统计，缺缓存的 CelebDF++ 视频：

| split | fake 缺失 | real 缺失 |
|---|---|---|
| train | 9,990 / 36,960（27.0%） | 0 / 623 |
| val | 2,327 / 8,071（28.8%） | 0 / 134 |
| test | 2,287 / 8,165（28.0%） | 1 / 133（`REAL/Real_WEB/id11_0004.mp4`） |

`src/video_bcnn/data.py:175` 对缺失缓存抛 `UnreadableVideoError`，`data.py:619-625` 把它转成 `None`，
`skip_unreadable_collate` 再把它丢掉——**训练、验证、测试都悄悄少了约 28% 的伪造视频**，训练 batch 的真假平衡也被打破，
而 `reports/test.json` 里没有任何跳过记录（本地试跑：请求 6 个测试视频，只评了 5 个，报告无痕迹）。

**修改**：

1. 把 V1 的缓存目录 `E:/PhD/Deepfake Video 3D-CNN+BCNN/artifacts/face_cache/` 复制到本仓库 `artifacts/face_cache/`，
   然后只补缺失的部分（脚本会跳过已存在的文件）：
   ```bash
   python scripts/cache_face_boxes.py --manifest artifacts/manifests/combined_manifest_p05.csv \
       --datasets CelebDFv3 --splits train val test --output-dir artifacts/face_cache
   ```
   参数必须与 V1 一致：`--detector dlib`、`--detect-stride 4`、`--max-detect-side 640`、`--dlib-upsample 1`（均为脚本默认值）。
   需要 dlib 81 点模型 `scripts/models/shape_predictor_81_face_landmarks.dat`（本机没有，需下载后放到该路径）。
2. 新增异常类 `MissingFaceCacheError`（**不要**继承 `UnreadableVideoError`），`_cached_detections` 缺文件时抛它，
   `__getitem__` 不捕获它，让程序直接失败。
3. 新增开跑前检查函数 `preflight_face_cache(records, config)`：当 `frame_mode == "face"` 且配置了 `face_box_cache` 时，
   逐条检查缓存文件存在，缺任何一个就报错并列出数量和前 10 个路径。在 `pretrain_extractor.py`、`train_3d_bcnn.py`、
   `evaluate_3d_bcnn.py`、`scripts/cache_video_features.py`、`scripts/pool_screen.py`、`scripts/feature_diagnostics.py`
   构建数据集之前调用。
4. 仍会被跳过的真正损坏视频，要留下记录：Phase A/C 每个 epoch 的 history 行增加 `skipped_validation_videos`；
   `evaluate_3d_bcnn.py` 的报告增加 `num_videos_requested`、`num_videos_scored`、`skipped_videos`（含路径列表）。

**验收**：对 `combined_manifest_p05.csv` 运行 preflight，CelebDF++ 缺失数为 0；单元测试：缓存缺失时 `dataset[i]` 抛
`MissingFaceCacheError` 而不是返回 `None`。

### P0-2　每个 epoch 的样本量是方案的 37 倍

**现象**：`pretrain_extractor.py:100-104` 创建 `BalancedBatchSampler` 时 `batches_per_epoch=None`，
`experiment.py:169-172` 默认按最大的组（伪造视频）凑满一个 epoch：

| | 现在 | 方案 |
|---|---|---|
| CelebDF++，batch 8 | 9,240 批 = **73,920 clips/epoch**，每个真实视频每 epoch 重复约 59 次 | 2,000 clips/epoch |
| DFD，batch 4 | 1,058 批 = 4,232 clips/epoch | 2,000 clips/epoch |
| E0/E1（batch 1，CelebDF++） | **73,920 步/epoch** | 2,000 步/epoch |

**修改**：两份 `phase_a_*.yaml` 的 `train` 增加 `clips_per_class_per_epoch: 1000`。`pretrain_extractor.py` 据此计算：

- 走 `BalancedBatchSampler` 时：`batches_per_epoch = 2 * 1000 // physical_batch_size`（CelebDF++ 250 批，DFD 500 批）；
- 走 `GroupBalancedEpochSampler`（batch 1）时：`samples_per_group = 1000`。

启动时打印并写入 `runtime.json`：`clips_per_epoch`、`optimizer_steps_per_epoch`。

**验收**：E0–E4 在两个数据集上都是 2,000 clips/epoch（单元测试断言两种采样器的长度）。

### P0-3　验证集不设上限；Phase C 每个 epoch 重算冻结特征

**现象**：

- `pretrain_extractor.py:79-80` 用了全部验证视频：CelebDF++ 每个 epoch 8,205 个视频 × 8 clip = 65,640 次解码和前向。
  `experiment.py:360` 已有 `capped_validation_records`（V1 用过：每数据集 1,000 个伪造视频，按伪造方法分层，真实视频全留），但没被调用。
- `train_3d_bcnn.py:179-181` 每个 epoch 都把验证集重新解码、重算特征，但提取器是冻结的，50 个 epoch 算出来都一样。
- `evaluation.py:41` 把验证集每个 clip 的特征都存进内存；E0–E2 是 15,488 维，全量验证集约 4 GB。
- `selection_clips_per_video: 8`，方案（第 1 节表格）为 4。

**修改**：

1. Phase A 和 Phase C 的验证集都用 `capped_validation_records(val_records, seed, 1000)`；配置增加
   `data.selection_max_fakes_per_dataset: 1000`。**最终测试（`evaluate_3d_bcnn.py --split test`）不设上限**。
2. 两份配置的 `selection_clips_per_video` 改为 4；`eval_clips_per_video` 保持 8。
3. `score_deterministic` 的特征方差改为在线累计（sum / sum of squares），不保存全部特征。
4. Phase C：训练开始前把验证集的逐 clip 特征算一次、存在内存里（1,134 视频 × 4 clip × 512 维，很小），
   之后每个 epoch 只在缓存特征上跑贝叶斯头。

**验收**：CelebDF++ 每个 epoch 的验证 ≤ 1,134 个视频 × 4 clip；Phase C 从第 2 个 epoch 起验证不再解码视频。

### P0-4　`spatial_pool_type: max` 一次改了两处

**现象**：`model.py:148-150` 三个 stage 内部的池化和 `model.py:176-179` 最后的自适应空间压缩共用 `spatial_pool_type`。
实测：设成 `max` 后，`pool1/pool2/pool3` 全部变成 `MaxPool3d`，最后一步也变成 `adaptive_max_pool3d`。
而筛查 S 和 DFD 的候选改动只针对**最后一步**；stage 内部的池化按方案必须保持 V1 的 `AvgPool3d`。

**修改**：拆成两个参数：

- `stage_pool_type`：`pool1/pool2/pool3` 用，默认且在所有 V2 配置中固定为 `avg`；
- `final_pool_type`：`_spatial_reduce` 用，`avg`（默认）或 `max`。

配置、`experiment_matrix.yaml`、`pool_screen.py`、`build_feature_extractor` 同步改名。
旧键 `spatial_pool_type` 若出现在配置里，直接报错并提示改名，避免含义不清。

**验收**：单元测试：`final_pool_type="max"` 时 `pool1..3` 仍是 `AvgPool3d`，输出维度仍为 512。

### P0-5　`pool_screen.py`（筛查 S）无法支撑方案里的判据

**现象**：

- 只计算描述统计量（方差、centroid 距离），**没有线性探针，没有随机初始化对照**，方案 6.2 节的判据无法执行；
- `pool_screen.py:62-64` 取 manifest 最前面的 40 个真、40 个假视频，只覆盖两三个人——这是 V1 已经踩过并在
  `feature_diagnostics.py` 里用 `spread_by_identity` 修掉的问题；
- 受 P0-4 影响，`max4x4` 实际上连 stage 池化也换成了 max，结果无效。

**修改**：重写 `pool_screen.py`，复用 `feature_diagnostics.py` 中的 `probe`、`spread_by_identity`、`collect_features`：

1. 取 DFD val 的全部真实视频和等量伪造视频，都用 `spread_by_identity` 选；每视频 4 clip，分块前向（≤2 clip/次）。
2. 候选（只改 `final_pool_type` 和输出网格，stage 池化不变）：平均 4×4、最大值 4×4、平均 4×7、平均 8×14、平均 22×22。
3. 对每个候选：用给定检查点（E2，若 DFD 跳过 E2 则 E1）拟合探针，记录留出 AUROC；
   再用 8 个随机初始化的同结构提取器各拟合一次，记录分布（min / mean / max）。
4. 判据写进输出 JSON：`delta_trained = probe(max4x4) - probe(avg4x4)`；
   `delta_random_i = probe_random_i(max4x4) - probe_random_i(avg4x4)`（i = 1..8，同一个随机网络上比较）；
   `delta_trained > max(delta_random_i)` 时 `recommend_final_pool_type = "max"`，否则 `"avg"`。

**验收**：输出 JSON 含每个候选的训练后探针 AUROC、8 次随机分布、上述判据和推荐值；覆盖的身份数 ≥ 10（不足时警告）。

---

## P1：与实验方案不一致，必须改到一致

### P1-6　TCN 残差块的写法与方案不同

`model.py:52` 现为 `self.activation(values + self.norm(self.conv(values)))`，即 GELU 在相加**之后**。
方案（3.4 节）是 `x + GELU(BN(conv(x)))`。实测：卷积权重清零时，现写法输出与输入最多差 4.085，约 50% 的输入被 GELU 压扁；
方案写法差值为 0。方案要求 TCN 能退化成恒等映射，以保证 E3→E4 的对比干净。

**修改**：`return values + self.activation(self.norm(self.conv(values)))`。

**验收**：单元测试：卷积权重全置 0、BN 为初始状态时，块的输出严格等于输入。

### P1-7　E7 缺少 TCN，且聚合方式写死为 GAP

`model.py:209-238` 的 `MC3FeatureExtractor` 没有 TCN；`experiment_matrix.yaml` 的 `E7_mc3` 写死 `temporal_aggregation: gap`。
方案：E7 = MC3-18 → 池化 (T,1,1) → **与 E4 相同的 TCN** → **E4′ 选定的聚合** → Linear；E7b 不带 TCN（已正确）。

**修改**：

1. `MC3FeatureExtractor` 增加 `temporal_head` 参数：`"tcn"` 时在池化后接两个 `ResidualTemporalBlock(512, d=1/2)`（与 E4 完全相同），
   `"none"` 时不接。`E7_mc3` 设 `temporal_head: tcn`，`E7b_mc3_frozen` 设 `temporal_head: none`。
2. `pretrain_extractor.py`、`train_3d_bcnn.py` 增加命令行参数 `--temporal-aggregation`，在应用 `--experiment` 之后覆盖
   `model.temporal_aggregation`，用于把 E4′ 的胜者带进 E7。运行目录名里包含聚合方式。
3. `feature_diagnostics.py --random-init` 用于 MC3 时，用 `pretrained: false` 构建（现在会加载 Kinetics 权重，不是随机）。

**验收**：E7（GAP 时）提取器参数 = MC3 主干 11,490,240 + TCN 1,574,912 = 13,065,152；E7b 提取器没有 TCN 子模块。

### P1-8　Phase A 的优化器、学习率调度和 epoch 数与方案不同

`pretrain_extractor.py:124-128`：AdamW（weight_decay 1e-4）+ CosineAnnealingLR；配置 `epochs: 30`。
方案（6.1 节）：Adam，lr 1e-4，**每 epoch 乘 0.95 的指数衰减**，**60 个 epoch**，E0–E4 全部相同。
E0 的作用是"V1 结构在新协议下的参照点"，换了优化器就和 V1 不可比。

**修改**：`torch.optim.Adam(parameters, lr=1e-4)`（无 weight decay）+ `torch.optim.lr_scheduler.ExponentialLR(gamma=0.95)`，
每个 epoch 结束调用一次；两份 `phase_a_*.yaml` 改为 `epochs: 60`、`lr_gamma: 0.95`，删除 `weight_decay`。

**验收**：`runtime.json` 记录优化器名、lr、gamma；第 n 个 epoch 的学习率 = 1e-4 × 0.95^(n-1)。

### P1-9　Phase C 训练和评测看到的特征不一致

- 不走缓存时，`train_3d_bcnn.py:168-170` 用**单个 clip** 的特征训练；
- `evaluation.py:87-97` 评测时先把 8 个 clip 的特征**求平均**，再送进贝叶斯头；
- Phase A 的评测（`score_deterministic`）却是逐 clip 打分再平均；
- `cache_video_features.py` 每个视频只存一个平均特征，方案（第 7 节步骤 ⑥）是每视频 16 个随机 clip。

**修改**：统一为方案写法——**逐 clip 打分，再对 clip 平均**：

1. `score_bayesian`：对每个视频的 8 个 clip 分别算特征（分块），对每个 clip 算后验预测（MC 采样），
   视频分数 = −（各 clip 后验均值的平均），视频不确定度 = 各 clip 后验标准差的平均。
2. `cache_video_features.py`：每个真实训练视频缓存 16 个 clip 的特征（训练模式采样：随机起点、步长 {1,2} 随机、p=0.5 翻转，
   固定随机种子），存成 `[视频数 × 16, 512]`，并存每行对应的视频路径和 clip 序号。
3. `FeatureCacheDataset` 按行迭代（每行一个 clip 特征）。
4. `train_3d_bcnn.py` 中的 `num_train_videos` 改名 `num_train_units`，传入实际的训练单元数（缓存时为视频数 × 16，
   不缓存时为视频数），KL 按它缩放。

**验收**：单元测试：贝叶斯头为非线性时，`score_bayesian` 的视频分数等于逐 clip 分数的平均，而不等于"平均特征"的分数。

### P1-10　Bootstrap 的聚类单位与方案不同

`evaluate_3d_bcnn.py:30-31` 只按 `source_clip` 聚类。方案（9.2 节）：按**身份**聚类，2000 次，
并注明携带稀缺类别（real）的簇数量。

**修改**：报告两个区间：`identity_bootstrap`（按 `target_id`，**作为主结果**）和 `source_family_bootstrap`（按 `source_clip`）。
每个区间都输出：簇总数、含真实视频的簇数、含伪造视频的簇数。

### P1-11　Phase C 没有早停

方案第 7 节步骤 ⑥ 是 `--early-stopping-patience 8`。`train_3d_bcnn.py` 一律跑满 50 个 epoch。

**修改**：增加早停：验证 AUROC 连续 8 个 epoch 未提升 0.002 以上即停止（与 V1 的 `selection_min_metric_improvement` 一致），
配置项 `train.early_stopping_patience: 8`。Phase A 不加早停（方案要求跑满 60 epoch）。

### P1-12　E4′ 选胜者的规则没有实现

方案 6.2 节：第一轮五个变体各 1 个 seed；第二轮 GAP 与前两名各补到 3 个 seed（42/43/44）；
胜者须同时满足：3-seed 均值高于 GAP、3 个 seed 两两配对都赢、对 3-seed 平均分数做配对 bootstrap 的 95% 区间不含 0。

**修改**：新增 `scripts/select_aggregator.py`：

- 输入：若干运行目录（每个目录下已有 `evaluate_3d_bcnn.py --split val` 生成的 `reports/val_scores.csv`）；
- 按视频路径对齐各运行的分数；同一聚合方式的 3 个 seed 取平均分数；
- 输出：每种聚合方式的 seed 均值和标准差、逐 seed 胜负、配对 bootstrap（按身份聚类，2000 次）的差值区间、最终推荐。

同时在 README 写明：E3 和 E4 在 CelebDF++ 上各跑 seed 42/43/44（E4 的三个 seed 与 E4′ 第二轮的 GAP 共用）；
E7 在 E4′ 选定之后再跑。

### P1-13　E0S 需要的文件没有复制过来

`scripts/shortcut_controls.py` 需要 `--video-sizes scripts/video_size_reports/video_sizes.csv`，本仓库没有；
方案 2.2 节提到的 `scripts/metadata_association.py` 也没有。

**修改**：从 V1 仓库复制 `scripts/video_size_reports/video_sizes.csv` 和 `scripts/metadata_association.py`；
检查 `metadata_association.py` 能读 `evaluate_3d_bcnn.py` 新输出的 `*_scores.csv`（列名可能变了，需要适配）。

**验收**：在 `combined_manifest_p05.csv` 上跑通 `shortcut_controls.py --splits val test`。

---

## P2：小问题

- **P2-14** `cache_video_features.py:49-51, 61`：字典键用 `resolve()` 后的路径，而 batch 里的路径没有 resolve；
  远程数据目录若是软链接会 `KeyError`。改为两边都用 manifest 的 `(dataset, path)` 作键（batch 里已有 `dataset` 字段，
  再加一个 `relative_path` 字段即可）。
- **P2-15** `experiment.py:392` 的 `score_loader` 是 V1 遗留、一次性前向整批所有 clip 的旧实现，现在没有被调用；
  删除它，避免以后被误用。
- **P2-16** 在 `runtime.json` 记录 `cudnn.deterministic` 的取值；远程第一次运行时对比 True/False 的每步耗时，写进日志。

---

## 修改后必须满足的方案对照表（逐项自查）

| 项 | 方案要求 | 代码位置 |
|---|---|---|
| 划分 | P0.5：按 `(dataset, source_clip)` 家族分层 70/15/15，零家族交叉，不丢视频 | `protocols.py`（已正确） |
| DFD 输入 | 整帧 `[::2, ::2]`，**不缩放**，`[3,T,540,960]` | `data.py`（已正确，实测） |
| CelebDF++ 输入 | dlib 缓存 + 眼线对齐 + 框 ×2.0（`face_margin: 0.5`）+ 边缘复制，`[3,T,256,256]` | `data.py`（已正确，实测） |
| clip 长度 | E0 为 8，其余 32 | 配置 + 矩阵（已正确，实测） |
| 输入采样步长 | 训练 {1,2} 随机，评测 2 | 配置（已正确，实测约各半） |
| 评测 clip | 每视频 8 个，均匀分布；验证 4 个 | P0-3 |
| 3D-CNN | 16/24/32，k(3,5,5)，pad(1,0,0)，stage 池化 **Avg**(1,4,4)/(1,2,2)，BN3d，ReLU，stride 1；90,280 参数 | P0-4 |
| 空间压缩 | `AdaptiveAvgPool3d((T,4,4))`（DFD 视筛查可改最后一步为 max）+ BN3d → `[B,T,512]` | P0-4 |
| TCN | 两层，k3，d=1/2，`x + GELU(BN(conv(x)))` | P1-6 |
| 时间聚合 | E4 为 GAP；E4′ 五个变体；E5/E6/E7 用选定的方式 | P1-7、P1-12 |
| Phase A | BCE，AMP，Adam 1e-4，γ=0.95，60 epoch，每类每 epoch 1000 clip，batch 按真假平衡 | P0-2、P1-8 |
| batch | CelebDF++ 8；DFD 4 × 累积 2；E0/E1 为 1 | 矩阵（已正确） |
| Phase C | 只用 real，提取器冻结（含 BN），fp32，512→256→64→1 + GELU，逐 clip 打分再平均，早停 8 | P1-9、P1-11 |
| 评测 | 每批 1 个视频、每次 ≤4 clip；测试集全量；阈值取验证集 real 的 5% FPR | 已正确；测试不设上限 |
| 指标 | AUROC 主；AP 两个方向 + base rate + 提升；EER；TPR@5%FPR；按身份聚类 bootstrap | P1-10 |
| 缺失数据 | 缓存缺失直接报错；损坏视频的跳过数写进报告 | P0-1 |

---

## 不要改动的部分（已核实正确）

- `protocols.py` 的 P0.5 / P0 / P1 划分，以及已生成的 `combined_manifest_p05.csv`（与 V1 视频集合完全一致、无重复、1,253 个家族零交叉、22 种伪造方法在每个 split 都有）；
- 提取器、E0/E3/E4、五种聚合方式的结构和参数量（已实测与方案一致）；
- DFD 的 `decimate` 路径不经过 `Resize`/`CenterCrop`；
- `BalancedBatchSampler` 的逐 batch 真假平衡（CelebDF++ 4+4，DFD 2+2）、AMP 与梯度累积的实现；
- `evaluation.py` 的"每批 1 个视频、clip 分块前向"；
- `metrics.py` 的两个方向 AP、base rate 与提升倍数；
- Phase C 的严格加载（`strict=True`）、冻结 BN、fp32、Pyro `ClippedAdam`（它解决了 V1 中 ρ 完全不动的问题——实测一步即变化）。
