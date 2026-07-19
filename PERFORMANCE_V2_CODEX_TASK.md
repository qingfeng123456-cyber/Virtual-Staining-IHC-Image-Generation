# AIC 虚拟染色项目性能升级：Codex Plan / Implementation 完整任务书

> 任务性质：在已经可运行的现有竞赛工程上做**增量性能升级**。  
> 目标：优先提升官方 SSIM 与 PSNR，同时保留多标记联合建模、可解释性和跨器官泛化能力。  
> 核心方案代号：**CAMP-VS v2 — Context-Aware Multi-Prototype Virtual Staining Restorer**。  
> 注意：本任务不是重新生成一个项目，也不是无验证地堆叠论文模块。所有新增模块必须可开关、可消融、可回滚，只有在严格的 ROI 分组验证中稳定涨分后才允许进入最终提交配置。

---

## 0. 你现在的角色和工作方式

你是本项目的高级算法工程师、研究复现工程师和竞赛性能优化负责人。当前仓库已经由之前的 Codex 任务构建完成，预计已经具备：

- 数据自动发现与审计；
- manifest；
- Residual U-Net 基线；
- NAF/ConvNeXt 风格多任务恢复模型；
- 共享/任务特异原型；
- Charbonnier、SSIM、MS-SSIM、梯度、频域损失；
- EMA、AMP、断点续训；
- D4 TTA；
- checkpoint ensemble；
- 推理与 submission validator；
- pytest 和 smoke pipeline。

但这只是“预计”，不得直接假设。你必须先检查真实仓库。

你的工作分为两个连续阶段。

### 0.1 Plan 阶段

当前先处于 Plan 模式。只读检查仓库、配置、测试、训练日志和数据，不要直接大规模改写。

Plan 阶段必须完成：

1. 确认当前工程真实结构和入口；
2. 运行或读取当前测试状态；
3. 找到现有最佳 checkpoint、配置、日志和验证指标；
4. 确认当前模型是否已经实现本任务书提到的功能；
5. 确认数据文件名是否能解析出 `ROI、row、col`；
6. 确认官方 train/val 划分是否存在 ROI 泄漏；
7. 判断 DAPI 与目标是否高度对齐；
8. 判断训练设备、显存和可承受的实验规模；
9. 给出增量文件修改计划和实验预算；
10. 禁止重复实现现有功能。

### 0.2 Implementation 阶段

用户批准计划后，立即开始实现，不要再重复询问。

Implementation 阶段必须遵循：

- 先冻结当前基线；
- 每次只引入一组可验证变化；
- 实现后运行相关单元测试；
- 运行 smoke train；
- 运行最小真实数据实验；
- 生成对比报告；
- 未涨分的模块保持关闭或删除；
- 遇到报错自行定位并修复；
- 不得留下 `TODO`、`pass`、伪代码、空函数；
- 不得声称完成未实际运行的长时间训练；
- 不得把 test 数据用于训练、预训练、归一化、模型选择或 ensemble 权重拟合。

---

# 1. 赛题目标与优化优先级

官方评分核心：

```text
Score = 0.7 × SSIM + 0.3 × Normalize(PSNR)
```

因此主模型必须优先优化：

1. 空间结构和局部对比度；
2. 像素级误差；
3. 保存为 JPG 后的真实 SSIM/PSNR；
4. 多器官稳定性；
5. 多标记联合训练带来的迁移收益。

不要把以下内容作为默认主线：

- GAN；
- PatchGAN；
- 感知损失；
- LPIPS；
- 扩散采样；
- 大型 H&E 病理基础模型；
- 手工测试集后处理；
- 会明显降低 PSNR/SSIM 的“视觉更真实”损失。

这些只能是独立实验分支，并且只有验证指标稳定提高才可以使用。

如果同时提交多个输出，官方可能对多个输出成绩取平均。因此：

- 联合模型可以训练四个标记；
- 正式排行榜提交默认只导出当前官方要求的目标；
- 不要为了展示多任务而误把四个输出一起放入正式 submission。

---

# 2. 研究结论到工程设计的映射

实现时遵循下列研究结论，但不要直接复制来源不清或许可证不兼容的代码。

## 2.1 ProtoMTG：共享与任务特异原型

可借鉴：

- 共享原型；
- 标记特异原型；
- 原型注意力；
- 原型激活与多样性约束；
- 多标记之间共享表示。

需要改进：

- 原型不能只存在于单一 bottleneck；
- 增加多尺度原型；
- 增加使用率监控；
- 防止死原型；
- 原型残差必须零初始化或小权重初始化；
- 不能让原型损失压制 SSIM/PSNR。

## 2.2 PyramidPix2Pix：高斯/拉普拉斯金字塔监督

可借鉴：

- 多尺度输出；
- Gaussian/Laplacian pyramid reconstruction；
- 低频强度和高频细节分离。

不要默认采用：

- adversarial loss；
- 直接复制 pix2pix 训练框架。

## 2.3 NAFNet / Restormer / MambaIRv2：图像恢复骨干

默认骨干：

- 高分辨率层使用 NAF 风格局部块；
- 低分辨率 bottleneck 使用纯 PyTorch Restormer 式高效全局块；
- 不在高分辨率特征上做标准全局 self-attention。

MambaIRv2：

- 仅作为可选 P2 实验；
- 不得成为工程运行的硬依赖；
- 如果需要 `mamba_ssm`、`causal_conv1d` 或平台特定编译扩展，默认关闭；
- 主工程必须在纯 PyTorch 环境下正常运行。

## 2.4 UNIStainNet / PGVMS：空间条件与标记嵌入

可借鉴：

- marker embedding；
- FiLM；
- 空间条件调制；
- 多尺度边缘条件；
- 统一多标记生成。

不直接使用：

- UNI、CONCH 等 H&E 领域基础模型作为默认；
- DAB 光密度约束；
- HER2 专属病理规则；
- 外部 H&E/IHC 数据。

DAPI 与 H&E 域不同。更安全的方案是：

- 使用赛事官方训练 DAPI 做自监督预训练；
- 或直接从 DAPI 生成空间条件特征；
- 不默认使用外部病理基础模型。

## 2.5 FAMO：多任务动态平衡

实现可选的任务优化模式：

```text
equal
famo
uncertainty
```

默认先保持 equal，只有 FAMO 在多标记训练中稳定提升目标任务指标才启用。

同时记录共享参数的任务梯度余弦相似度，用于判断负迁移，但不要每步计算，避免训练过慢。

## 2.6 弱配准与注册方法

先依据数据审计判断：

- 若同一 mIHC 切片中的 DAPI 和目标高度对齐，不启用注册网络；
- 若只存在约 1–2 像素小偏移，可实验 shift-tolerant loss；
- 若存在明显局部形变，才允许设计独立的训练期对齐分支。

禁止：

- 推理时扭曲输出；
- 为了提高验证分把预测向 GT 平移；
- 使用测试标签；
- 未经审计直接引入复杂注册 GAN。

---

# 3. 第一阶段：冻结现有基线

在修改任何模型之前，创建一个可追溯基线快照。

## 3.1 创建 Git 分支

如果当前目录是 Git 仓库：

```bash
git status
git rev-parse HEAD
git switch -c performance-v2
```

若工作区有未提交修改：

- 不覆盖；
- 输出差异摘要；
- 先创建安全备份或提交点；
- 不得删除用户代码。

若不是 Git 仓库：

- 不强制初始化；
- 生成文件哈希快照。

## 3.2 基线快照

新增：

```text
artifacts/performance_v2/baseline_snapshot.json
artifacts/performance_v2/baseline_files.sha256
docs/PERFORMANCE_V2_BASELINE.md
```

至少记录：

- 当前 commit；
- Python、Torch、CUDA、GPU；
- 数据根目录；
- train/val/test 数量；
- 器官；
- 目标标记；
- 当前最佳模型；
- 参数量；
- FLOPs 或 MACs；
- 当前配置；
- seed；
- 训练 epoch；
- float-domain 指标；
- uint8-domain 指标；
- 保存为 JPG 后重新读取的指标；
- 按 organ/marker/ROI 指标；
- 当前推理速度；
- 峰值显存；
- checkpoint hash；
- manifest hash。

若没有可用训练日志，运行当前 `smoke` 和一个预算可控的 baseline benchmark。

## 3.3 基线不可被覆盖

后续输出使用新的目录：

```text
outputs/performance_v2/
artifacts/performance_v2/
configs/performance_v2/
```

不得覆盖现有 best checkpoint、日志和 submission。

---

# 4. 最优先的数据结构改进：ROI 邻域上下文

赛题 patch 文件名类似：

```text
ROI000_00_00.jpg
ROI000_00_01.jpg
ROI000_01_00.jpg
```

这意味着同一 ROI 中 patch 很可能具有二维网格关系。现有单 patch 模型只能看到 256×256 局部，而标记表达往往与更大组织环境有关。

实现一个完全由 DAPI 构成、训练和测试都可用的邻域上下文系统。

## 4.1 坐标解析

新增或扩展：

```text
src/virtual_staining/data/roi_index.py
src/virtual_staining/data/neighborhood.py
```

实现：

```python
@dataclass(frozen=True)
class PatchCoordinate:
    roi_id: str
    row: int
    col: int
```

支持：

- `ROI000_00_00`
- 行列超过两位；
- 大小写差异；
- 文件扩展名差异；
- 无法解析时返回明确状态，不要猜测。

生成：

```text
artifacts/performance_v2/roi_grid_audit.json
artifacts/performance_v2/roi_grid_missing.csv
artifacts/performance_v2/figures/roi_mosaics/
```

审计：

- 每个 ROI 的 row/col 范围；
- 空洞；
- 重复坐标；
- train/val 是否共享 ROI；
- 相邻 patch 边界是否连续；
- 行列方向是否正确。

通过比较相邻边界条带的像素误差，自动判断：

- col 增大是否向右；
- row 增大是否向下；
- 是否存在坐标转置；
- 是否存在翻转。

若边界连续性证据不足，邻域模块默认关闭并报告原因。

## 4.2 防泄漏规则

邻居只允许来自：

- 同一 organ；
- 同一 split；
- 同一 ROI；
- 同一输入模态 DAPI。

严禁：

- train 样本加载 val 邻居；
- val 样本加载 train 邻居；
- 使用邻居的目标标记作为中心监督；
- 测试上下文使用任何训练目标；
- 测试统计参与训练归一化。

中心 patch 的监督目标仍只使用中心 GT。

最终锁定配置后，允许使用全部官方有标签数据重新训练，此时 train+val 合并为 final_train，邻域也只能在该 final_train 内构造。

## 4.3 邻域形式

默认 3×3：

```text
(-1,-1) (-1,0) (-1,+1)
( 0,-1) ( 0,0) ( 0,+1)
(+1,-1) (+1,0) (+1,+1)
```

Dataset 返回：

```python
{
    "input": center_dapi,                    # [C,256,256]
    "context_tiles": context_tiles,          # [9,C,256,256]
    "context_valid_mask": valid_mask,        # [9]
    "context_offsets": offsets,              # [9,2]
    "target": ...,
    "organ_id": ...,
    "task_id": ...,
    "metadata": ...
}
```

缺失邻居默认：

- 图像张量使用 center patch 复制；
- `valid_mask=0`；
- attention 中严格 mask；
- 不把缺失邻居当作真实信息。

可配置：

```yaml
context:
  enabled: true
  grid_size: 3
  missing_policy: center
  include_center: true
```

不要把 9 张 256 图像简单拼成 768×768 后以全分辨率送入大网络，显存浪费过大。

## 4.4 同步增强

所有邻域 tile 与 center/target 必须共享：

- hflip；
- vflip；
- rot90；
- row/col offset 的相应变换；
- valid mask 的相应变换。

需要单元测试验证：

- hflip 后左右邻居交换；
- vflip 后上下邻居交换；
- rot90 后坐标正确旋转；
- center/target 像素几何完全同步；
- context mask 同步。

只对 DAPI 做的亮度、gamma、噪声增强，可以：

- 对 9 个 tile 使用同一参数，保持 ROI 强度一致；
- 或使用“全局参数 + 极小局部扰动”，但默认使用同一参数。

---

# 5. CAMP-VS v2 模型

新增主模型：

```text
src/virtual_staining/models/camp_vs_v2.py
```

并拆分为：

```text
models/
├─ camp_vs_v2.py
├─ naf_local_encoder.py
├─ context_tile_encoder.py
├─ context_fusion.py
├─ restoration_transformer.py
├─ hierarchical_prototypes.py
├─ task_organ_conditioning.py
├─ laplacian_decoder.py
└─ intensity_calibrator.py
```

不要重复已有 NAF block。优先复用当前仓库已经测试过的模块。

## 5.1 总体计算图

```text
Center DAPI
    │
    ├── Sobel / Laplacian structural channels（可选）
    │
    ▼
High-resolution Local Encoder
    │
    ├── 1/2 feature
    ├── 1/4 feature
    ├── 1/8 feature
    └── 1/16 feature
                  ┌───────────────────────────────┐
3×3 DAPI tiles ──► Shared Tiny Context Encoder   │
                  └──────────────┬────────────────┘
                                 ▼
                   9 context tokens + position embeddings
                                 │
                 zero-init FiLM / gated context fusion
                                 │
                                 ▼
          Restormer-lite global bottleneck + hierarchical prototypes
                                 │
                                 ▼
     Shared Decoder + marker adapter + organ adapter + residual expert
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
          low-frequency base             high-frequency detail
                  └──────────────┬──────────────┘
                                 ▼
                     bounded intensity calibration
                                 ▼
                      target virtual stain image
```

## 5.2 Local encoder

默认：

```yaml
local_encoder:
  type: naf
  widths: [48, 96, 192, 384]
  depths: [2, 2, 4, 6]
  drop_path: 0.0
```

要求：

- full 256×256 center input；
- high-resolution stages保持卷积/NAF；
- GroupNorm、LayerNorm2d 或当前稳定归一化；
- 不使用 BatchNorm；
- skip feature 可用于 decoder；
- 输入通道自动适配灰度/RGB/Sobel。

若已有 NAF 风格编码器，复用并添加清晰 adapter，不重新写一套近似重复代码。

## 5.3 Context tile encoder

对 9 个 tile 使用同一轻量编码器：

```text
Conv stem
→ depthwise residual blocks
→ 1/4 resolution
→ global average pooling
→ context token
```

输出：

```python
context_tokens: [B, 9, D]
```

加入：

- 二维相对位置 embedding；
- center 标志 embedding；
- valid mask；
- 可选 organ embedding。

上下文编码器参数应明显小于 local encoder，默认不超过主模型参数的 20%。

支持 `context_stop_gradient` 实验，但默认允许端到端训练。

## 5.4 Context fusion

实现两级融合。

### 5.4.1 多尺度 FiLM

在 local encoder 的 1/4、1/8、1/16 stage：

1. 对有效 context tokens 做 masked attention pooling；
2. 生成 `gamma, beta`；
3. 对 local feature 做：

```python
f = f * (1 + gamma) + beta
```

要求：

- gamma/beta 最后一层零初始化；
- 初始行为等价于无 context 基线；
- 支持 context dropout；
- context dropout 默认 0.1，增强对缺失邻居的鲁棒性。

### 5.4.2 Bottleneck cross-attention

local bottleneck tokens 作为 query，9 个 context tokens 作为 key/value：

```text
Q: [B, H*W, D]
K,V: [B, 9, D]
```

要求：

- 标准 PyTorch MultiheadAttention 或自实现明确版本；
- 对缺失邻居做 key padding mask；
- 位置 embedding；
- 输出以可学习小系数残差加入；
- 不允许显存随 256² 全局注意力爆炸。

## 5.5 全局恢复块

默认实现纯 PyTorch `RestorationTransformerBlock`：

- LayerNorm2d；
- Restormer 风格 channel attention 或高效 spatial mixer；
- depthwise convolution；
- gated feed-forward；
- residual scaling；
- 仅在 1/8 和 1/16 使用。

配置：

```yaml
global_mixer:
  type: restormer_lite
  blocks_1_8: 2
  blocks_1_16: 4
  heads: [4, 8]
```

可选：

```yaml
global_mixer:
  type: mambair_v2_lite
```

但必须：

- 缺少 mamba 依赖时优雅回退；
- 单元测试跳过可选分支；
- 不破坏默认安装；
- 不把外部 CUDA 扩展写入 core requirements。

## 5.6 Marker 和 organ 条件

实现：

```text
marker embedding
organ embedding
```

通过 FiLM 注入：

- bottleneck；
- 每个 decoder stage；
- intensity calibrator。

要求：

- marker 名称规范化；
- unknown organ 有独立 embedding；
- 单器官数据时 organ conditioning 可关闭；
- organ embedding 不得由测试像素统计推断，使用目录/manifest 已知元数据；
- 没有 organ 标签时使用 `unknown`。

FiLM 初始为 identity。

## 5.7 共享 decoder + 轻量专家

保留共享 decoder，但增强任务特异性：

每个 decoder stage：

```text
shared restoration block
+ marker residual adapter
+ optional organ residual adapter
+ optional 2-expert gated residual
```

默认只启用：

- shared block；
- marker adapter；
- organ adapter。

Mixture-of-experts 作为 P2 实验，不默认开启。

任务 adapter：

```text
1×1 reduce
→ depthwise 3×3
→ SimpleGate/GELU
→ 1×1 expand
→ zero-init residual
```

参数占比小，避免复制四套完整 decoder。

## 5.8 Hierarchical prototypes v2

若现有模型已经有 prototype module，在其基础上升级，不重复实现。

在 1/8 和 1/16 两个尺度设置：

- `P_shared`
- `P_marker[task]`
- 可选 `P_organ[organ]`

默认：

```yaml
prototypes:
  enabled: true
  scales: [8, 16]
  shared_count: 8
  marker_count: 8
  organ_count: 4
  dim: 128
  temperature: 0.1
  residual_init: 0.0
```

流程：

1. feature projection；
2. L2 normalize；
3. cosine similarity；
4. masked softmax；
5. prototype aggregation；
6. projection；
7. zero-init residual。

正则：

- usage entropy；
- pairwise orthogonality/diversity；
- activation/commitment；
- 权重极小；
- 所有损失可关闭。

监控：

```text
prototype_usage.csv
dead_prototypes.csv
prototype_similarity.npy
prototype_attention_visuals/
```

如果某 prototype 在连续若干 epoch 中使用率接近 0：

- 只记录；
- 默认不自动重置；
- 提供可选、确定性的训练期重置；
- 不在 resume 时破坏状态。

## 5.9 Laplacian base-detail decoder

将输出显式分为：

1. 低频 base；
2. 高频 detail。

实现方式建议：

```python
base_logits = upsample(low_resolution_head(feature_1_4))
detail_logits = detail_head(full_resolution_feature)
detail_logits = max_detail_amplitude * torch.tanh(detail_logits)
final_logits = base_logits + detail_logits
prediction = torch.sigmoid(final_logits)
```

默认：

```yaml
output:
  base_detail: true
  max_detail_amplitude: 1.0
  deep_supervision_scales: [1, 2, 4, 8]
```

需要输出中间结果用于可视化：

- base；
- detail；
- final。

不得在推理后单独锐化图像。

## 5.10 Global intensity calibrator

不同器官、ROI 和标记的整体强度分布可能不同。实现一个轻量、端到端训练的校准头：

输入：

- bottleneck global pooling；
- marker embedding；
- organ embedding。

输出：

- bounded logit gain；
- bounded logit bias。

```python
gain = 1 + max_gain_delta * tanh(raw_gain)
bias = max_bias * tanh(raw_bias)
calibrated_logits = gain * logits + bias
```

最后层零初始化，因此初始 `gain=1, bias=0`。

默认范围温和：

```yaml
intensity_calibrator:
  enabled: true
  max_gain_delta: 0.15
  max_bias: 0.15
```

禁止：

- 使用测试集均值/方差拟合；
- 对测试图逐张搜索最佳 gain/bias；
- 人工调整测试结果。

---

# 6. 官方数据自监督预训练

实现可选 DAPI-native masked reconstruction，不使用外部数据。

新增：

```text
src/virtual_staining/models/dapi_mae.py
src/virtual_staining/engine/pretrainer.py
configs/performance_v2/dapi_pretrain.yaml
```

## 6.1 原则

- 只使用当前 fold 的训练 DAPI；
- 不使用 val/test DAPI；
- 不使用任何目标标签；
- 不下载外部数据；
- 预训练 local encoder；
- context encoder 可选共同预训练。

## 6.2 任务

默认轻量 masked image modeling：

- block mask；
- mask ratio 0.4–0.6；
- reconstruct DAPI；
- L1/Charbonnier + SSIM；
- decoder 只用于预训练；
- 迁移 encoder 权重到 CAMP-VS v2。

可选邻域一致性：

- center embedding 与有效近邻 aggregate embedding；
- 小权重 cosine consistency；
- 不把同 ROI 所有 patch 强制完全相同；
- 默认关闭，先做消融。

## 6.3 是否进入最终配置

DAPI-MAE 只有满足以下条件才进入最终训练：

- 相同 seed；
- 相同训练轮数；
- 至少两次重复；
- official val 的 SSIM/PSNR或组合代理稳定提高；
- 不是仅加快收敛但最终无提升。

---

# 7. 损失函数 v2

新增：

```text
losses/
├─ pyramid.py
├─ statistics.py
├─ scheduled_composite.py
└─ shift_tolerant.py
```

## 7.1 主损失

```text
L =
w_mse      × MSE
+ w_charb  × Charbonnier
+ w_ssim   × (1 - SSIM)
+ w_msssim × (1 - MS-SSIM)
+ w_pyr    × LaplacianPyramidLoss
+ w_grad   × GradientLoss
+ w_stats  × IntensityStatisticsLoss
+ auxiliary losses
```

不默认使用：

- GAN loss；
- VGG perceptual；
- LPIPS；
- CLIP/UNI/CONCH feature loss。

## 7.2 MSE 的作用

官方 PSNR 直接由 MSE 决定，因此 v2 必须显式加入 MSE，尤其在后期 metric-alignment fine-tune 中。

不要全程只用 MSE，避免初期过度平滑。

## 7.3 Laplacian pyramid loss

使用固定 Gaussian kernel，构建：

```text
level 0: full
level 1: 1/2
level 2: 1/4
level 3: 1/8
```

比较：

- Gaussian levels；
- Laplacian residual levels。

默认 level 权重：

```yaml
[1.0, 0.5, 0.25, 0.125]
```

支持灰度和 RGB。

## 7.4 Intensity statistics loss

每张图、每通道计算：

- mean；
- standard deviation；
- 可选 low-frequency pooled map。

默认：

```text
L_stats = L1(mean_pred, mean_gt) + 0.5 × L1(std_pred, std_gt)
```

权重很小，防止只匹配全局统计而丢失空间结构。

## 7.5 两阶段损失调度

### Phase A：稳健结构学习

训练前 70% epoch，初始建议：

```yaml
mse:       0.10
charb:     0.35
ssim:      0.30
ms_ssim:   0.10
pyramid:   0.10
gradient:  0.03
statistics:0.02
```

### Phase B：官方指标对齐

最后 30% epoch，初始建议：

```yaml
mse:       0.35
charb:     0.15
ssim:      0.35
ms_ssim:   0.05
pyramid:   0.07
gradient:  0.02
statistics:0.01
```

使用线性或 cosine 插值过渡，不要突然切换造成 loss spike。

以上只是初始候选，必须通过 ablation 选择。

## 7.6 Shift-tolerant loss

只有数据审计确认存在小偏移才启用。

候选：

```text
shifts = {-1,0,+1} × {-1,0,+1}
```

对每个 shift 计算低分辨率 Charbonnier/SSIM，使用：

- softmin；
- 或 stop-gradient 选择最小 shift。

限制：

- 只在训练 loss 中；
- 输出本身不平移；
- 最大 shift 可配置且默认 1；
- 推理不使用；
- 若配对高度对齐，关闭。

## 7.7 Prototype 和多任务损失

默认极小：

```yaml
prototype_activation: 0.0005
prototype_diversity: 0.0005
prototype_usage_entropy: 0.0002
```

FAMO 只作用于四个 task reconstruction losses，不把所有 auxiliary loss 当作独立 task。

---

# 8. 训练策略：多任务预训练 → 目标专属 → 器官专属 → 指标对齐

新增：

```text
src/virtual_staining/engine/multistage_trainer.py
configs/performance_v2/stages/
```

## 8.1 Stage 0：复现当前基线

目的：

- 确认旧模型在当前代码环境仍可复现；
- 冻结基准；
- 不做新模型比较前跳过此步。

## 8.2 Stage 1：多标记、多器官预训练

使用所有**当前官方已经发布且允许使用**的训练标记和器官。

目标：

- 学习共享 DAPI 组织表示；
- 学习 marker embedding；
- 学习共享原型；
- 学习 context。

采样：

- 先按 organ 均衡或平方根反频率；
- 再按 marker 均衡；
- 同一 batch 可混合任务；
- 记录每任务 loss。

可选：

- equal；
- FAMO。

默认先 equal。

## 8.3 Stage 2：当前目标标记专属微调

从 Stage 1 权重初始化：

- 保留 local/context encoder；
- 保留共享 decoder；
- 只激活指定 marker head/adapter；
- 其他 marker head 不参与推理；
- 可先冻结 encoder 5 epoch，再全部解冻；
- 学习率降低到 Stage 1 的 0.25–0.5。

目的：

- 消除多任务负迁移；
- 针对当前 leaderboard marker 优化。

## 8.4 Stage 3：器官专属 adapter 微调

若当前阶段器官已知：

- colon、liver、stomach 分别保存专属 adapter；
- 共享主干从多器官模型继承；
- 微调 organ adapter、marker adapter、decoder 后半部分和 calibrator；
- 小学习率；
- 防止小器官数据过拟合。

对于 stomach：

- 数据量较小但半决赛权重高；
- 优先使用 multi-organ pretrain；
- 适度 oversample stomach；
- 可在 batch 中保留少量其他器官 replay；
- 以 stomach official val 为主选择模型；
- 不因 colon 样本多而让训练目标被 colon 支配。

## 8.5 Stage 4：metric alignment fine-tune

最后 10–30 epoch：

- 使用 Phase B loss；
- 学习率降到原来的 0.05–0.1；
- 保持 EMA；
- 可启用 SWA 或 checkpoint averaging；
- 不再增加强增强；
- 不再修改架构。

## 8.6 活性分层采样

部分标记可能大面积低信号。预先在 train GT 上计算：

- mean；
- std；
- 95% quantile；
- gradient energy；
- foreground/activity proxy。

将训练样本分为低、中、高 activity bin。

Sampler：

- ROI balanced；
- activity stratified；
- 不让低信号背景样本完全主导；
- 也不要极端过采高信号，默认 bin 权重温和。

所有统计仅来自训练 split。

## 8.7 优化器

默认：

```yaml
optimizer: adamw
lr: 2.0e-4
weight_decay: 1.0e-4
warmup_epochs: 5
scheduler: cosine
grad_clip: 1.0
ema_decay: 0.999
```

Stage 2–4 使用更小 lr。

保留：

- AMP；
- gradient accumulation；
- OOM 自适应；
- resume。

## 8.8 正则

默认：

- no dropout in high-resolution restoration blocks；
- context token dropout 0.1；
- stochastic depth 0–0.05；
- weight decay；
- mild DAPI intensity augmentation；
- D4 geometry。

禁止默认使用：

- MixUp；
- CutMix；
- 任意角旋转；
- 大幅弹性形变；
- 重度颜色抖动；
- 随机 resize；
- 会破坏 center/context 几何关系的增强。

---

# 9. 验证体系：严禁用一个随机 split 判断涨分

## 9.1 划分原则

优先逻辑：

1. 若官方 train/val 已提供：
   - official val 作为外层最终验证；
   - official train 内部按 ROI 做 inner grouped folds；
2. 若官方未提供 val：
   - 按 ROI 做 grouped K-fold；
3. 绝不按 patch 随机划分。

检查：

- 同 ROI 跨 split；
- 重复图像；
- 邻接 patch 泄漏；
- hash 重复。

## 9.2 指标域

每次验证都计算三套：

1. float `[0,1]`；
2. uint8；
3. 使用最终 JPG 参数保存后重新读取。

模型选择以**最终 JPG round-trip 指标**作为最重要参考，因为官方评测读取提交文件。

JPG：

```text
quality=100
subsampling=0
optimize=False
```

除非官方另有要求。

## 9.3 分组指标

至少报告：

- marker；
- organ；
- ROI；
- activity bin；
- context availability；
- image mean bin；
- 最差 10%；
- border patch vs interior patch；
- float/uint8/JPG 差异。

## 9.4 官方综合分代理

官方 PSNR Normalize 的具体边界若未知，不伪造。

同时输出：

```text
raw_ssim
raw_psnr
rank_proxy
configurable_official_proxy
```

`rank_proxy` 建议使用：

- validation 内 SSIM percentile rank；
- PSNR percentile rank；
- `0.7 × ssim_rank + 0.3 × psnr_rank`。

若用户提供官方 Normalize 公式，再切换为精确代理。

## 9.5 多器官权重

若 colon/liver/stomach 都有合法验证数据，额外输出：

```text
weighted_ssim = 0.1*colon + 0.2*liver + 0.7*stomach
weighted_psnr = 0.1*colon + 0.2*liver + 0.7*stomach
```

并用相同逻辑计算代理分。

## 9.6 Bootstrap 置信区间

以 ROI 为重采样单位，输出：

- 指标差值；
- 95% bootstrap CI；
- win/tie/loss ROI 数。

一个改进要进入默认配置，至少满足：

- 无泄漏；
- 两个 seed 或两个 fold 中趋势一致；
- SSIM 和 PSNR 至少一项明确提高；
- 另一项不出现明显下降；
- ROI bootstrap 不显示提升完全由极少数 ROI 驱动；
- JPG 指标提升而不只是 float 指标。

不承诺任何固定涨分数值。

---

# 10. 消融实验优先级

新增：

```text
configs/performance_v2/ablation/
artifacts/performance_v2/ablation_registry.csv
docs/PERFORMANCE_V2_ABLATION.md
```

## 10.1 P0：最高优先级

按顺序：

### A0
当前最佳模型复现。

### A1
当前模型 + MSE/SSIM 两阶段 loss schedule。

### A2
A1 + Laplacian pyramid loss / base-detail head。

### A3
A2 + ROI 3×3 context encoder + zero-init FiLM。

### A4
A3 + bottleneck context cross-attention。

### A5
A4 + target-specific fine-tune。

### A6
A5 + organ-specific adapter fine-tune。

### A7
A6 + D4 TTA。

### A8
多 seed / compatible checkpoint averaging / prediction ensemble。

P0 是默认实施重点。

## 10.2 P1：高价值但需验证

- hierarchical prototypes v2；
- FAMO；
- DAPI-MAE；
- activity stratified sampler；
- global intensity calibrator；
- Restormer-lite bottleneck；
- grouped inner CV；
- learned ensemble weights。

## 10.3 P2：可选高级实验

- MambaIRv2-lite bottleneck；
- mixture-of-experts decoder；
- shift-tolerant loss；
- range-wise ensemble；
- neighbor consistency self-supervision；
- 5×5 context。

## 10.4 P3：不要优先

- GAN；
- diffusion；
- large pathology foundation model；
- external pretraining data；
-复杂 registration GAN。

只有 P0/P1 已经完成且验证预算充足，才考虑 P2/P3。

---

# 11. 自动化实验调度

新增 CLI：

```text
benchmark-baseline
audit-roi-grid
pretrain-dapi
train-v2
finetune-target
finetune-organ
finetune-metric
run-ablation
compare-runs
build-model-soup
optimize-ensemble
predict-v2
```

示例：

```bash
python -m virtual_staining.cli benchmark-baseline \
  --config configs/competition_multitask.yaml

python -m virtual_staining.cli audit-roi-grid \
  --manifest artifacts/manifests/train_manifest.csv

python -m virtual_staining.cli train-v2 \
  --config configs/performance_v2/camp_multitask.yaml

python -m virtual_staining.cli finetune-target \
  --config configs/performance_v2/camp_target_finetune.yaml \
  --target CD68 \
  --checkpoint outputs/performance_v2/.../best.ckpt

python -m virtual_staining.cli run-ablation \
  --suite configs/performance_v2/ablation/p0.yaml
```

## 11.1 预算分级

实现：

```yaml
budget:
  smoke:
  screen:
  confirm:
  full:
```

### smoke
- 数十样本；
- 1–2 epoch；
- 验证代码链。

### screen
- 一个 inner fold；
- 20–40 epoch；
- 固定 seed；
- 用于淘汰明显无效模块。

### confirm
- 两个 folds 或两个 seeds；
- 80–120 epoch；
- top candidates。

### full
- 最终完整 schedule；
- official val；
- 多 seed；
- final ensemble。

不要用 smoke 结果声称算法涨分。

## 11.2 实验注册表

每次 run 写入：

```text
artifacts/performance_v2/experiment_registry.csv
```

字段：

- run_id
- parent_run
- git_commit
- config_hash
- manifest_hash
- model
- target
- organ
- fold
- seed
- context
- pretrain
- prototype
- task_optimizer
- loss_schedule
- params
- flops
- peak_vram
- train_time
- float_ssim
- float_psnr
- uint8_ssim
- uint8_psnr
- jpg_ssim
- jpg_psnr
- weighted_score_proxy
- checkpoint
- status
- failure_reason

---

# 12. Ensemble、checkpoint averaging 与 model soup

## 12.1 候选成员

最终候选模型应保持多样性，而不是只平均完全相同的 checkpoint：

1. 当前最佳旧模型；
2. local-only CAMP；
3. context CAMP；
4. multitask-pretrained + target-finetuned CAMP；
5. organ-adapted CAMP；
6. 不同 seed；
7. 可兼容的 loss schedule variants。

## 12.2 Prediction ensemble

使用 validation/OOF 预测拟合：

```text
w_i >= 0
sum(w_i) = 1
```

优化目标：

- configurable score proxy；
- 或 SSIM/PSNR Pareto；
- 仅使用验证集；
- 一组全局权重；
- 可按 marker/organ 单独拟合；
- 不允许逐测试图选择模型。

实现：

- uniform average；
- coordinate search；
- SLSQP/simplex；
- 交叉验证 ensemble weights；
- 过拟合保护。

默认先 uniform，再尝试 learned weights。

## 12.3 Greedy model soup

只对：

- 完全相同架构；
- 相同参数 key；
- 同一初始化或同一预训练起点；
- 相近 fine-tune basin；

执行权重平均。

Greedy soup：

1. 按验证指标排序；
2. 从最好模型开始；
3. 逐一尝试加入；
4. 只有 validation JPG 指标不下降才保留。

注意：

- GroupNorm/LayerNorm 不需要重估 BN；
- 若模型有非参数状态，正确处理；
- prototype bank、EMA 权重和 calibrator 必须一致加载；
- soup 后重新完整验证；
- soup 不等于 prediction ensemble。

## 12.4 EMA、SWA 与 soup

分别输出：

- raw；
- EMA；
- SWA；
- greedy soup；
- prediction ensemble。

只保留实际提高者。

## 12.5 Range-wise ensemble

作为 P2：

- 根据 prediction mean/std/activity proxy 分 bin；
- 权重只在 OOF/val 拟合；
- 测试时依据模型预测本身选择固定 LUT；
- 不能使用 test GT；
- 若收益不稳定，关闭。

---

# 13. 配置文件

新增：

```text
configs/performance_v2/
├─ camp_smoke.yaml
├─ camp_local_only.yaml
├─ camp_context.yaml
├─ camp_multitask.yaml
├─ camp_target_finetune.yaml
├─ camp_organ_finetune.yaml
├─ camp_metric_finetune.yaml
├─ camp_infer.yaml
├─ dapi_pretrain.yaml
├─ ensemble.yaml
└─ ablation/
   ├─ p0.yaml
   ├─ p1.yaml
   └─ p2.yaml
```

`camp_multitask.yaml` 核心初始值：

```yaml
model:
  name: camp_vs_v2

  local_encoder:
    type: naf
    widths: [48, 96, 192, 384]
    depths: [2, 2, 4, 6]

  context:
    enabled: true
    grid_size: 3
    token_dim: 192
    encoder_width: 32
    context_dropout: 0.10
    fusion_scales: [4, 8, 16]
    bottleneck_cross_attention: true
    residual_init: 0.0

  global_mixer:
    type: restormer_lite
    blocks_1_8: 2
    blocks_1_16: 4
    heads_1_8: 4
    heads_1_16: 8

  conditioning:
    marker_embedding: true
    organ_embedding: true
    film: true
    zero_init: true

  adapters:
    marker: true
    organ: true
    mixture_of_experts: false

  prototypes:
    enabled: true
    scales: [8, 16]
    shared_count: 8
    marker_count: 8
    organ_count: 4
    dim: 128
    temperature: 0.10
    residual_init: 0.0

  output:
    base_detail: true
    max_detail_amplitude: 1.0
    deep_supervision: true

  intensity_calibrator:
    enabled: true
    max_gain_delta: 0.15
    max_bias: 0.15

loss:
  schedule: two_phase
  phase_a_ratio: 0.70

  phase_a:
    mse: 0.10
    charbonnier: 0.35
    ssim: 0.30
    ms_ssim: 0.10
    pyramid: 0.10
    gradient: 0.03
    statistics: 0.02

  phase_b:
    mse: 0.35
    charbonnier: 0.15
    ssim: 0.35
    ms_ssim: 0.05
    pyramid: 0.07
    gradient: 0.02
    statistics: 0.01

  prototype:
    activation: 0.0005
    diversity: 0.0005
    usage_entropy: 0.0002

  shift_tolerant:
    enabled: false
    max_shift: 1

multitask:
  optimizer: equal
  log_gradient_cosine_every: 500

train:
  stages:
    - multitask_pretrain
    - target_finetune
    - organ_finetune
    - metric_finetune
  optimizer: adamw
  lr: 0.0002
  weight_decay: 0.0001
  warmup_epochs: 5
  scheduler: cosine
  amp: true
  ema: true
  ema_decay: 0.999
  grad_clip: 1.0
  batch_size: auto
  gradient_accumulation: auto

validation:
  primary_domain: jpg
  group_by_roi: true
  bootstrap_by_roi: true
  bootstrap_samples: 1000
  save_predictions: true

inference:
  use_ema: true
  tta: d4
  context: true
  jpg_quality: 100
  jpg_subsampling: 0
```

解析后生成 `configs/performance_v2/resolved_local.yaml`，不覆盖模板。

---

# 14. 单元测试

新增或扩展：

```text
tests/
├─ test_roi_index.py
├─ test_neighborhood.py
├─ test_context_transforms.py
├─ test_context_encoder.py
├─ test_context_fusion.py
├─ test_camp_shapes.py
├─ test_hierarchical_prototypes.py
├─ test_laplacian_decoder.py
├─ test_pyramid_loss.py
├─ test_loss_schedule.py
├─ test_intensity_calibrator.py
├─ test_famo_adapter.py
├─ test_multistage_resume.py
├─ test_jpg_metrics.py
├─ test_model_soup.py
└─ test_ensemble_optimizer.py
```

必须覆盖：

1. ROI 坐标解析；
2. 方向审计；
3. 3×3 邻域；
4. 缺失邻居；
5. 不跨 split；
6. 不跨 ROI；
7. hflip/vflip/rot90 后邻居位置；
8. context mask；
9. local-only 和 context 模式；
10. 单任务 shape；
11. 四任务 shape；
12. 灰度/RGB；
13. context cross-attention mask；
14. zero-init context 与 baseline 初始等价性；
15. prototype finite；
16. prototype loss finite；
17. base/detail reconstruction；
18. calibrator 初始 identity；
19. two-phase loss 连续性；
20. forward/backward；
21. AMP；
22. checkpoint resume 包含 stage；
23. D4 TTA 与 context 坐标同步；
24. JPG round-trip metric；
25. soup 参数 key；
26. ensemble 权重非负且和为 1；
27. submission 文件名未变化；
28. Windows `num_workers=0`；
29. 中文路径；
30. smoke pipeline。

运行：

```bash
python -m compileall src tests
pytest -q
```

---

# 15. 性能与显存约束

## 15.1 参数预算

默认目标：

- 主模型参数量不超过当前主模型的约 1.5–2.0 倍；
- context encoder 不超过总参数 20%；
- 256×256 可训练；
- 8–16GB 显存可通过 accumulation 运行。

如果超预算：

1. context token dim 降低；
2. context encoder width 降低；
3. global blocks 减少；
4. base channels 48→40/32；
5. 不先砍掉 full-resolution center 输入。

## 15.2 性能分析

输出：

```text
artifacts/performance_v2/complexity_report.json
```

包含：

- params；
- MACs/FLOPs；
- peak VRAM；
- images/s；
- context preprocessing time；
- D4 TTA cost；
- ensemble cost。

## 15.3 Context 缓存

context tile 会重复读取。实现可选：

- 文件路径索引；
- Pillow/OpenCV 正确关闭；
- 小型 LRU；
- 不把全部数据强制载入内存；
- 不缓存增强后的张量；
- 多进程安全；
- Windows 默认 workers=0。

---

# 16. 错误闭环

实现阶段按以下顺序：

1. 当前测试；
2. baseline snapshot；
3. ROI grid audit；
4. neighborhood dataset；
5. neighborhood tests；
6. context encoder；
7. CAMP shape/backward；
8. loss v2；
9. smoke；
10. 真实小预算 A1；
11. A2；
12. A3；
13. 其余 P0；
14. confirm；
15. full；
16. ensemble；
17. submission validation。

遇到错误时主动检查：

- 旧 config 与新 schema 兼容；
- checkpoint key；
- context 张量维度；
- row/col 变换；
- mask dtype；
- MultiheadAttention batch_first；
- 灰度通道；
- HWC/CHW；
- LayerNorm2d；
- MS-SSIM 最小尺寸；
- loss schedule resume；
- EMA 加载；
- soup 使用 raw 还是 EMA；
- float/uint8/JPG 指标；
- D4 context coordinate transform；
- OOM；
- Windows multiprocessing；
- 中文路径；
- submission stem。

不得通过捕获所有 Exception 隐藏问题。

---

# 17. 失败判定与回滚

以下情况视为模块失败：

- 只提高训练集；
- float 提高但 JPG 下降；
- 单 seed 提高、重复后消失；
- SSIM 微升但 PSNR 大幅下降；
- context 只改善 interior patch、严重损害 border；
- prototype 使用坍缩；
- FAMO 让目标 marker 下降；
- DAPI-MAE 只加快早期收敛但最终无提升；
- Mamba 分支依赖不稳定；
- learned ensemble 在 OOF 与 official val 方向相反；
- 推理成本超出官方环境；
- 代码无法复现。

失败模块：

- 配置默认关闭；
- 保留清晰实验记录；
- 不强行写入最终模型；
- README 标注结果。

---

# 18. 最终模型选择建议

按验证结果，优先形成三类候选：

## Candidate A：稳健单模型

```text
CAMP context
+ two-phase MSE/SSIM
+ Laplacian base-detail
+ target fine-tune
+ organ adapter
+ EMA
+ D4
```

## Candidate B：多任务创新模型

```text
Candidate A
+ shared/marker hierarchical prototypes
+ multitask pretraining
+ marker/organ FiLM
+ prototype visualizations
```

用于：

- 机器分；
- 联合建模证明；
- 技术报告；
- 创新附加分。

## Candidate C：最终集成

```text
old best
+ Candidate A seed 1/2/3
+ Candidate B
→ validation-fitted nonnegative ensemble
```

正式提交只输出当前 marker。

---

# 19. 文档

新增：

```text
docs/
├─ RESEARCH_TO_CODE_MAP.md
├─ PERFORMANCE_V2_BASELINE.md
├─ PERFORMANCE_V2_MODEL.md
├─ PERFORMANCE_V2_ABLATION.md
├─ ROI_CONTEXT_DESIGN.md
├─ MULTISTAGE_TRAINING.md
├─ ENSEMBLE_STRATEGY.md
├─ PERFORMANCE_V2_FAILURES.md
└─ PERFORMANCE_V2_FINAL_REPORT.md
```

`RESEARCH_TO_CODE_MAP.md` 必须说明：

- ProtoMTG → hierarchical shared/task prototypes；
- PyramidPix2Pix → pyramid supervision；
- NAFNet → local restoration；
- Restormer/MambaIRv2 → low-resolution global context；
- HookNet → local/context dual branch思想；
- UNIStainNet → spatial modulation + marker embedding，但不用外部 H&E foundation model；
- FAMO → optional task balancing；
- registration literature → audit-gated shift-tolerant loss；
- model soups / ensemble → final generalization。

明确哪些模块：

- 已采用；
- 重新实现；
- 仅借鉴思想；
- 未采用及原因；
- 许可证和代码来源。

不要直接复制无许可证仓库代码。

---

# 20. AGENTS.md 增补规则

将以下规则增量写入现有 `AGENTS.md`，不要删除旧规则：

- performance-v2 是增量升级，不重写稳定基线；
- 所有新模块必须 feature flag；
- 不得跨 split 加载邻居；
- context 只使用 DAPI；
- 正式模型选择使用 JPG round-trip 指标；
- 一项模块未通过 ROI-grouped ablation 不进入默认；
- 禁止用 test 数据拟合；
- 禁止默认 GAN/diffusion；
- 修改 neighborhood 后必须运行 context transform tests；
- 修改模型后运行 shape/backward/smoke；
- 修改 loss 后运行 numerical tests；
- 修改 ensemble 后验证权重和 OOF；
- 不声称未完成的 full train；
- 最终回复必须列出实际运行命令、结果、失败实验和保留配置。

---

# 21. 最终交付物

最终至少交付：

1. 可运行的 CAMP-VS v2；
2. ROI context 数据管线；
3. loss v2；
4. multistage training；
5. target/organ fine-tune；
6. DAPI-MAE 可选分支；
7. FAMO 可选分支；
8. ensemble 和 soup；
9. 全部测试；
10. configs；
11. 文档；
12. baseline 对比；
13. ablation；
14. 最佳 checkpoint；
15. final submission；
16. submission validator；
17. 完整最终报告。

若完整训练受硬件限制，仍需完成：

- 全部代码；
- 全部测试；
- smoke；
- 至少一组真实数据 screen 实验；
- 明确给出 full 命令；
- 不伪造最终指标。

---

# 22. 最终验收门槛

只有下列全部满足，才允许声称“性能升级工程完成”：

- [ ] 当前旧测试未被破坏；
- [ ] `compileall` 成功；
- [ ] `pytest -q` 成功；
- [ ] baseline snapshot 完成；
- [ ] ROI grid audit 完成；
- [ ] 邻域方向验证完成；
- [ ] 无跨 split context 泄漏；
- [ ] CAMP local-only forward/backward 成功；
- [ ] CAMP context forward/backward 成功；
- [ ] 单任务/多任务成功；
- [ ] base/detail 输出正确；
- [ ] loss schedule 可 resume；
- [ ] smoke train 成功；
- [ ] smoke infer 成功；
- [ ] JPG round-trip metrics 成功；
- [ ] 至少完成 A0–A3 的 screen 对比；
- [ ] 未涨分模块默认关闭；
- [ ] target fine-tune 可运行；
- [ ] organ adapter 可运行；
- [ ] ensemble 只使用 val/OOF；
- [ ] submission validator 通过；
- [ ] 正式提交只输出当前 marker；
- [ ] 没有 TODO/pass；
- [ ] 文档列出真实运行结果；
- [ ] 未伪造完整训练或 leaderboard 结果。

---

# 23. Plan 模式现在必须输出的内容

现在先进行只读检查，并按以下格式回复用户：

1. **当前仓库真实结构**
2. **当前基线功能清单**
3. **当前测试状态**
4. **当前最佳模型与指标**
5. **数据与 ROI 网格判断**
6. **现有模型和 CAMP-VS v2 的差异**
7. **可以复用的模块**
8. **需要新增/修改的文件**
9. **P0/P1/P2 实施顺序**
10. **实验预算与硬件适配**
11. **风险与回滚方案**
12. **准备运行的命令**
13. **唯一无法自动判断的事项**

不要在 Plan 阶段直接重写项目。

若当前仓库缺少某个预期模块，不要停止；在计划中说明如何兼容。若数据、配置和目标都能自动推断，不要先询问用户。

---

# 24. 用户批准后的第一条执行链

批准后，严格从以下动作开始：

```text
1. 保存基线快照
2. 运行全部旧测试
3. 创建 performance-v2 输出目录
4. 审计 ROI 网格与边界连续性
5. 实现 ROI index 和 neighborhood
6. 运行 neighborhood tests
7. 实现 CAMP local-only
8. 验证加载旧权重可行性
9. 实现 context branch 与 zero-init fusion
10. 实现 loss v2
11. 运行完整 smoke
12. 运行 A0/A1/A2/A3 screen
13. 根据真实结果决定后续 P0/P1
```

不得跳过基线直接宣称新模型更好。


---

# 研究参考清单（供实现和文档引用，不要求照搬代码）

1. ProtoMTG: Prototypical Multi-Task Learning for the Generation of Multiple Stained Immunohistochemical Images, IEEE TMI 2025.
2. PSPStain: Pathological Semantics-Preserving Learning for H&E-to-IHC Virtual Staining, MICCAI 2024.
3. PGVMS: A Prompt-Guided Unified Framework for Virtual Multiplex IHC Staining with Pathological Semantic Learning, IEEE TMI 2026.
4. UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC, 2026 preprint.
5. Generative AI for Misalignment-Resistant Virtual Staining to Accelerate Histopathology Workflows, Nature Communications 2026.
6. High-resolution Medical Image Translation via Patch Alignment-Based Bidirectional Contrastive Learning, MICCAI 2024.
7. BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix, CVPR Workshop 2022.
8. Simple Baselines for Image Restoration / NAFNet, ECCV 2022.
9. Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022.
10. MambaIRv2: Attentive State Space Restoration, CVPR 2025.
11. HookNet: Multi-resolution Convolutional Neural Networks for Histopathology, Medical Image Analysis 2021.
12. Masked Autoencoders Are Scalable Vision Learners, CVPR 2022.
13. FAMO: Fast Adaptive Multitask Optimization, NeurIPS 2023.
14. Model Soups, ICML 2022.
15. EnsIR: Ensemble Algorithm for Image Restoration, NeurIPS 2024.

实现原则：
- 优先读取论文与官方仓库；
- 记录许可证；
- ProtoMTG 等仓库若许可证或完整性不明确，只做 clean-room reimplementation；
- 不新增外部训练数据；
- 所有研究模块以本赛题 ROI-grouped SSIM/PSNR 实验结果为最终依据。
