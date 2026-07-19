# 2026 AIC「基于虚拟染色的免疫组化图像生成」完整工程构建任务

你现在处于 Codex 的 Plan 模式。请把本任务视为一个必须可运行、可训练、可验证、可推理、可打包提交的正式算法竞赛工程，而不是只生成示例代码或伪代码。

## 0. 工作方式与总目标

你的工作分为两个连续阶段：

1. **Plan 阶段**
   - 先只读检查当前工作区、现有文件、本地数据目录、Python/CUDA/PyTorch 环境。
   - 输出详细实施计划、风险、自动推断结果和验收标准。
   - 除非确实找不到数据或出现无法由代码自动判断的提交目标，否则不要反复询问用户。
   - 不要因为原始目录不标准而停止；应设计自动发现和兼容逻辑。

2. **Implementation 阶段**
   - 当用户批准计划或切换出 Plan 模式后，立即按照计划完整实现。
   - 不要只创建骨架，不要留下 TODO、pass、伪代码、空函数或“稍后实现”。
   - 持续运行测试、定位错误、修复并重跑，直到所有验收门槛通过。
   - 不要在首次报错后把问题甩给用户。先自行检查路径、数据格式、依赖、张量形状、图像通道、Windows 多进程和 CUDA 显存等常见问题。
   - 最终必须给出一份明确的运行报告：创建了什么、测试了什么、哪些命令成功、完整训练命令是什么、提交文件在哪里。

本项目的核心目标是：

- 输入：DAPI 染色的 256×256 图像 patch。
- 输出：与输入空间尺寸完全一致的目标 mIHC 标记图像。
- 目标标记可能包括：`HLA-DR`、`CD45RO`、`Vimentin`、`CD68`。
- 训练数据可能按器官分为 `colon`、`liver`、`stomach`。
- 竞赛主要评价 SSIM 与 PSNR，目标是构建优先优化这两个指标的稳定模型。
- 同时支持：
  - 单目标训练与提交；
  - 四目标联合训练；
  - 从联合模型中只导出指定目标；
  - 多折、多种子、EMA、TTA 和检查点集成。
- 输出提交目录必须兼容：
  `results/test/<TARGET>/<原输入文件名去后缀>_fake.jpg`

## 1. 不可违反的原则

1. 只使用赛事官方数据。
2. 禁止使用测试标签、隐藏测试信息或人工修改测试结果。
3. 测试集图像不得参与训练、归一化统计、自监督预训练、伪标签训练或模型选择。
4. 允许使用开源框架，但主方案不要依赖必须联网下载的权重。
5. 不要直接照搬原始 ProtoMTG 仓库。该公开仓库存在缺失配置和入口不完整风险，只可参考其思想。
6. 所有路径使用 `pathlib.Path`，必须支持：
   - Windows；
   - 含中文、空格的路径；
   - 相对路径与绝对路径；
   - 大小写差异和连字符差异，例如 `HLA_DR`、`HLA-DR`。
7. 不要硬编码本机盘符、用户名、数据绝对路径或 GPU 编号。
8. 不要默认 `num_workers>0`。Windows 默认配置使用 `num_workers=0`，用户可在配置中提高。
9. 不允许把“官方综合分”伪造成本地可精确复现的数值。官方 PSNR 归一化方式未知时：
   - 必须报告原始 SSIM、PSNR；
   - 可以报告明确标记为 `local_proxy_score` 的代理分；
   - 代理分归一化上下界必须可配置；
   - README 中说明代理分不等于官方分。
10. 推理必须是确定性的。固定随机种子；默认关闭随机采样式生成。
11. 默认主线不要使用 GAN 或扩散模型。先构建更适合 SSIM/PSNR、小样本和稳定复现的确定性图像恢复模型。
12. 不要静默改变图像尺寸、通道数、颜色模式、文件数量或文件名。

## 2. 首先执行：工作区与数据自动审计

在写模型前，先实现并运行数据审计。

### 2.1 数据根目录发现

按以下优先级寻找数据：

1. 用户现有配置中声明的目录；
2. 环境变量 `AIC_DATA_ROOT`；
3. 当前仓库下名称包含 `dataset`、`datasets`、`data`、`比赛数据`、`训练数据` 的目录；
4. 当前仓库父目录下一层的候选目录；
5. 用户给出的路径。

不要递归扫描整个系统盘。只扫描工作区及其父目录一层，避免耗时和隐私风险。

若找到多个候选目录，依据以下特征打分并选择最可能者：

- 含 `DAPI` 目录；
- 含目标标记目录；
- 含 `train`、`val`、`test`；
- 含大量 `.jpg/.jpeg/.png/.tif/.tiff`；
- 文件名形如 `ROI000_00_00.jpg`。

把候选、打分和最终选择写入 `artifacts/data_discovery.json`。

### 2.2 自动识别目录结构

不要假设目录严格等于某一种形式。兼容：

- `organ/split/marker/*.jpg`
- `split/organ/marker/*.jpg`
- `split/marker/*.jpg`
- `organ/marker/*.jpg`
- 训练集和验证集已经单独划分；
- 只有 train，需要自行划分；
- 测试集只有 DAPI；
- DAPI 和各标记文件扩展名不同；
- 目标图像为灰度、RGB 伪彩色或单通道保存为 RGB。

实现规范化标记名称函数：

- `hla_dr` → `HLA-DR`
- `cd45ro` → `CD45RO`
- `vimentin` → `Vimentin`
- `cd68` → `CD68`
- `dapi` → `DAPI`

### 2.3 构建 manifest

生成：

- `artifacts/manifests/train_manifest.csv`
- `artifacts/manifests/val_manifest.csv`
- `artifacts/manifests/test_manifest.csv`
- `artifacts/data_audit.json`
- `docs/DATA_AUDIT.md`

每行至少包含：

- `organ`
- `split`
- `roi_id`
- `patch_id`
- `canonical_key`
- `dapi_path`
- 四个目标路径列
- 图像宽高
- DAPI 通道数
- 每个目标通道数
- 文件格式
- 是否成功配对
- 文件大小
- 快速校验和，可使用 SHA1 前若干位

配对必须基于规范化 stem，不可依赖目录遍历顺序。

### 2.4 防止数据泄漏

文件名类似 `ROI000_00_00` 时，`ROI000` 视为组 ID。

- 若官方已提供 train/val：保持官方划分，并检查同一 ROI 是否跨 split。
- 若没有 val：使用 `GroupShuffleSplit` 或自实现固定种子的组划分，按 `roi_id` 分组，默认 80/20。
- 禁止随机按 patch 划分，因为相邻 patch 极易泄漏。
- 输出泄漏报告：
  - train/val 重复 canonical key；
  - train/val 重复哈希；
  - 同 ROI 跨 split；
  - 完全相同图像；
  - 缺失目标；
  - 破损图像。

发现破损或缺失时：
- 默认剔除训练 manifest 中不完整样本；
- 记录到 `artifacts/bad_samples.csv`；
- 不删除原始文件；
- 若缺失比例超过 1%，在最终报告中突出警告。

### 2.5 数据统计

对训练数据分标记统计：

- 数量；
- shape；
- mode；
- min/max/mean/std；
- 1%、5%、50%、95%、99% 分位数；
- DAPI 与目标的 Pearson/Spearman 粗略相关；
- 目标图像颜色通道差异；
- 目标是否本质为单通道伪彩色；
- 背景占比；
- 边缘强度；
- 每个 ROI 的 patch 数。

随机生成不少于 16 组配对可视化到：
`artifacts/figures/data_pairs/`

必须检查 DAPI 与目标是否空间对齐。使用边缘互相关或结构相似性做粗略诊断，并在报告中区分：

- 同一切片同坐标、近似对齐；
- 存在小幅偏移；
- 严重错配。

不要自动做会改变比赛标签的配准；仅报告并提供可选训练鲁棒策略。

## 3. 工程结构

创建以下结构，可在已有仓库上合理调整，但功能必须完整：

```text
project_root/
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
├─ requirements-core.txt
├─ requirements-dev.txt
├─ .gitignore
├─ configs/
│  ├─ default.yaml
│  ├─ smoke.yaml
│  ├─ baseline_unet.yaml
│  ├─ competition_single.yaml
│  ├─ competition_multitask.yaml
│  └─ infer.yaml
├─ src/
│  └─ virtual_staining/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ config.py
│     ├─ constants.py
│     ├─ data/
│     │  ├─ discovery.py
│     │  ├─ audit.py
│     │  ├─ manifest.py
│     │  ├─ dataset.py
│     │  ├─ transforms.py
│     │  └─ samplers.py
│     ├─ models/
│     │  ├─ registry.py
│     │  ├─ baseline_unet.py
│     │  ├─ naf_blocks.py
│     │  ├─ prototype_mixer.py
│     │  └─ multi_marker_restorer.py
│     ├─ losses/
│     │  ├─ charbonnier.py
│     │  ├─ ssim.py
│     │  ├─ gradient.py
│     │  ├─ frequency.py
│     │  ├─ prototype.py
│     │  └─ composite.py
│     ├─ metrics/
│     │  ├─ image_metrics.py
│     │  └─ aggregation.py
│     ├─ engine/
│     │  ├─ trainer.py
│     │  ├─ validator.py
│     │  ├─ inferencer.py
│     │  ├─ checkpoint.py
│     │  ├─ ema.py
│     │  └─ ensemble.py
│     ├─ submission/
│     │  ├─ writer.py
│     │  └─ validator.py
│     └─ utils/
│        ├─ seed.py
│        ├─ device.py
│        ├─ logging.py
│        ├─ image_io.py
│        └─ paths.py
├─ scripts/
│  ├─ run_all.ps1
│  ├─ run_all.bat
│  └─ run_all.sh
├─ tests/
│  ├─ test_discovery.py
│  ├─ test_manifest.py
│  ├─ test_dataset.py
│  ├─ test_transforms.py
│  ├─ test_model_shapes.py
│  ├─ test_losses.py
│  ├─ test_metrics.py
│  ├─ test_smoke_train.py
│  ├─ test_inference.py
│  └─ test_submission.py
├─ artifacts/
├─ outputs/
├─ results/
└─ docs/
   ├─ DATA_AUDIT.md
   ├─ MODEL_DESIGN.md
   ├─ EXPERIMENT_GUIDE.md
   ├─ SUBMISSION_GUIDE.md
   ├─ MODEL_CARD.md
   └─ TECHNICAL_REPORT_OUTLINE.md
```

使用 `src` 布局并确保可通过：

```bash
python -m virtual_staining.cli --help
```

如果未执行 editable install，也应在 README 中给出：

```bash
pip install -e .
```

## 4. 环境与依赖策略

### 4.1 Python

目标 Python 版本：3.10 或 3.11。

### 4.2 PyTorch

不要在普通依赖文件中强制安装某个 CUDA 版 PyTorch，以免破坏用户现有环境。

实现环境检测脚本，报告：

- Python 版本；
- 操作系统；
- torch 版本；
- CUDA 是否可用；
- CUDA runtime；
- GPU 名称；
- 总显存和空闲显存；
- cudnn；
- 是否支持 AMP/bfloat16；
- CPU 核心数；
- 内存。

若 PyTorch 未安装：
- README 分别给出 CPU、常见 CUDA 环境的官方安装入口说明；
- 不要猜测用户 CUDA 版本并自动安装错误 wheel。

### 4.3 其余依赖

尽量少而稳定，建议包括：

- numpy
- pandas
- Pillow
- opencv-python-headless
- scikit-image
- scikit-learn
- PyYAML
- tqdm
- matplotlib
- tensorboard
- pytest
- ruff
- pytorch-msssim，可选；若不使用则自行实现可微 SSIM

不要依赖必须联网下载模型的包。主模型应完全在仓库内实现。

## 5. 图像读取、通道与数值范围

实现统一 `ImageSpec`：

- width
- height
- channels
- mode
- dtype
- value range
- save format
- JPEG quality
- subsampling

读取时：

- 默认保留原始通道；
- DAPI 若 RGB 三通道几乎相同，可安全压缩为 1 通道，并在审计报告中证明；
- 若三通道差异明显，保留 RGB；
- 每个目标独立推断输出通道数；
- 模型内部统一 float32 `[0,1]`；
- 模型最后使用 sigmoid 或稳定的 clamp，将输出限制到 `[0,1]`；
- 保存前正确四舍五入为 uint8；
- JPEG 使用 `quality=100, subsampling=0, optimize=False`，避免不必要压缩损失；
- 不进行二次 JPEG 编解码。

必须写测试验证：

- 同一图像读取、保存、再读取误差；
- 灰度和 RGB；
- 中文路径；
- 值域不会变为 `[0,255]` 后再次乘 255；
- 通道顺序不会发生 RGB/BGR 混乱。

## 6. 数据增强

训练增强必须对 DAPI 和目标同步应用几何变换：

- 水平翻转；
- 垂直翻转；
- 90°、180°、270°旋转；
- 可选轻微平移，仅在审计显示存在小偏移时启用。

只对输入 DAPI 应用的轻量强度增强：

- gamma 0.9–1.1；
- brightness/contrast 小范围；
- 极轻高斯噪声；
- 可选轻微 Gaussian blur。

默认关闭：

- 任意角旋转；
- 大尺度形变；
- 随机 resize；
- 会破坏空间对应的 crop；
- CutMix/MixUp；
- 大幅颜色抖动；
- test-time 随机增强。

使用 full 256×256 patch 训练。除非显存确实不足，不要裁成 224 后再缩放。

## 7. 模型路线

必须实现两条路线。

### 7.1 可验证基线：Residual U-Net

实现一个纯 PyTorch 的稳定 Residual U-Net：

- 输入通道自动配置；
- 输出通道自动配置；
- 4 层 encoder-decoder；
- GroupNorm，避免小 batch 下 BatchNorm 不稳定；
- SiLU/GELU；
- residual blocks；
- skip connections；
- bilinear upsample + conv，避免反卷积棋盘格；
- 参数量和 FLOPs 统计；
- 支持单任务。

该模型用于：

- 验证数据管线；
- 得到最小可提交分数；
- 与主模型做消融；
- 作为集成成员。

### 7.2 主模型：MultiMarkerRestorer

实现一个从零可训练、确定性的多标记图像恢复网络，原则如下。

#### 7.2.1 总体结构

- 可选 Sobel 边缘输入：
  - 原 DAPI；
  - 固定 Sobel x/y 得到的梯度幅值；
  - 通过配置决定是否拼接。
- 共享多尺度 encoder：
  - 使用简化 NAFBlock/ConvNeXt 风格块；
  - 4 个尺度；
  - 默认通道 `[48, 96, 192, 384]`；
  - 每层深度可配置；
  - GroupNorm 或 LayerNorm2d；
  - depthwise convolution；
  - simple gate；
  - channel attention；
  - residual scaling。
- bottleneck 加入多任务原型模块；
- 共享 decoder 主干；
- 每个标记在每个 decoder 层有轻量 task adapter；
- 每个标记有独立输出 head；
- 支持一次输出全部目标，也支持传入 `task_name` 只输出一个目标；
- 支持 1 通道和 3 通道目标；
- 支持深监督，输出 64、128、256 三个尺度；
- 不使用随机 latent，不使用扩散采样，不使用判别器。

#### 7.2.2 多任务原型模块

借鉴“共享原型 + 任务特异原型”的思想，但自行实现干净、可测试的版本：

- 共享原型矩阵 `P_shared: [K_shared, C]`；
- 每个任务独立原型 `P_task[t]: [K_task, C]`；
- bottleneck 特征做 L2 normalize；
- 与原型计算 cosine similarity；
- softmax 得到 attention；
- 分别聚合共享和任务原型；
- 经线性投影后与原特征残差融合；
- 原型数量、温度和融合权重可配置；
- task embedding 通过 FiLM 调节 decoder 特征。

实现可选原型损失：

1. activation/commitment loss：鼓励有效特征接近至少一个原型；
2. diversity loss：降低原型间余弦相似；
3. 不要让原型损失权重大到压制重建目标；
4. 默认权重很小，并允许完全关闭。

输出原型注意力图，用于可解释性，不参与测试后处理。

#### 7.2.3 任务适配器

每个标记每个 decoder stage 使用轻量 adapter：

- depthwise 3×3；
- pointwise 1×1；
- GELU；
- residual；
- 由 task embedding 产生 scale/shift。

不要复制四套完整 encoder。允许配置：

- `shared_decoder_with_adapters`
- `separate_heads`
- 可选 `separate_decoders` 仅用于实验，不作为默认。

## 8. 损失函数

主方案默认不使用 GAN loss。

每个目标的基础损失：

```text
L_task =
  w_charb * Charbonnier
+ w_ssim * (1 - SSIM)
+ w_msssim * (1 - MS-SSIM)
+ w_grad * GradientLoss
+ w_freq * FrequencyAmplitudeLoss
```

建议默认初值：

- Charbonnier：0.40
- SSIM：0.35
- MS-SSIM：0.10
- Gradient：0.10
- Frequency：0.05

所有权重写入 YAML，可调。

### 8.1 Charbonnier

- 比 L2 更稳健；
- `epsilon` 可配置；
- 支持结构权重图。

### 8.2 SSIM/MS-SSIM

- 训练损失在 `[0,1]` 计算；
- 灰度和 RGB 均支持；
- window size 对 256 图像合理；
- 数值稳定；
- 对尺寸过小的深监督输出自动调整尺度数。

### 8.3 结构权重

为避免稀疏标记模型输出过度平滑或全背景，实现与亮暗极性无关的结构权重：

- 从目标局部梯度、Laplacian 或局部方差构造 soft weight；
- `weight = 1 + alpha * normalized_structure`;
- 不使用固定“亮像素就是阳性”的假设；
- alpha 可配置且默认温和。

### 8.4 梯度损失

- Sobel x/y；
- 比较预测与 GT 梯度；
- 支持 Charbonnier/L1。

### 8.5 频域损失

- 比较 log amplitude spectrum；
- 只给予低权重；
- 避免相位损失导致训练不稳定。

### 8.6 多尺度监督

256、128、64 三尺度：

- 1.0
- 0.5
- 0.25

下采样 GT 时使用 area 或 antialias bilinear，保持一致。

### 8.7 跨标记相关约束

联合训练时实现可选低权重 `L_corr`：

- 在 batch 内计算各标记输出的空间统计或多尺度 pooled feature；
- 比较预测和 GT 的任务间相关矩阵；
- 默认权重 0.01–0.03；
- 必须可关闭；
- 若导致某单任务指标下降，最终提交配置关闭。

总损失：

```text
L_total =
mean(weighted_task_losses)
+ lambda_corr * L_corr
+ lambda_proto_act * L_proto_act
+ lambda_proto_div * L_proto_div
```

任务权重默认等权。实现基于训练损失 EMA 的可选自动归一化，但默认关闭，避免不稳定。

## 9. 训练策略

### 9.1 优化器和调度

默认：

- AdamW；
- lr `2e-4`；
- weight decay `1e-4`；
- warmup 5 epochs；
- cosine decay；
- gradient clipping 1.0；
- AMP；
- EMA decay 0.999；
- early stopping patience 30；
- 保存 top 3 和 last。

允许根据 batch size 线性调整 lr，但实际使用值必须写入日志。

### 9.2 显存自适应

实现 `hardware_profile: auto`：

- 无 GPU：smoke 可 CPU 跑，完整训练给出命令但不要假装已完成；
- ≤8GB：base channels 32，batch 2–4，gradient accumulation；
- 10–16GB：base channels 48，batch 4–8；
- ≥20GB：base channels 64，batch 8–16；
- OOM 时自动：
  1. 清缓存；
  2. batch 减半；
  3. accumulation 翻倍；
  4. 只允许最多重试 3 次；
  5. 把调整写入日志。
- 不要捕获所有异常并误判为 OOM，只处理明确 CUDA OOM。

### 9.3 可复现性

统一设置：

- Python random；
- NumPy；
- PyTorch CPU/GPU；
- cuDNN benchmark 与 deterministic 可配置；
- DataLoader generator；
- worker init；
- seed 写入 checkpoint。

checkpoint 必须包含：

- model；
- EMA；
- optimizer；
- scheduler；
- scaler；
- epoch；
- global step；
- config；
- manifest hash；
- image spec；
- target list；
- metric history；
- git commit，若仓库是 git。

恢复训练必须可以接着跑，并测试一次 resume。

### 9.4 验证与模型选择

每个目标分别计算：

- mean SSIM；
- median SSIM；
- mean PSNR；
- median PSNR；
- 按 ROI 聚合后的 SSIM/PSNR；
- 最差 10% 样本；
- 全部样本 CSV。

同时计算宏平均。

模型选择规则：

1. 默认以 `SSIM` 为第一关键指标；
2. PSNR 为第二关键指标；
3. 可选本地代理分：
   `0.7 * SSIM + 0.3 * normalized_PSNR`
4. `normalized_PSNR` 上下界来自配置；
5. 明确标注代理分不等于官方分。

保存：

- `best_ssim.ckpt`
- `best_psnr.ckpt`
- `best_proxy.ckpt`
- `last.ckpt`

### 9.5 训练配置

必须提供：

1. `smoke.yaml`
   - 极少样本；
   - 1–2 epoch；
   - CPU 可跑；
   - 用于完整链路测试。

2. `baseline_unet.yaml`
   - 单任务；
   - 50–100 epoch；
   - 较小模型。

3. `competition_single.yaml`
   - 指定一个 target；
   - 主模型；
   - 150–250 epoch；
   - EMA；
   - D4 TTA；
   - top-k ensemble。

4. `competition_multitask.yaml`
   - 四目标；
   - 原型模块；
   - task adapters；
   - 150–250 epoch；
   - 可导出一个或多个目标。

## 10. 指标实现

最终验证指标使用 `scikit-image` 作为参考实现：

- `structural_similarity`
- `peak_signal_noise_ratio`

要求：

- 明确 `data_range`；
- 灰度：二维；
- RGB：`channel_axis=-1`；
- 不在评估前偷偷 blur、resize 或颜色转换；
- 逐图计算后再平均；
- 不把所有图拼起来一次计算；
- PSNR 为无穷时合理处理并在报告中说明；
- 写单元测试：
  - identical images → SSIM 约 1；
  - identical images → PSNR inf 或按设定 cap；
  - 加噪声后二者下降；
  - 灰度/RGB结果合理。

训练中的可微 SSIM 与最终参考 SSIM需要做一致性对照测试。

## 11. 推理、TTA、集成

### 11.1 推理

CLI 示例：

```bash
python -m virtual_staining.cli predict \
  --config configs/infer.yaml \
  --checkpoint outputs/.../best_ssim.ckpt \
  --data-root AUTO \
  --target CD68
```

推理必须：

- 自动读取 checkpoint 中 image spec；
- 保持排序稳定；
- 支持 batch；
- 不 shuffle；
- 不需要训练标签；
- 不修改测试图；
- 可 CPU/GPU；
- 显存不足自动减小 batch；
- 输出耗时和峰值显存；
- 逐图保存原尺寸。

### 11.2 D4 TTA

实现：

- identity；
- hflip；
- vflip；
- hvflip；
- rot90；
- rot180；
- rot270；
- transpose/对角反射，确保严格可逆。

对预测执行逆变换后平均。

TTA 必须可关闭，并在验证集上比较是否真实提升。若某目标下降，不在该目标提交配置中使用。

### 11.3 检查点集成

支持：

- 同模型 top-k checkpoint；
- 多 seed；
- 多 fold；
- baseline + main model；
- 算术平均；
- 可选根据验证 SSIM 求非负权重。

权重拟合只能使用验证集，禁止使用测试结果。

不要默认做复杂像素后处理。允许的最终操作仅限：

- clamp `[0,1]`；
- uint8 转换；
- 按验证确定的模型平均。

## 12. 提交生成与严格校验

实现：

```bash
python -m virtual_staining.cli make-submission \
  --config configs/infer.yaml \
  --pred-dir outputs/.../predictions \
  --test-manifest artifacts/manifests/test_manifest.csv \
  --target CD68 \
  --output-dir results
```

目标目录：

```text
results/
└─ test/
   └─ CD68/
      ├─ ROI025_00_00_fake.jpg
      └─ ...
```

实现 `validate-submission`，校验：

- 目录层级；
- target 名称；
- 文件数与测试输入完全一致；
- 缺失文件；
- 多余文件；
- 重复 stem；
- 命名后缀 `_fake`；
- 扩展名；
- 256×256；
- 通道/mode；
- 是否能解码；
- dtype；
- 像素范围；
- 全黑/全白异常比例；
- NaN/Inf；
- 与输入 stem 一一对应。

生成：

- `artifacts/submission_report.json`
- `artifacts/submission_files.csv`
- `submission_<target>.zip`

ZIP 内根目录必须是 `results/`，不能多包一层项目目录。

不要同时提交四个目标，除非配置明确指定 `submit_targets` 为四项。联合训练可以只导出官方当前要求的目标。

## 13. CLI

使用 argparse 或 Typer。若使用 Typer，确保版本兼容。必须提供以下命令：

```text
env
discover-data
audit-data
build-manifest
train
resume
validate
predict
ensemble
make-submission
validate-submission
run-pipeline
```

示例：

```bash
python -m virtual_staining.cli env
python -m virtual_staining.cli discover-data
python -m virtual_staining.cli audit-data --data-root AUTO
python -m virtual_staining.cli train --config configs/smoke.yaml
python -m virtual_staining.cli run-pipeline --config configs/smoke.yaml
```

`run-pipeline` 依次完成：

1. 环境检查；
2. 数据发现；
3. manifest；
4. 数据审计；
5. smoke train；
6. validate；
7. smoke predict；
8. submission build；
9. submission validation。

任何一步失败应返回非零退出码。

## 14. 测试和质量门槛

实现并实际运行：

```bash
python -m compileall src tests
pytest -q
```

至少覆盖：

1. 数据目录自动发现；
2. 文件名规范化；
3. ROI 分组划分；
4. 防泄漏；
5. 灰度/RGB读取；
6. paired transforms；
7. dataset 单任务/多任务；
8. 模型单任务 shape；
9. 模型四任务 shape；
10. forward + backward；
11. loss finite；
12. CPU AMP 路径不报错；
13. CUDA AMP，若有 GPU；
14. EMA 更新；
15. checkpoint 保存/恢复；
16. SSIM/PSNR；
17. D4 TTA 可逆；
18. 推理文件命名；
19. JPEG 保存；
20. submission zip 根目录；
21. 中文路径；
22. Windows `num_workers=0`；
23. 缺文件和破损图像的明确错误；
24. smoke pipeline。

代码质量：

- type hints；
- docstrings；
- 关键逻辑注释；
- ruff 检查；
- 不要过度抽象；
- 不要引入循环 import；
- 不要全局修改 `sys.path`；
- 不要在 import 时启动训练；
- 所有入口有 `if __name__ == "__main__"`；
- 训练日志不依赖 notebook。

## 15. 自动修复闭环

Implementation 阶段严格执行以下闭环：

1. 创建最小工程；
2. `compileall`；
3. 单元测试；
4. 数据发现；
5. 审计；
6. smoke train；
7. smoke validate；
8. smoke predict；
9. 生成 smoke submission；
10. 校验 submission；
11. 修复错误；
12. 从失败步骤向后全部重跑；
13. 全部通过后再优化文档和默认配置。

常见错误必须主动排查：

- import 路径错误；
- Python 包未安装；
- Windows DataLoader 卡死；
- DAPI 和目标 stem 不一致；
- RGB/BGR；
- 张量 HWC/CHW；
- uint8/float 值域；
- SSIM channel_axis；
- MS-SSIM 尺寸限制；
- 输出通道与目标不一致；
- checkpoint key 中 `module.` 前缀；
- EMA 权重未加载；
- AMP scaler 在 CPU；
- CUDA OOM；
- 非有限 loss；
- JPEG 编码损失；
- zip 多一层目录；
- `ROI..._fake_fake.jpg`；
- test 数量与结果数量不一致。

不得使用“已实现但未测试”作为最终结论。

## 16. 实验体系

生成 `docs/EXPERIMENT_GUIDE.md`，给出推荐顺序：

### E0 数据审计
- 通道、配对、ROI 泄漏、目标分布。

### E1 Baseline
- Residual U-Net；
- Charbonnier；
- 无 TTA。

### E2 指标对齐损失
- +SSIM；
- +MS-SSIM；
- +gradient；
- 做逐项消融。

### E3 主模型
- NAF 风格共享 encoder-decoder；
- task adapters；
- 单任务训练。

### E4 多任务
- 四目标联合；
- 不启用 prototype；
- 与单任务比较。

### E5 Proto
- shared/task-specific prototypes；
- activation/diversity；
- 输出可解释图。

### E6 TTA
- 无 TTA vs D4。

### E7 Ensemble
- top3；
- 3 seeds；
- baseline + main。

生成统一实验表 CSV，字段包括：

- run_id
- git_commit
- config_hash
- seed
- fold
- target
- model
- params
- flops
- train_time
- peak_vram
- val_ssim
- val_psnr
- roi_ssim
- roi_psnr
- tta
- ensemble
- checkpoint

## 17. 推荐的冲分决策

工程默认采用以下优先级：

1. 数据配对和 ROI 防泄漏；
2. 正确的值域、通道、保存格式；
3. Residual U-Net 稳定基线；
4. Charbonnier + SSIM/MS-SSIM；
5. NAF 风格主模型；
6. EMA；
7. D4 TTA；
8. 多 seed/top-k ensemble；
9. 多任务共享；
10. 原型解释性；
11. 最后才考虑扩散/GAN。

不要为了“创新”牺牲基础分数。

扩散模型只作为后续研究分支写入文档，不要默认实现为主训练入口。若未来增加，必须是可选插件，且不能破坏现有 deterministic pipeline。

## 18. 配置文件关键字段

`configs/default.yaml` 至少包含：

```yaml
project:
  name: aic_virtual_staining
  output_root: outputs
  artifact_root: artifacts
  seed: 2026

data:
  root: AUTO
  organ: auto
  train_split: train
  val_split: val
  test_split: test
  targets: [HLA-DR, CD45RO, Vimentin, CD68]
  submit_targets: [CD68]
  group_key: roi_id
  val_ratio: 0.2
  image_size: 256
  input_mode: auto
  target_modes: auto
  num_workers: 0
  pin_memory: true
  persistent_workers: false

model:
  name: multi_marker_restorer
  base_channels: 48
  encoder_depths: [2, 2, 4, 6]
  decoder_depths: [2, 2, 2]
  use_sobel_input: true
  use_task_adapters: true
  use_prototypes: true
  shared_prototypes: 8
  task_prototypes: 8
  prototype_temperature: 0.1
  deep_supervision: true

loss:
  charbonnier: 0.40
  ssim: 0.35
  ms_ssim: 0.10
  gradient: 0.10
  frequency: 0.05
  structure_weight_alpha: 1.0
  correlation: 0.02
  prototype_activation: 0.001
  prototype_diversity: 0.001

train:
  epochs: 200
  batch_size: auto
  gradient_accumulation: auto
  optimizer: adamw
  lr: 0.0002
  weight_decay: 0.0001
  warmup_epochs: 5
  scheduler: cosine
  amp: true
  amp_dtype: auto
  grad_clip: 1.0
  ema: true
  ema_decay: 0.999
  early_stopping_patience: 30
  save_top_k: 3
  resume: null

validation:
  primary_metric: ssim
  psnr_norm_min: 15.0
  psnr_norm_max: 40.0
  save_visuals: true
  worst_k: 32

inference:
  batch_size: auto
  use_ema: true
  tta: d4
  ensemble_checkpoints: []
  jpeg_quality: 100
  jpeg_subsampling: 0

submission:
  root_name: results
  split_name: test
  fake_suffix: _fake
  extension: .jpg
  create_zip: true
```

若数据审计发现目标只有单通道或目录结构不同，自动生成一个解析后的本地配置：
`configs/resolved_local.yaml`
但不要覆盖原默认配置。

## 19. 文档

README 必须面向初学者，包含：

1. 赛题任务；
2. 项目特性；
3. 文件结构；
4. 环境安装；
5. 放置数据；
6. 自动发现数据；
7. 数据审计；
8. smoke 测试；
9. 单任务训练；
10. 多任务训练；
11. 断点续训；
12. 验证；
13. 推理；
14. TTA；
15. ensemble；
16. 生成提交；
17. 校验 ZIP；
18. 常见报错；
19. Windows/PyCharm 操作；
20. GPU 显存配置；
21. 复现实验；
22. 不同阶段如何替换 organ；
23. 如何选择提交 target；
24. 为什么本地代理分不等于官方分。

额外生成：

- `docs/MODEL_DESIGN.md`：完整架构和损失公式；
- `docs/MODEL_CARD.md`：数据范围、局限、风险；
- `docs/TECHNICAL_REPORT_OUTLINE.md`：不少于 2000 字技术报告的详细提纲；
- `docs/SUBMISSION_GUIDE.md`：目录与 ZIP 截图式文字说明；
- `docs/ABLATION_PLAN.md`：单任务、多任务、prototype、TTA、ensemble 消融；
- `docs/FAILURE_ANALYSIS.md`：全背景、过平滑、伪影、错位、颜色偏差的分析方法。

## 20. AGENTS.md

首先创建 `AGENTS.md`，写入本项目长期规则：

- 先运行相关测试再声称完成；
- 不允许 TODO/pass；
- 路径必须跨平台；
- 不下载外部训练数据；
- 测试集不参与训练统计；
- 修改数据管线后必须运行 manifest、dataset、smoke 测试；
- 修改模型后必须运行 shape、backward、smoke train；
- 修改推理后必须运行 inference、submission 测试；
- 最终回复列出实际运行过的命令和结果；
- 对长期训练不要捏造完成状态；
- 若没有 GPU，明确说明只完成 CPU smoke。

## 21. 最终验收标准

只有以下全部满足，才可以声称“项目构建完成”：

- [ ] `python -m compileall src tests` 成功；
- [ ] `pytest -q` 全部通过；
- [ ] 数据根目录成功发现或给出唯一明确缺失信息；
- [ ] manifest 生成成功；
- [ ] 无 train/val ROI 泄漏；
- [ ] 数据审计报告生成；
- [ ] smoke 模型 forward/backward 成功；
- [ ] smoke 训练至少完成 1 个 epoch；
- [ ] checkpoint 可保存并恢复；
- [ ] 验证可输出 SSIM/PSNR；
- [ ] smoke 推理生成图像；
- [ ] 输出尺寸、通道、值域正确；
- [ ] submission 目录构建成功；
- [ ] submission validator 通过；
- [ ] ZIP 根目录正确；
- [ ] README 命令可以直接复制；
- [ ] Windows 脚本存在；
- [ ] 无 TODO/pass；
- [ ] 最终报告不夸大未执行的完整训练。

## 22. Plan 阶段的输出格式

现在先执行只读检查，并按以下格式输出计划：

1. **发现的工作区结构**
2. **发现的数据候选目录**
3. **数据结构初步判断**
4. **环境与硬件**
5. **需要自动推断的事项**
6. **唯一可能需要用户确认的事项**
7. **将创建/修改的文件**
8. **分阶段实施步骤**
9. **测试与验收门槛**
10. **完整训练预计资源**
11. **主要风险与规避**
12. **准备执行的命令清单**

如果数据已在工作区内且能够推断，不要先问用户数据路径。

当用户批准后，直接实施并持续自测修复，不要再次复述整个计划。
