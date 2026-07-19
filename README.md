# AIC 虚拟染色竞赛工程

这是一个完整、可复现的 PyTorch 虚拟染色工程。模型读取 DAPI 图像 patch，生成与输入同尺寸、同位置的 HLA-DR、CD45RO、Vimentin 或 CD68 图像。

当前仓库已经清理为“代码交付状态”：

- 不包含开发阶段的样例数据；
- 不包含 smoke、消融实验、失败实验或样例训练产生的 checkpoint；
- 不包含预测图、TensorBoard 日志、审计图片、ZIP 和提交结果；
- dataset/、outputs/ 和 artifacts/ 会在放入真实数据或运行程序后重新产生内容；
- 真实数据、训练权重和生成结果默认被 .gitignore 排除。

你不需要修改 Python 源码才能换数据。把官方数据放进 dataset/official/，完成审计后运行训练命令即可。

---

## 1. 项目解决什么问题

这是一个有监督的图像到图像恢复任务：

~~~text
输入 DAPI
   ↓
虚拟染色模型
   ↓
输出 HLA-DR / CD45RO / Vimentin / CD68
~~~

同名 DAPI 和目标图像必须描述同一个组织位置。模型学习细胞核、纹理、边缘和空间结构与目标标记之间的映射。

工程内部将图像转换为 float32、范围 [0, 1]、CHW 排列的 Tensor；生成提交时再恢复为正确的 uint8 JPEG，避免 RGB/BGR、错误取整和重复压缩带来的指标损失。

主要指标：

- **SSIM**：关注亮度、对比度和局部结构；
- **PSNR**：由像素均方误差推导，越高通常表示像素误差越小；
- **JPG round-trip 指标**：把预测真正保存为竞赛 JPEG，再解码计算 SSIM/PSNR，比内存中的 float 指标更接近最终提交。

---

## 2. 之前为什么有 6～7 GB

清理前项目总计约 6.68 GiB，其中 outputs/ 占 6.64 GiB。直接原因不是代码，而是 142 个 checkpoint。

| 内容 | 清理前占用 |
|---|---:|
| 142 个 .ckpt | 约 6.62 GiB |
| 其他训练日志、预测图和 ZIP | 约 16 MiB |
| 样例数据 | 约 34 MiB |
| 样例审计 artifacts | 约 5 MiB |
| 源码、测试、配置、脚本和文档 | 约 1.5 MiB |

每次实验可能同时保存 last、best_ssim、best_psnr、best_proxy 和多个 top-k 权重；再叠加多轮 smoke、A0/A1/A2 消融和失败重试，就产生了大量相近的 checkpoint。

本次已经删除：

- outputs/ 中全部样例 checkpoint 和运行产物；
- artifacts/ 中全部样例 manifest、审计图和实验快照；
- dataset/ 中旧匿名样例数据；
- pytest、ruff、Python 字节码和 editable-install 缓存。

真实训练仍会生成 checkpoint，这是正常且必要的。第 13 节说明应该保留什么，以及如何避免再次膨胀。

---

## 3. 整体框架

~~~text
官方 train / val / test
          │
          ▼
自动发现数据根、split、organ 和 marker
          │
          ▼
同名配对、哈希查重、ROI 分组、manifest
          │
          ▼
通道/值域/对齐/边界/泄漏审计
          │
          ▼
Dataset + 同步增强 + DataLoader
          │
          ▼
U-Net / MultiMarkerRestorer / CAMP-VS v2
          │
          ▼
组合损失 + AdamW + AMP + EMA + checkpoint
          │
          ▼
float / uint8 / JPG 验证
          │
          ▼
确定性推理、可选 TTA/ensemble
          │
          ▼
results/test/<TARGET>/<stem>_fake.jpg + ZIP
~~~

### 3.1 目录说明

| 路径 | 用途 |
|---|---|
| src/virtual_staining/ | 核心 Python 源码 |
| src/virtual_staining/data/ | 发现、审计、manifest、Dataset、ROI 邻域和增强 |
| src/virtual_staining/models/ | U-Net、MultiMarkerRestorer、CAMP-VS v2 |
| src/virtual_staining/losses/ | 像素、结构、梯度、频率、金字塔等损失 |
| src/virtual_staining/metrics/ | SSIM、PSNR、三域指标和 ROI 聚合 |
| src/virtual_staining/engine/ | 训练、验证、推理、EMA、checkpoint、soup、ensemble |
| src/virtual_staining/submission/ | 提交目录、命名、图像和 ZIP 校验 |
| configs/ | 默认、单目标、多目标、推理和 smoke 配置 |
| configs/performance_v2/ | V2 特性开关、消融和保守配置 |
| tests/ | 单元测试与工程链路测试 |
| scripts/ | 环境安装和验收脚本 |
| docs/ | 模型、实验、提交和 V2 技术文档 |
| dataset/ | 放赛事真实数据；当前为空 |
| artifacts/ | 运行后生成 manifest、数据审计和指标 |
| outputs/ | 运行后生成 checkpoint、日志和预测 |
| results/ | 正式提交结果 |

### 3.2 数据层

数据层会：

- 自动识别 split、organ 和 marker 的排列；
- 规范化 HLA_DR、hla-dr、CD45RO 等名称；
- 按规范化文件 stem 配对 DAPI 与目标；
- 检查缺失、重复、损坏、尺寸、通道和哈希；
- 有官方 train/val 时保留官方划分；
- 没有官方 val 时按 ROI/group 分组切分；
- 生成 train、val、test manifest；
- 防止 test 参与训练、统计、自监督或模型选择；
- 对 ROI_row_col 建立可验证邻域，但不会把纯数字 stem 猜成权威坐标。

### 3.3 模型层

**Residual U-Net**

- 适合学习和快速基线；
- 残差块、GroupNorm、SiLU、四尺度编码解码；
- 参数较少，适合先验证数据管线。

**MultiMarkerRestorer**

- 当前保守正式配置使用的主模型；
- NAF 风格局部编码器；
- 可选 Sobel 输入；
- marker embedding、task adapter 和 FiLM；
- 共享/任务原型；
- 多尺度深监督；
- 支持单目标和四目标联合训练。

**CAMP-VS v2**

- 是增量实验框架，不替换稳定基线；
- 包含 3×3 DAPI context、Restormer-lite、分层原型、organ conditioning、base/detail 和校准器；
- 每项功能都有 feature flag 和回滚路径；
- context 只能读取 DAPI，不能读取相邻标签；
- 只有真实 ROI 网格、无邻域泄漏且 JPG 指标稳定提高后，相关模块才允许进入最终配置。

第一次用真实数据不要打开全部 V2 开关。推荐先使用：

~~~text
configs/performance_v2/retained_unpromoted.yaml
~~~

它保留三域验证和严格数据规则，但关闭尚未通过真实 ROI 消融的上下文、base/detail、预训练和高级集成。

### 3.4 损失与训练

可用损失包括 MSE、Charbonnier、SSIM/MS-SSIM、Sobel gradient、frequency、pyramid、statistics、prototype regularization 和 deep supervision。

训练器支持：

- AdamW；
- warmup + cosine 学习率；
- BF16/FP16 AMP；
- 梯度累积和裁剪；
- EMA；
- early stopping；
- last、best 和 top-k checkpoint；
- optimizer、scheduler、scaler、EMA 与随机状态的完整恢复；
- 只针对明确 CUDA OOM 的有限重试；
- raw 与 EMA 分开验证，不假定 EMA 一定更好。

### 3.5 推理与提交

- 按 manifest 稳定排序；
- 保持原图尺寸、mode 和 stem；
- 可选 D4 测试时增强；
- 支持结构兼容的 checkpoint 平均；
- learned ensemble 和 model soup 只能使用 validation/OOF；
- official test 为空时正式预测会失败；
- 文件命名为 <stem>_fake.jpg；
- ZIP 第一层必须是 results/；
- validator 检查数量、命名、重复、缺失、尺寸、mode、解码和 ZIP 成员。

---

## 4. 新手需要学习什么

建议“先会运行，再理解原理，最后做实验”。

### 4.1 Python

至少会变量、字符串、列表、字典、条件、循环、函数、类、import、异常、文件读写、pathlib.Path，以及在 PowerShell 运行命令。

### 4.2 NumPy 与 Tensor

要理解 shape、dtype、索引、切片、广播、HWC/CHW、NCHW、uint8 0～255、float 0～1、RGB/灰度、CPU/CUDA Tensor。

### 4.3 深度学习

需要理解：

- forward、prediction、loss、backward；
- optimizer 和 learning rate；
- epoch、batch、iteration；
- train、validation、test；
- 过拟合与欠拟合；
- 卷积、残差、归一化、激活函数；
- encoder、decoder、skip connection；
- AMP、梯度累积、EMA、checkpoint。

### 4.4 图像与竞赛

重点理解：

- JPEG 是有损压缩；
- 卷积、Sobel、金字塔和频域；
- SSIM 与 PSNR；
- 配准图像必须同步旋转和翻转；
- ROI/group split 为什么能减少泄漏；
- test 为什么不能用于调参；
- 本地指标提升不等于 leaderboard 一定提升。

### 4.5 工程工具

建议了解 Conda、pip、Git、pytest、YAML、错误堆栈和 nvidia-smi。

---

## 5. B 站学习路线

链接于 **2026-07-17** 核验。安装界面可能随版本变化，环境安装以本文第 6 节为准。

1. [黑马程序员 Python 零基础全套教程](https://www.bilibili.com/video/BV1qW4y1a7fU/)  
   UP 主：黑马程序员。学习语法、函数、容器、文件、异常、模块和类；数据库、爬虫和大数据可跳过。

2. [莫烦 Python：NumPy & Pandas](https://www.bilibili.com/video/BV1Ex411L7oT/)  
   UP 主：莫烦Python。优先看 NumPy，理解数组、shape、索引、拼接、分割和 copy。

3. [PyTorch 深度学习快速入门教程（小土堆）](https://www.bilibili.com/video/BV1hE411t7RN/)  
   UP 主：我是土堆。重点看 Tensor、Dataset/DataLoader、nn.Module、损失、优化器、GPU、训练循环和模型保存。

4. [动手学深度学习 v2：官方系列入口](https://www.bilibili.com/video/BV1oX4y137bC/)  
   UP 主：跟李沐学AI。推荐数据操作、自动求导、损失与优化、卷积、BatchNorm、ResNet、注意力；RNN/NLP 可暂时跳过。

5. [10 小时学会图像处理 OpenCV 入门](https://www.bilibili.com/video/BV1Fo4y1d7JL/)  
   UP 主：黑马程序员。用于理解像素、读写、滤波、边缘和图像操作。

6. [OpenMMLab 公开课：计算机视觉与 OpenMMLab 概述](https://www.bilibili.com/video/BV1R341117FJ/)  
   UP 主：OpenMMLab。用于建立分类、检测、分割和图像编辑的整体概念。

推荐顺序：

~~~text
Python → NumPy → 小土堆 PyTorch → 李沐卷积/优化/ResNet
       → OpenCV/计算机视觉 → 边运行本项目边补知识
~~~

学完 Python、NumPy 和 PyTorch Dataset/训练循环后，就可以做第 9 节的一轮 sanity train，不必等全部课程看完。

---

## 6. 创建 MEDICAL 环境

### 6.1 一键脚本

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_medical.ps1
~~~

### 6.2 手动安装

当前 RTX 4060 环境可以使用：

~~~powershell
conda create -n MEDICAL python=3.11 pip -y
conda run -n MEDICAL python -m pip install --upgrade pip setuptools wheel
conda run -n MEDICAL python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
conda run -n MEDICAL python -m pip install -e '.[dev]'
conda run -n MEDICAL python -m pip check
~~~

换电脑后应到 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择匹配命令，不要机械照抄 CUDA wheel。本项目不编译自定义 CUDA 扩展，因此没有系统 nvcc 也能运行。

验证：

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli env
~~~

确认 CUDA 可用、GPU 名称正确、torch.version.cuda 不是 None，并查看 BF16/FP16 和显存。

---

## 7. 真实数据放在哪里

把官方压缩包解压到：

~~~text
项目根目录/dataset/official/
~~~

推荐结构：

~~~text
dataset/
└─ official/
   ├─ train/
   │  ├─ colon/
   │  │  ├─ DAPI/
   │  │  ├─ HLA-DR/
   │  │  ├─ CD45RO/
   │  │  ├─ Vimentin/
   │  │  └─ CD68/
   │  ├─ liver/    （同样五个 marker）
   │  └─ stomach/  （同样五个 marker）
   ├─ val/
   │  ├─ colon/    （同样五个 marker）
   │  ├─ liver/
   │  └─ stomach/
   └─ test/
      ├─ colon/DAPI/
      ├─ liver/DAPI/
      └─ stomach/DAPI/
~~~

### train/val 规则

- 同一个 patch 的 DAPI 和四个目标必须同名；
- 当前 manifest 构建器要求 DAPI、HLA-DR、CD45RO、Vimentin、CD68 配对齐全；
- 图像尺寸一致；
- 保留官方原始 JPEG，不要提前重编码；
- 有官方 train/val 时不要重新混合。

### test 规则

- 只放 DAPI；
- 不放伪造标签；
- 保持官方文件名；
- test 不参与统计、预训练、模型选择或调参。

推荐官方文件名形式：

~~~text
ROI000_00_00.jpg
ROI000_00_01.jpg
ROI000_01_00.jpg
~~~

这能解析出 ROI、row、col，用于可靠邻域和 ROI 分组验证。纯数字文件名不能作为权威坐标。

代码也支持 organ/split/marker、split/marker 和 organ/marker，但新手建议使用上面的 split/organ/marker。

---

## 8. 数据路径怎么设置

### 推荐：AIC_DATA_ROOT

每次打开新 PowerShell，在项目根运行：

~~~powershell
$env:AIC_DATA_ROOT = (Resolve-Path .\dataset\official).Path
~~~

之后 --data-root AUTO 会读取这个环境变量。

### 直接指定

也可以每条命令写：

~~~powershell
--data-root dataset/official
~~~

显式路径具有最高优先级，即使工作区还有其他数据目录也不会选错。

### resolved_local.yaml

第一次运行 audit-data 后，程序会把真实数据根、输入/目标通道和本机安全设置写入 configs/resolved_local.yaml。更换数据包后重新构建 manifest 和审计，不要沿用旧 artifacts。

不需要修改任何 Python 路径。

---

## 9. 从真实数据到训练

以下命令均在项目根执行。

### 9.1 环境与数据发现

~~~powershell
$env:AIC_DATA_ROOT = (Resolve-Path .\dataset\official).Path
conda run -n MEDICAL python -m virtual_staining.cli env
conda run -n MEDICAL python -m virtual_staining.cli discover-data --data-root AUTO
~~~

查看 artifacts/data_discovery.json，确认 selected_root 是 dataset/official，而不是某个 marker 子目录。

### 9.2 构建 manifest

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli build-manifest --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO
~~~

检查：

- artifacts/manifests/train_manifest.csv 非空；
- val_manifest.csv 非空；
- test_manifest.csv 行数等于官方 test DAPI 数；
- 没有意外 missing_targets；
- train/val 无重复 canonical key、重复哈希或 ROI 泄漏。

### 9.3 数据与 ROI 审计

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli audit-data --data-root dataset/official
conda run -n MEDICAL python -m virtual_staining.cli audit-roi-grid --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --output-dir artifacts/performance_v2
~~~

至少确认图像可解码、尺寸/通道正确、配对完整、train/val 无泄漏、ROI 坐标解析正确。如果要启用 context，方向、边界连续性和 context gate 必须通过。

### 9.4 一轮 sanity train

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli train --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --target CD68 --run-id cd68_sanity --max-epochs 1 --set train.save_top_k=1
~~~

它只验证真实 Dataset、forward/backward、CUDA/AMP、checkpoint 和 validation。它不是正式训练结果。成功后可删除 outputs/performance_v2/cd68_sanity/。

### 9.5 正式单目标训练

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli train --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --target CD68 --run-id cd68_official_seed2026 --set train.save_top_k=1
~~~

保守配置默认：

- MultiMarkerRestorer base32；
- 120 epoch；
- batch size 2、gradient accumulation 4；
- AMP、EMA；
- float、uint8、JPG 三域验证；
- raw/EMA 分别比较；
- context、D4 和未晋级 V2 模块关闭。

8 GiB 显卡从这里开始。用正式数据第一个 epoch 的耗时估算总训练时间。

### 9.6 断点续训

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli resume --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --checkpoint outputs/performance_v2/cd68_official_seed2026/checkpoints/last.ckpt
~~~

恢复会检查目标、ImageSpec 和 manifest hash，并恢复 optimizer、scheduler、scaler、EMA、epoch 和随机状态。

### 9.7 验证

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli validate --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --target CD68 --checkpoint outputs/performance_v2/cd68_official_seed2026/checkpoints/best_ssim.ckpt
~~~

优先看真实 ROI 分组的 JPG round-trip SSIM/PSNR，同时检查 float、uint8、raw 和 EMA，不能只选择最好看的单一数值。

---

## 10. 推理和竞赛提交

只有 official test 放入 dataset/official/test/ 后才运行。

### 10.1 推理

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli predict --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --checkpoint outputs/performance_v2/cd68_official_seed2026/checkpoints/best_ssim.ckpt --manifest artifacts/manifests/test_manifest.csv --target CD68 --output-dir outputs/performance_v2/cd68_official_seed2026/predictions
~~~

test manifest 为空时失败是正确行为，不要用 validation 替代。

### 10.2 生成结果和 ZIP

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli make-submission --config configs/performance_v2/retained_unpromoted.yaml --pred-dir outputs/performance_v2/cd68_official_seed2026/predictions --test-manifest artifacts/manifests/test_manifest.csv --target CD68 --output-dir submission_ready
~~~

预期：

~~~text
submission_ready/
├─ results/test/CD68/<stem>_fake.jpg
└─ submission_CD68.zip
~~~

### 10.3 校验目录和 ZIP

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli validate-submission --submission-dir submission_ready/results --test-manifest artifacts/manifests/test_manifest.csv --target CD68 --zip-path submission_ready/submission_CD68.zip --artifact-dir submission_ready/validation
~~~

只有 validator 成功才上传 ZIP。不要手工再压缩一次，否则容易多一层目录。

---

## 11. 更换目标、器官和多任务

默认配置是 CD68。如果目标改为 HLA-DR、CD45RO 或 Vimentin，必须同时设置：

~~~yaml
data:
  targets: [HLA-DR]
  submit_targets: [HLA-DR]
~~~

只传 --target 而没有允许 submit_targets，生成提交时会被拒绝。也可在相关命令中统一追加：

~~~powershell
--set 'data.targets=[HLA-DR]' --set 'data.submit_targets=[HLA-DR]'
~~~

默认 data.organ=auto。只训练 colon 可追加：

~~~powershell
--set data.organ=colon
~~~

四目标训练：

~~~powershell
conda run -n MEDICAL python -m virtual_staining.cli train --config configs/competition_multitask.yaml --data-root AUTO --run-id multitask_official_seed2026 --set train.save_top_k=1
~~~

多任务可能共享表示，也可能负迁移，必须和相同 ROI 协议下的单目标模型比较。

---

## 12. Performance V2 使用原则

V2 是研究框架，不是把所有开关一次打开。

正确顺序：

1. retained_unpromoted.yaml 建立真实 ROI/JPG 基线；
2. 固定 split、seed、预算和验证协议；
3. 每次只加入一个模块；
4. 至少两个 fold 或 seed 趋势一致；
5. 检查 ROI bootstrap、organ、border/interior 和 activity；
6. JPG 至少一项明确提高，另一项不能明显退化；
7. 通过后才写入最终配置。

3×3 context 还必须满足：

- 文件名能解析真实 ROI、row、col；
- 行列方向和边界连续性通过；
- train/val 无同 ROI 或相邻 patch 泄漏；
- context 只读取 DAPI；
- A3 在真实 ROI 分组 JPG 指标稳定提升。

未满足时应阻断 context，而不是猜坐标。

---

## 13. 控制磁盘空间

### 必须保留

- 最终 best checkpoint；
- 需要续训时的 last.ckpt；
- effective_config.yaml；
- validation 指标与逐图 CSV；
- manifest、审计、环境信息；
- 生成提交时使用的代码版本；
- 官方原始数据。

### 可以删除

- 失败 run；
- 已淘汰 smoke/sanity run；
- 被更好模型替代的重复 top-k；
- 可重建的预测图；
- TensorBoard events；
- 临时 submission_ready；
- Python/pytest/ruff 缓存。

真实训练后不要无差别清空整个 outputs/。

从一开始减少保存：

~~~powershell
--set train.save_top_k=1 --set validation.save_predictions=false
~~~

删除单个淘汰 run：

~~~powershell
$run = Resolve-Path .\outputs\performance_v2\obsolete_run
Remove-Item -LiteralPath $run -Recurse -Force
~~~

删除前必须确认 Resolve-Path 指向准备删除的 run，绝不能代入项目根、outputs 父目录或真实数据目录。

---

## 14. 开发和测试

~~~powershell
conda run -n MEDICAL python -m compileall src tests
conda run -n MEDICAL ruff check .
conda run -n MEDICAL pytest -q
conda run -n MEDICAL python -m virtual_staining.cli --help
~~~

测试使用临时合成图片，不依赖已删除的样例数据和样例权重。

修改数据代码后运行 discovery、manifest、dataset、transform 和 smoke 测试；修改模型或损失后运行 shape、backward、finite-loss 和 AMP；修改推理或提交后运行 inference、JPEG、命名和 ZIP 测试。

---

## 15. 常见问题

**找不到数据**  
确认 dataset/official 存在，设置 AIC_DATA_ROOT，运行 discover-data，并查看 artifacts/data_discovery.json。

**train manifest 为空**  
检查 train/val 五个 marker 是否同名配齐、可解码且尺寸一致，查看 bad_samples.csv 和 leakage_report.json。

**predict 提示 official test 为空**  
说明 test/DAPI 未放好或 manifest 未重建。不要绕过检查，也不要拿 val 冒充 test。

**CUDA 不可用**  
运行 env 和 nvidia-smi，确认 MEDICAL 安装的是 CUDA 版 torch，而不是 CPU wheel。

**显存不足**  
关闭其他 GPU 程序，然后尝试：

~~~powershell
--set train.batch_size=1 --set train.gradient_accumulation=8
~~~

仍不足时降低 base channels，关闭 context/cross-attention/global mixer；保持完整 256×256。

**Windows DataLoader 卡住**  
保持 num_workers=0。项目支持中文和空格路径，无需移动目录。

**颜色异常**  
不要自行用 OpenCV 默认 BGR 保存，项目 I/O 已统一 RGB/灰度和 uint8 round-trip。

**ZIP 多一层目录**  
使用 make-submission，并给 validate-submission 传 --zip-path；ZIP 第一层必须是 results/。

---

## 16. 最短操作清单

~~~powershell
$env:AIC_DATA_ROOT = (Resolve-Path .\dataset\official).Path

conda run -n MEDICAL python -m virtual_staining.cli discover-data --data-root AUTO
conda run -n MEDICAL python -m virtual_staining.cli build-manifest --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO
conda run -n MEDICAL python -m virtual_staining.cli audit-data --data-root dataset/official
conda run -n MEDICAL python -m virtual_staining.cli audit-roi-grid --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --output-dir artifacts/performance_v2
conda run -n MEDICAL python -m virtual_staining.cli train --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --target CD68 --run-id cd68_official_seed2026 --set train.save_top_k=1
conda run -n MEDICAL python -m virtual_staining.cli validate --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --target CD68 --checkpoint outputs/performance_v2/cd68_official_seed2026/checkpoints/best_ssim.ckpt
conda run -n MEDICAL python -m virtual_staining.cli predict --config configs/performance_v2/retained_unpromoted.yaml --data-root AUTO --target CD68 --checkpoint outputs/performance_v2/cd68_official_seed2026/checkpoints/best_ssim.ckpt --manifest artifacts/manifests/test_manifest.csv --output-dir outputs/performance_v2/cd68_official_seed2026/predictions
conda run -n MEDICAL python -m virtual_staining.cli make-submission --config configs/performance_v2/retained_unpromoted.yaml --pred-dir outputs/performance_v2/cd68_official_seed2026/predictions --test-manifest artifacts/manifests/test_manifest.csv --target CD68 --output-dir submission_ready
conda run -n MEDICAL python -m virtual_staining.cli validate-submission --submission-dir submission_ready/results --test-manifest artifacts/manifests/test_manifest.csv --target CD68 --zip-path submission_ready/submission_CD68.zip --artifact-dir submission_ready/validation
~~~

这套流程覆盖真实数据审计、训练、验证、test 推理、提交生成和 ZIP 校验。最终成绩仍取决于官方数据、完整训练、模型选择和赛事评分规则；任何未实际完成的长训练或 leaderboard 结果都不能提前声称。
