# 技术报告详细提纲

## 一、摘要

说明虚拟染色任务的输入、四类目标、空间一一对应要求和主要评价指标。概括工程选择确定性图像恢复而不是 GAN/扩散的原因：样本规模有限、SSIM/PSNR偏好像素与结构一致、确定性更容易复现和集成。摘要必须区分本地验证与官方评测，不写未实际取得的数值。列出核心贡献：自动数据审计、严格防泄漏、稳定基线、共享多任务恢复网络、任务适配器、轻量原型模块和提交验证闭环。

## 二、问题背景与挑战

介绍 DAPI 表征细胞核而 mIHC 标记反映不同生物结构，二者映射具有一对多和组织异质性。分析 JPEG 压缩、稀疏标记、亮暗极性差异、小样本、可能的轻微错位和器官域偏移。说明相邻 patch 若来自同一 ROI，随机 patch 划分会高估泛化。当前样例只有数字 stem，真实 ROI 无法恢复，因此报告必须把 surrogate grouping 的限制单独列出。

## 三、数据发现与治理

描述配置、环境变量、工作区候选和父目录一层的发现优先级。说明候选评分如何综合 DAPI、目标目录、split、图像量和文件名特征，并解释为什么当前选择 `dataset/dataset_sample` 而不是 marker 目录。给出标记名称规范化规则以及 `organ/stem` canonical key。强调配对按规范化 stem 而非遍历顺序。

详述 manifest 字段：organ、split、roi_id、patch_id、canonical key、输入/目标相对路径、尺寸、通道、格式、文件大小、配对状态和校验和。说明真实 ROI 正则、surrogate block、完整哈希 union-find 和固定种子 group split。列出泄漏检查：canonical key、哈希、ROI、完全相同图像、缺失目标和破损图像。明确原始文件永不删除。

## 四、数据统计与图像规格

介绍流式 histogram、Welford 均值方差、固定抽样分位数、Pearson/Spearman、背景占比、边缘强度和每组 patch 数。解释 storage channels 与 logical channels 的区别，以及近灰度 RGB 的量化阈值。说明模型内部 `[0,1]`、CHW、RGB，保存时 uint8 round、JPEG quality 100、subsampling 0，并通过 round-trip 测试限制编码损失。

空间对齐部分给出 Sobel 边缘和 ±4 像素互相关诊断，区分近似对齐、小偏移和严重错配。强调不根据诊断自动修改官方标签；轻微同步平移只作为可关闭的鲁棒增强。展示至少 16 组 DAPI/目标配对及通道统计图，并说明图像均来自训练部分。

## 五、基线模型

给出 Residual U-Net 的四尺度结构图和每层通道。解释 GroupNorm 在小 batch 下优于 BatchNorm，SiLU、residual block 和 skip connection 的作用，解释 bilinear upsample+conv 如何避免反卷积棋盘格。列出输入/输出通道自动配置、sigmoid 和参数/FLOPs统计。基线承担数据管线验证、最小提交和 ensemble diversity 三个角色。

## 六、MultiMarkerRestorer 主模型

逐层描述可选 Sobel 输入、共享四尺度 NAF 风格 encoder、bottleneck、共享 decoder、任务 adapter 和独立 head。NAF block 部分解释 depthwise convolution、simple gate、channel attention 和 residual scaling。说明共享 encoder 避免复制四套完整网络，任务 adapter 以较小参数量保持标记特异性。

描述 task embedding 如何生成 FiLM scale/shift；给出单目标 `task_name` 和一次四目标输出的数据流。解释 64、128、256 深监督的位置和权重。报告实际参数量、FLOPs、峰值显存和不同 base channel profile，不以理论配置冒充本机实际配置。

## 七、共享与任务原型

定义共享原型矩阵和每任务原型矩阵，写出 feature/prototype L2 normalize、cosine similarity、temperature softmax、prototype aggregation、projection 和 residual fusion 公式。解释共享原型希望表达跨标记结构，任务原型希望表达标记特异模式。给出 activation/commitment 与 diversity loss，说明权重必须远小于重建损失。

解释原型注意力图只能用于模型内部可解释性和错误分析，不代表生物学因果。列出无原型、共享原型、共享+任务原型的消融，并在任何目标下降时关闭该模块。

## 八、损失与优化

分别推导 Charbonnier、SSIM、MS-SSIM、Sobel gradient、log amplitude frequency、结构权重、任务相关和原型损失。解释结构权重由梯度或局部变化产生，避免假定亮像素为阳性。解释 frequency 不比较相位且低权重，防止训练不稳定。说明小尺寸深监督输出如何减少 MS-SSIM 层级。

训练策略包括 AdamW、学习率、weight decay、五 epoch warmup、cosine decay、gradient clipping、AMP、EMA、early stopping和 top-k。说明 bfloat16/float16选择、GradScaler条件和关键 loss float32。给出 ≤8 GiB 的 batch/累积方案及最多三次明确 OOM 自适应。

## 九、验证、推理与集成

说明最终指标使用 scikit-image，逐图计算后再 mean/median，灰度二维、RGB `channel_axis=-1`，明确 `data_range=1` 或 255。报告每目标、宏平均、分组聚合、最差 10% 和无穷 PSNR处理。代理分公式及上下界来自配置，并醒目标记不等于官方分。

描述 checkpoint 的完整字段和 resume 验证。推理固定顺序、无 shuffle、原尺寸保存、EMA权重和耗时/峰值显存。列出 D4 八种变换及严格逆变换，强调按目标验证后启用。ensemble 包括 top-k、多 seed、多 fold、baseline+main；非负权重只用 validation 拟合。

## 十、提交工程

展示 `results/test/<TARGET>/<stem>_fake.jpg`，说明文件数、命名、尺寸、mode、像素范围、异常图和 manifest 对应校验。解释 ZIP 首层为什么必须为 results。区分 smoke submission 和官方 submission：当前官方 test 缺失，smoke 只能证明软件链路，不能宣称正式提交通过。

## 十一、实验和消融结果组织

按 E0–E7 顺序组织表格。每张表包含 config hash、seed、fold、target、参数量、FLOPs、训练时间、显存、SSIM/PSNR、分组指标、TTA、ensemble和 checkpoint。结果章节只填实际运行数据；未完成的 200 epoch 实验明确标记未完成。分析单任务与多任务、prototype、各损失、TTA和 ensemble 的收益及代价。

## 十二、失败案例与局限

展示全背景、过平滑、棋盘格、错位、颜色偏差和稀疏目标的典型案例。将问题分别追溯到数据、值域、模型、损失、保存或域偏移。重点说明当前只有 colon 样例、真实 ROI 不可验证、官方 test 缺失、JPEG 标签和器官泛化限制。声明模型不能用于临床诊断。

## 十三、结论与后续研究

总结数据正确性、稳定基线和确定性恢复优先于复杂生成模型的工程经验。后续可研究更多真实 ROI 划分、跨器官验证、轻量自监督但严格排除测试图、模型蒸馏和校准。扩散/GAN只能作为独立可选研究分支，必须保持现有确定性 pipeline 和提交接口不受影响。

