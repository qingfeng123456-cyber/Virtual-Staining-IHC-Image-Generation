# Model Card

## 用途

模型用于官方竞赛数据内的 DAPI→mIHC patch 级虚拟染色研究和提交，不是临床诊断工具，也不能代替真实免疫组化实验或病理医师判断。

## 数据范围

当前本地样例仅包含 colon、1,346 组 256×256 JPEG，以及 HLA-DR、CD45RO、Vimentin、CD68 四个目标。没有 liver/stomach、真实 ROI 元数据或官方 test。训练和审计只使用赛事数据，测试 DAPI 不进入训练统计。

## 方法

默认模型是从零训练的确定性图像恢复网络，另有 Residual U-Net 基线。输出由 sigmoid/clamp 限制到 `[0,1]`。不使用预训练权重、GAN、扩散或随机 latent。

## 指标

验证使用 scikit-image 的逐图 SSIM/PSNR，明确 data range。代理分不是官方综合分。任何报告必须说明数据划分来源、目标、是否 TTA/ensemble 及完整训练是否实际完成。

## 风险和限制

- 数字 stem 无法恢复真实 ROI，surrogate grouping 不能证明组织级无泄漏。
- JPEG 标签包含压缩误差；二次编码可能降低指标。
- 不同器官、扫描仪、染色批次和通道 mode 可能造成域偏移。
- 稀疏标记可能导致全背景或过平滑预测。
- 原型注意力是模型内部解释线索，不是生物学因果证据。
- 高 SSIM/PSNR 不保证临床意义、细胞级计数或诊断可靠性。

## 合规

禁止测试标签、隐藏信息、人工修改预测和测试驱动的模型选择。正式提交前必须运行 validator；当前缺少官方 test 时只能声明 smoke 工程验证。

