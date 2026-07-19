# 失败分析

## 全背景或全白

先核对 `[0,1]` 值域、sigmoid、uint8 round 和目标极性，再看训练/预测直方图与背景占比。确认 sparse marker 是否导致均值解；提高结构权重前先验证配对。不要用测试直方图调整阈值。

## 过平滑

比较 Charbonnier-only、SSIM、gradient 和 frequency 消融，查看高频能量及最差样本。过高 SSIM/MS-SSIM 或多尺度权重可能牺牲细节；frequency 只能低权重。检查 bilinear 上采样后卷积和深监督是否工作。

## 棋盘格或边缘伪影

确认没有反卷积，检查 padding、D4 逆变换和 patch 边界。若只在 JPEG 出现，比较保存前 tensor 与重新解码图，排查重复编码、subsampling 或 RGB/BGR。

## 空间错位

查看审计的边缘互相关、最优偏移分布和配对可视化。先排除 stem 配错、排序配对或不同扩展名问题。只报告原始错位；默认不自动配准标签。轻微平移增强必须由训练集审计触发，不能根据测试输出决定。

## 颜色偏差

核对 storage mode、logical channels 和 checkpoint ImageSpec。RGB 目标不能被 OpenCV BGR 交换；逻辑单通道输出保存为 RGB 时应复制同一通道。比较各通道差异和 round-trip 误差。

## 非有限 loss

逐项关闭 frequency、MS-SSIM、prototype 和 correlation，检查输入范围、空 batch、过小深监督尺寸和 AMP dtype。SSIM、FFT及 reduction 应在 float32。只处理明确 CUDA OOM，不能把 NaN 误判为显存问题。

## 验证异常高

检查 canonical key、完整哈希和真实/代理 group 是否跨 split。当前数字 stem 没有真实 ROI，异常高分尤其需要警惕相邻 patch 泄漏。不得把 surrogate group 验证描述成组织级证明。

## 提交失败

用 manifest 比较 expected/actual stem，检查 `_fake_fake`、空 test、目录多一层、尺寸/mode和 ZIP 首层。禁止手工逐个改名，应修复 writer 后重新生成并重跑 validator。

