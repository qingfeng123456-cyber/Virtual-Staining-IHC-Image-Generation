# 消融计划

所有消融固定 manifest、seed、epoch budget、优化器和保存方式，逐目标报告 SSIM/PSNR，不用宏平均掩盖单目标退化。

| 编号 | 模型/改变 | 目的 |
|---|---|---|
| A0 | Residual U-Net + Charbonnier | 最小稳定基线 |
| A1 | A0 + SSIM | 验证指标对齐收益 |
| A2 | A1 + MS-SSIM | 验证多尺度结构 |
| A3 | A2 + Gradient | 验证边缘恢复 |
| A4 | A3 + Frequency | 验证纹理频谱，保持低权重 |
| A5 | MultiMarkerRestorer 单任务 | 比较主干能力 |
| A6 | 四目标共享、无 prototype | 判断多任务共享收益 |
| A7 | A6 + task adapter/FiLM | 判断任务条件化收益 |
| A8 | A7 + shared/task prototype | 判断原型模块收益 |
| A9 | 无 TTA vs D4 | 按目标决定推理策略 |
| A10 | 单 checkpoint vs top-3/多 seed | 判断集成收益 |

原型实验同时记录 activation/diversity loss、原型间余弦矩阵和注意力图。若单目标指标下降，最终提交配置关闭对应模块。TTA 和 ensemble 只以 validation 决策。每项至少保存 effective config、环境、训练时间、峰值显存和 checkpoint。

