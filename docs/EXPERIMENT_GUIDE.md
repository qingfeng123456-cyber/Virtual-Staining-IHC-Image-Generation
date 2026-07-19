# 实验指南

## 统一规则

所有实验保存 run_id、effective config、config hash、seed、fold、target、模型、参数量、FLOPs、训练时间、峰值显存、验证指标、TTA、ensemble 和 checkpoint。模型选择只使用 validation，测试输出不能影响权重或配置。

## 推荐顺序

1. **E0 数据审计**：确认配对、通道、mode、ROI/surrogate grouping、重复哈希、目标分布和对齐。
2. **E1 Baseline**：Residual U-Net、Charbonnier、无 TTA，先得到可复现基线。
3. **E2 指标损失**：依次加入 SSIM、MS-SSIM、gradient，单变量消融。
4. **E3 主模型**：NAF 风格 encoder-decoder 与单任务 adapter。
5. **E4 多任务**：四目标联合，先关闭 prototype，与四个单任务比较。
6. **E5 Proto**：启用 shared/task prototypes、activation/diversity，检查注意力而非只看总分。
7. **E6 TTA**：每目标比较 none 和 D4；下降则关闭。
8. **E7 Ensemble**：top-3、三种子、baseline+main；权重只在 validation 拟合。

统一实验表写入 `artifacts/experiments.csv`，至少含：`run_id,git_commit,config_hash,seed,fold,target,model,params,flops,train_time,peak_vram,val_ssim,val_psnr,roi_ssim,roi_psnr,tta,ensemble,checkpoint`。

## 单变量原则

一次只改变一个主要因素，保留相同 manifest、seed 和 validation。多任务比较必须按每目标和宏平均分别判断，不能用某一目标提升掩盖另一目标下降。代理分只能作为辅助排序，原始 SSIM 和 PSNR 始终保留。

## 完整训练

先完成 smoke 和一个短基线，依据首个 epoch 的实测速度估算 200 epoch。长训练中断时保留 `last.ckpt` 并用 `resume`，不要重新划分数据。多种子建议 2026、3407、777；fold 必须按真实 ROI 或明确 surrogate group 划分。

