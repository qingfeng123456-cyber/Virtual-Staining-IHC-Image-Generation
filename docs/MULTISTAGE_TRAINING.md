# Performance V2 多阶段训练与恢复契约

更新日期：2026-07-16。

## 阶段顺序

1. A0 baseline reproduction 固定数据、fold、seed、架构和 float/uint8/JPG 协议。
2. 可选 fold-local DAPI-MAE 仅使用当前训练 fold 的 DAPI；不得读取 target label、validation
   或 test。
3. Multitask pretraining 先按 organ、再按 marker 平衡采样；equal task weights 始终保留为
   对照。
4. Target fine-tuning 选择单一 marker，可短暂冻结 encoder 并降低学习率。
5. Organ fine-tuning 只更新约定的 organ/marker adapter、late decoder 和 calibrator。
6. Metric alignment 使用 phase-B loss、小学习率和轻增强，不改变架构。

每个架构、loss、sampler、pretrain、adapter、TTA 与 ensemble 都必须由 feature flag 控制。
未经真实 ROI/JPG 消融晋级的阶段不能写入最终默认配置。

## DAPI-only 预训练安全契约

预训练 loader 使用 DAPI-only dataset，并拒绝非 train split。预训练与下游 restoration 各自在
run 目录写入排序一致的 `manifests/dapi_pretrain_manifest.csv`；checkpoint 记录其 hash。迁移
前必须满足：

- source `uses_target_labels=false`；
- source/target DAPI manifest hash 完全一致；
- 只迁移声明范围 `local_encoder`；
- 记录 source checkpoint SHA256、成功/跳过/shape mismatch keys；
- 迁移后同步 EMA；
- DAPI pretrain checkpoint 与普通 restoration initial checkpoint 不得同时指定。

不匹配必须硬失败，不能退化为静默部分加载。test 图像不得用于预训练、统计、校准或选择。

## 已执行 smoke

`dapi_mae_contract_smoke_20260716` 实际使用 2 张 train DAPI、CPU、1 epoch、2 global step，
loss=0.519372；manifest hash 为
`35ec3b271252ac8b362cf1bf42f13a7e35ed4403`。

`camp_pretrain_transfer_smoke_20260716` 验证同一 hash，local_encoder 80/80 tensors 成功迁移，
无 shape mismatch/source-only/missing key，EMA 同步。它先训练 1 epoch，再从 last checkpoint
恢复到 epoch 2，最终 global step=4。raw 与 EMA 每个 epoch 独立验证，两个 epoch 都选择 raw；
第二个 epoch 的 JPG smoke 为 raw 0.399602 / 8.945846，EMA 0.391720 / 8.618183。

上述 2 train / 2 validation tiny CPU run 只验证契约、迁移、checkpoint 和 resume，不证明
DAPI-MAE 提升。完整 fold-local pretraining 尚未执行，当前 retained config 中关闭。

## Target 与 organ lineage smoke

`performance_v2_target_finetune_contract_20260716` 使用 2 train / 2 validation、CPU、1 epoch，
完成 global step=2。JPG round-trip 为 raw 0.399616 / 8.956604、EMA
0.399594 / 8.945862，按 JPG SSIM、再 PSNR 的规则选择 raw。

`performance_v2_organ_finetune_lineage_20260716` 从 target checkpoint 继续 1 epoch。其 checkpoint
记录 stage index=2、stage epoch=1、global epoch=2，completed stages 保留
`multitask_pretrain` 和 `target_finetune`；没有把 child stage 误重置成一条新的 lineage。JPG
为 raw 0.399569 / 8.963526、EMA 0.399609 / 8.956094，因此选择 EMA，说明权重源是按当前
checkpoint 实测选择而非固定假设。

这两次 tiny run 只验证父 checkpoint 溯源、阶段转换、训练参数重配置、global epoch 与
raw/EMA 独立验证。当前数据不是 verified ROI fold，A3 仍被阻断，所以它们不是 A5/A6 的
正式性能消融，也不会开启 retained config 中的 fine-tune/organ adapter。

## checkpoint 与 resume

checkpoint 必须保存 stage name/index/epoch、model/optimizer/scheduler/scaler、EMA、RNG、
DataLoader generator、loss-schedule progress、manifest hashes、fold/activity sampler provenance、
raw/EMA 选择和可选 task optimizer state。resume 必须重新验证：

- restoration train/validation manifest；
- DAPI pretrain manifest 与迁移 provenance；
- target set、ImageSpec、stage contract；
- grouped fold/activity sampler state。

任何数据、target、stage 或 feature contract 变化都必须使用新 run ID，不能静默续跑或覆盖
已有 run。

## 权重选择

raw 与 EMA 分别计算 float、uint8 和最终 JPG round-trip；以 JPG mean SSIM、再以 JPG PSNR
决胜，checkpoint 记录 `selected_weight_source`。A2 20 epoch 样例中 raw JPG 为
0.780959 / 24.574023，EMA 为 0.718424 / 22.361521，因此选择 raw。这个决定只绑定该
checkpoint，长训练仍需重新比较。

最终阶段晋级还必须通过真实 ROI-grouped promotion gate；单个 smoke 的权重选择不能替代
双 fold/seed confirm。
