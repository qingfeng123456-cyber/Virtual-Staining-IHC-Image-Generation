# Performance V2 消融协议与当前结果

更新日期：2026-07-16。

## 固定协议

P0 必须按 A0→A1→A2→A3 串行；每个阶段只改变 suite 声明的一个模块组：

1. A0：MultiMarkerRestorer base32，raw/no-TTA，V2 flag 关闭。
2. A1：仅增加显式 MSE 与连续 two-phase MSE/SSIM schedule。
3. A2：在 A1 上增加 Laplacian pyramid loss 与 base/detail head。
4. A3：在 A2 上增加经验证的 3×3 DAPI context 和 zero-init FiLM。

screen 使用 fold 0、seed 2026、20 epoch。confirm 至少使用两个独立的 ROI-grouped fold/seed。
模型选择以最终 JPG round-trip 为主，同时报告 float、uint8；至少一个 JPG 指标的 ROI-bootstrap
95% CI 必须明确提高，另一个不得显著下降，organ、border/interior、activity 分层不能集中
退化。无权威坐标时 A3 在训练前阻断。

## 已执行的严格 screen

本地 screen 使用完整 1,088 train / 258 validation、batch 4、gradient accumulation 2；A0–A2
各完成 20 epoch、global step 2,720。表中是 best checkpoint 的 raw/no-TTA 全量复评：

| 阶段 | 参数量 | float SSIM / PSNR | uint8 SSIM / PSNR | JPG SSIM / PSNR | 结论 |
|---|---:|---:|---:|---:|---|
| A0 | 4,107,364 | 0.781163 / 24.476280 | 0.780417 / 24.472490 | 0.779860 / 24.470988 | 本地 screen parent；非 ROI-safe |
| A1 | 4,107,364 | 0.780914 / 24.460609 | 0.780152 / 24.457405 | 0.779589 / 24.455889 | 退化，回滚到 A0 |
| A2 | 4,116,774 | 0.782342 / 24.578094 | 0.781553 / 24.575715 | 0.780959 / 24.574023 | 样例趋势；promotion=false |
| A3 | 未训练 | 未评估 | 未评估 | 未评估 | `blocked_unverified_grid` |

A1 vs A0 的 surrogate ROI bootstrap：JPG SSIM -0.000064，95% CI
[-0.000998, 0.000852]；JPG PSNR -0.006549 dB，95% CI
[-0.023958, 0.015235]；high-activity PSNR 有显著下降。A1 不晋级。

A2 vs A1：逐图 JPG 均值 +0.001370 SSIM、+0.118134 dB；按 9 个 surrogate group 聚合为
+0.001477 SSIM，95% CI [-0.000371, 0.003130]，以及 +0.108753 dB PSNR，95% CI
[0.058067, 0.166842]。然而 grid/方向/边界未验证，只有一份 fold/seed 证据，所以
`promotable=false`、`final_default_eligible=false`。surrogate CI 不能替代真实 ROI CI。

A3 阻断原因包括 `incomplete_filename_coordinates`、`unverified_filename_coordinates`、
`coordinate_direction_not_verified`、`boundary_continuity_not_verified` 和
`context_gate_disabled`。它没有训练、checkpoint 或性能数字。

证据文件：

- `artifacts/performance_v2/strict_p0_screen_snapshot_20260716.json`；
- `artifacts/performance_v2/A1_vs_A0_sample_promotion.json`；
- `artifacts/performance_v2/A2_vs_A1_sample_promotion.json`；
- `artifacts/performance_v2/performance_v2_p0_strict_screen_report.json`。

## CUDA runner 契约 smoke

另以 seed 2031、2 train / 2 validation、1 epoch、batch 1、BF16 在 RTX 4060 Laptop 上实际
执行 A0→A3，验证 runner 的 stage 顺序、parent 绑定、raw/EMA 三域记录和阻断行为：

| 阶段 | raw JPG SSIM / PSNR | EMA JPG SSIM / PSNR | 选择 / 状态 |
|---|---:|---:|---|
| A0 | 0.164230 / 8.122149 | 0.162124 / 8.085483 | raw / `not_promotable` |
| A1 | 0.164222 / 8.122141 | 0.162128 / 8.085498 | raw / `not_promotable` |
| A2 | 0.148075 / 6.944163 | 0.147417 / 6.870796 | raw / `not_promotable` |
| A3 | 未训练 | 未评估 | `blocked_unverified_grid` |

三次训练均为 2 optimizer step、OOM 重试 0；A3 在训练前阻断。报告位于
`artifacts/performance_v2/ablation_contract_registry_performance_v2_p0_strict_smoke_fold0_seed2031_report.json`。
这些两图指标只验证工程契约，不能覆盖 20 epoch screen，更不能作为 ROI-grouped 晋级证据。

此外，A5 target fine-tune 与 A6 organ fine-tune 的 tiny CPU lineage smoke 已实际执行，验证了
parent checkpoint、stage/global epoch 延续和 raw/EMA 独立选择。它们没有通过 A3 前置门禁，
因此属于工程验证，不属于正式 A5/A6 性能消融。

## 后续顺序

A3 只有在正式 grid 审计通过并在真实 ROI/JPG 指标晋级后，才允许将 A4 cross-attention、
A5 target fine-tune、A6 organ adapter、A7 D4 和 A8 ensemble 作为可晋级性能实验。已完成的
A5/A6 tiny lineage smoke 不改变这一前置条件。P1 的 hierarchical
prototypes、calibrator、Restormer-lite、activity sampler、grouped CV、FAMO、DAPI-MAE 和
learned ensemble 必须逐项加入；P2 保持可选。失败模块保留实验记录和回滚路径，默认关闭。
