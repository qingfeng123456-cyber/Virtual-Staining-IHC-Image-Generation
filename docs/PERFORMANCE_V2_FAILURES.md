# Performance V2 失败、阻断与回滚记录

更新日期：2026-07-16。

## 数据级阻断

- 本地 1,346 个 paired patch 全部是数字 stem，严格 ROI coordinate parser 成功 0 行。
- `filename_grid_verified=false`，坐标方向、边界连续性和真实 ROI 无泄漏均未验证。
- baseline snapshot 记录了 11 对跨 surrogate train/val 边界的连续 patch；surrogate group
  不能替代真实 ROI。
- official test 数量为 0；正式 predict/submission 保持未验收。
- 官方 PSNR normalize 公式未知，只报告原始 PSNR 和明确命名的本地 proxy。

回滚：context、cross-attention、neighbor consistency 和所有依赖权威邻域的模块保持关闭；
不得用数字 stem 或图像内容猜坐标绕过门禁。

## A0–A3 严格 screen

| 阶段 | 实际状态 | 证据与处置 |
|---|---|---|
| A0 | `not_promotable` | 20 epoch 样例 parent；validation 不是 verified ROI fold，只作回滚基线 |
| A1 | `not_promotable` | 三域均轻微退化；high-activity JPG PSNR 显著下降，回滚 A0 |
| A2 | `not_promotable` | 样例 JPG 趋势为正，但 SSIM CI 跨 0、只有一个 fold/seed 且 ROI/grid 未验证 |
| A3 | `blocked_unverified_grid` | 训练前阻断；无 checkpoint、无指标、无涨分声明 |

A1 的完整 258 张 JPG 均值相对 A0 为 -0.000271 SSIM、-0.015099 dB。surrogate ROI
bootstrap 的 SSIM/PSNR CI 都跨 0，且 activity high PSNR 的 CI 全部低于 0。因此 A1 明确
失败，不进入 retained config。

A2 相对 A1 的 surrogate ROI JPG PSNR CI 为 [0.058067, 0.166842] dB，但 SSIM CI 为
[-0.000371, 0.003130]，且数据门禁与两份独立证据要求均未满足。它只记录为“样例正向趋势”，
不能称作正式涨分。当前 retained config 仍回滚到 A0 架构/损失，不启用 A2。

A3 的精确阻断包括 `incomplete_filename_coordinates`、`unverified_filename_coordinates`、
`coordinate_direction_not_verified`、`boundary_continuity_not_verified`、
`context_gate_disabled`。不得降低 `require_verified_grid` 回避。

另一次 CUDA BF16 runner 契约 smoke 使用 seed 2031、2 train / 2 validation、1 epoch。A0/A1/A2
raw JPG 分别为 0.164230 / 8.122149、0.164222 / 8.122141、0.148075 / 6.944163；三项均正确
标记 `not_promotable`，A3 再次于训练前阻断。两张 validation 的短程波动不是性能结论，既不
用于否决 A2，也不能被挑选为涨分证据。

## raw/EMA 与 D4 回滚

legacy baseline 只有 16 optimizer step；raw+D4 JPG 为 0.406767 / 11.092427，EMA+D4 仅
0.303197 / 9.809074。A2 20 epoch checkpoint 的 raw JPG 为 0.780959 / 24.574023，EMA 为
0.718424 / 22.361521。两个实测 checkpoint 都选择 raw。

回滚：不默认加载 EMA，但训练仍独立保存/验证 raw 与 EMA，不能把短程结果外推为永久禁用
EMA。

A7 工程检查请求 D4，但 promotion gate 因无权威 ROI、border 分层和可用 JPG 晋级证据写入
`enabled=false`；predict 正确回退到 `tta=none`。D4 没有进入最终默认配置。

## DAPI-MAE 解释边界

DAPI-only 预训练、local_encoder 80/80 tensor 迁移、EMA 同步和 resume 至 2 epoch 均已通过
tiny CPU smoke；DAPI manifest hash 一致，且 `uses_target_labels=false`。但数据量只有 2 张
train / 2 张 validation，不能推断预训练是否涨分。完整 fold-local DAPI-MAE 保持未执行，
retained config 中 `pretrain.enabled=false`。

Target fine-tune 和 organ fine-tune 的 2 train / 2 validation CPU lineage smoke 已执行。前者
选择 raw，后者按 JPG SSIM 优先规则选择 EMA；organ checkpoint 正确保留 parent stage、
stage epoch 和 global epoch。它们只证明 A5/A6 工程路径和恢复契约可运行，没有通过 A3
前置门禁或真实 ROI 消融，因此 retained config 仍关闭对应功能。

## 操作错误与纠正

1. submission 第一次校验传入了不存在的 `submission.zip`；writer 实际输出
   `submission_CD68.zip`。工具返回失败后使用正确文件重跑，expected/actual/validated 均为
   8，`valid=true`、`zip_errors=[]`。该 ZIP 仍只是隔离 smoke。
2. 复杂度报告第一次把不存在的 A3 run 目录作为输入；工具失败，没有填造 A3 资源数据。
   移除该输入后从失败步骤重跑，生成 `complexity_report.json`，状态诚实保持 `partial`。
3. 旧版 ablation runner 使用固定报告名，后一次契约 smoke 可能覆盖已有报告。执行前已冻结
   legacy 副本，完成后恢复原报告；当前实现改为由 registry、suite、budget、fold 和 seed
   组成证据绑定文件名，并在同名文件已存在时硬失败。
4. 第一次 schema-v2 baseline companion 恰逢另一进程重建 active manifests，捕获了三个瞬态
   hash。该文件作为失败证据保留并明确 superseded，不做覆盖修补；schema-v3 先将四份 manifest
   冻结为 byte copy，再与原 schema-v1 snapshot 校验，当前有效 binding SHA256 为
   `4c7905b0c36f9814bb7abc7cbc2602854b7f55bdc6e3a6c57319c3dcb37cd78f`。
5. strict P0 snapshot 创建后，P0 suite、ROI audit 和追加式 registry 的当前字节发生变化，三个
   run 目录也各新增两份 post-capture validation 文件。没有逆向伪造旧 P0/audit；新增
   `strict_p0_screen_verification_20260716.json` 精确复核原 10-file run aggregates、冻结 registry
   prefix，并把无法恢复的两份历史字节明确列为 `historical_bytes_unavailable`。

历史集成阶段还保留过一个
`performance_v2_p0_A2_smoke_seed2026_invalid_constant_loss`：它未继承 A1 loss schedule，已标
为 `invalid_ablation_config_missing_A1_loss_schedule`，不参与严格 20 epoch 比较。严格 screen
使用独立 run ID 和冻结 snapshot，没有覆盖该失败产物。

## 尚未执行，不得声称成功或失败

- 80 epoch confirm、120–200 epoch full、两个独立 fold/seed；
- 真实 ROI bootstrap、A3 context 实训、A4–A8 正式性能消融（A5/A6 仅做过 tiny lineage
  工程 smoke）；
- P1/P2 各性能模块的正式消融；
- official test 推理、正式 submission 和 leaderboard。

这些项目是“未执行”，不是零分，也不是通过。正式 ROI 数据到位后必须从 discovery、manifest
和 grid audit 重启有效 A0→A3。

## 固定回滚准则

任何功能若只改善 float、降低最终 JPG、依赖少数 ROI、伤害 organ/border/activity 分层、发生
prototype collapse、超过资源预算，或缺少两个 fold/seed 的一致性，都保持关闭。实验记录必须
保留 parent、effective config、hash、raw/EMA 三域指标与失败原因。当前精确保留配置是
`configs/performance_v2/retained_unpromoted.yaml`。
