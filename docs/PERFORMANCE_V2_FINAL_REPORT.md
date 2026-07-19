# Performance V2 实施与实测报告

更新日期：2026-07-16。

## 结论先行

Performance V2 已在冻结 legacy baseline 之上完成增量工程实现，并实际完成本地样例的严格
A0、A1、A2 20 epoch screen、DAPI-only 预训练/迁移/恢复 smoke、raw/EMA 三域复测、D4
晋级门禁和隔离 smoke submission。另已实际运行一次 CUDA BF16 的 A0→A3 ablation runner
契约 smoke，并完成 target fine-tune 与 organ fine-tune 的 checkpoint lineage smoke。A3 在
两条执行链中都于训练前按设计被标记为
`blocked_unverified_grid`。

这些结果不能解释为正式涨分。当前 1,346 组 colon 样例只有数字 stem，没有权威
`ROI_row_col` 坐标；validation 仍由 surrogate group 构成，只有一个 fold/seed 证据，official
test 为空。A2 相对 A1 在样例上呈正向趋势，但严格 promotion 结果仍为 `false`。当前保留配置
是 `configs/performance_v2/retained_unpromoted.yaml`：legacy MultiMarkerRestorer base32，所有
尚未通过真实 ROI/JPG 门禁的 V2 性能模块关闭。

80 epoch confirm、120–200 epoch full、双 fold/seed、真实 ROI bootstrap、A3 实训、正式推理、
正式 ZIP 和 leaderboard 均未执行。

## baseline 冻结与 legacy 三域复测

冻结对象为
`outputs/smoke_pipeline_final_acceptance/checkpoints/best_ssim.ckpt`，SHA256 为
`252e129ea268552ee7a2ea9cc8af898d0d0418fcf50bb2a6d70292edf269c8d6`。它是 660,907
参数的 Residual U-Net base8、CD68，只训练了 2 epoch、16 个 optimizer step。冻结文件清单
与实施前快照分别位于 `artifacts/performance_v2/baseline_files.sha256` 和
`artifacts/performance_v2/baseline_snapshot.json`。

原始 schema-v1 快照没有被回写。首次 schema-v2 companion 在并行测试重建 active manifest
时捕获了三个临时哈希，已保留为失败证据并明确废止。修正后的不可变 schema-v3 companion 为
`artifacts/performance_v2/baseline_snapshot_binding_v3_20260716.json`，文件 SHA256 是
`4c7905b0c36f9814bb7abc7cbc2602854b7f55bdc6e3a6c57319c3dcb37cd78f`。它在一个 artifact
中绑定原快照（SHA256 `a06c6da373e417c14a8d71344f819470e6c49affcc6a19b1a06b326872226480`）、
原 hash 清单、checkpoint、effective config、四份 baseline-owned manifest 字节副本、三域 benchmark、ROI audit 和
实测 CPU inference；GPU peak 明确记为 unavailable，没有事后推造。冻结脚本在目标已存在时
失败，不覆盖 v1、失效的 v2 或修正后的 v3 artifact。

当前评估器对完整 258 行 validation 的实测如下；JPG 使用 RGB、quality=100、
subsampling=0、optimize=false。

| 权重 / TTA | float SSIM / PSNR | uint8 SSIM / PSNR | JPG SSIM / PSNR |
|---|---:|---:|---:|
| raw / none | 0.394293 / 11.084787 | 0.393903 / 11.084804 | 0.393385 / 11.085175 |
| raw / D4 | **0.407567 / 11.092067** | **0.407169 / 11.092082** | **0.406767 / 11.092427** |
| EMA / none | 0.222996 / 9.678874 | 0.222812 / 9.678828 | 0.222246 / 9.678773 |
| EMA / D4 | 0.304139 / 9.808947 | 0.303858 / 9.808947 | 0.303197 / 9.809074 |

legacy 短程 checkpoint 的 raw+D4 最好，EMA 明显滞后；这只说明该 checkpoint 的实际权重
选择，不能外推为长训练中永久禁用 EMA。后续训练会分别保存、验证和记录 raw/EMA。

## ROI grid 与邻域门禁

`artifacts/performance_v2/roi_grid_audit.json` 记录：总行数 1,346，严格坐标解析成功 0，
`parse_fraction=0.0`，`filename_grid_verified=false`，水平/垂直方向和边界连续性均无法由权威
坐标验证，`context_enabled=false`。主要阻断原因为：

- `unverified_filename_coordinates`；
- `coordinate_direction_not_verified`；
- `boundary_continuity_not_verified`。

数字 stem 的图像边缘连续性只能作审计证据。冻结 snapshot 还记录了 11 对跨 surrogate
train/val 边界的连续 patch，所以 surrogate-32 不能证明 ROI 或邻域无泄漏。context、
cross-attention 和 neighbor consistency 不得用数字序号或内容匹配绕过门禁。

## 严格 P0 20 epoch screen

A0、A1、A2 均使用 CD68、seed 2026、完整 1,088 train / 258 validation、20 epoch、batch 4、
gradient accumulation 2；每项最终 global step 为 2,720。下表是冻结 best checkpoint 以 raw、
无 TTA 在当前三域 Validator 上重新评估的 258 张逐图结果，不是 registry 中某个 epoch 的
近似值。

| 阶段 | 严格单变量变更 | 参数量 | float SSIM / PSNR | uint8 SSIM / PSNR | JPG SSIM / PSNR |
|---|---|---:|---:|---:|---:|
| A0 | legacy MultiMarkerRestorer base32，V2 flag 关闭 | 4,107,364 | 0.781163 / 24.476280 | 0.780417 / 24.472490 | 0.779860 / 24.470988 |
| A1 | 在 A0 上增加显式 MSE 与连续 two-phase MSE/SSIM schedule | 4,107,364 | 0.780914 / 24.460609 | 0.780152 / 24.457405 | 0.779589 / 24.455889 |
| A2 | 在 A1 上增加 Laplacian pyramid loss 与 base/detail head | 4,116,774 | 0.782342 / 24.578094 | 0.781553 / 24.575715 | 0.780959 / 24.574023 |
| A3 | 3×3 DAPI context 与 zero-init FiLM | 未训练 | 未评估 | 未评估 | `blocked_unverified_grid` |

A1 相对 A0 的逐图均值在三个域均轻微退化；JPG 为 -0.000271 SSIM、-0.015099 dB。
surrogate ROI 聚合的 bootstrap 结果同样没有明确提升：JPG SSIM 差值 -0.000064，95% CI
[-0.000998, 0.000852]；JPG PSNR 差值 -0.006549 dB，95% CI
[-0.023958, 0.015235]，且 high-activity 分层 PSNR 出现显著下降。因此 A1 保留为失败消融，
不进入默认配置。

A2 相对 A1 的逐图 JPG 均值为 +0.001370 SSIM、+0.118134 dB。按 9 个 surrogate group
聚合时，JPG SSIM 差值 +0.001477，95% CI [-0.000371, 0.003130]；JPG PSNR 差值
+0.108753 dB，95% CI [0.058067, 0.166842]，9/0 ROI win/loss。它是值得在正式数据上复验
的样例趋势，但不能晋级：坐标/grid/方向/边界均未验证，且只有 fold 0、seed 2026 一份独立
证据。`artifacts/performance_v2/A2_vs_A1_sample_promotion.json` 的最终结论是
`promotable=false`、`final_default_eligible=false`。

严格 screen 与代码/配置/manifest/checkpoint 的冻结哈希见
`artifacts/performance_v2/strict_p0_screen_snapshot_20260716.json`。A3 的报告明确记录
`training_started=false`、`metrics_claimed=false`。

## CUDA ablation runner 契约 smoke

为验证实际 CLI 编排、父子 stage、raw/EMA 选择、CUDA AMP、registry 和 A3 预训练阻断，另以
seed 2031、2 train / 2 validation、batch 1、gradient accumulation 1、BF16 在 RTX 4060
Laptop 上实际运行 A0→A3。命令显式将实际训练限制为 1 epoch；下表数字只来自两张 validation
图像，不能与 20 epoch screen 比较，也不是算法消融证据。

| 阶段 | raw JPG SSIM / PSNR | EMA JPG SSIM / PSNR | 实际选择 / 状态 | 峰值 VRAM |
|---|---:|---:|---|---:|
| A0 | 0.164230 / 8.122149 | 0.162124 / 8.085483 | raw / `not_promotable` | 829,913,088 B |
| A1 | 0.164222 / 8.122141 | 0.162128 / 8.085498 | raw / `not_promotable` | 829,913,600 B |
| A2 | 0.148075 / 6.944163 | 0.147417 / 6.870796 | raw / `not_promotable` | 844,012,544 B |
| A3 | 未训练 | 未评估 | `blocked_unverified_grid` | 未分配 |

三次 CUDA 训练均完成 2 个 optimizer step，OOM 重试为 0。A3 报告继续保留
`incomplete_filename_coordinates`、方向/边界未验证及 `context_gate_disabled` 等原因。证据绑定
到 `artifacts/performance_v2/ablation_contract_registry.csv` 和
`artifacts/performance_v2/ablation_contract_registry_performance_v2_p0_strict_smoke_fold0_seed2031_report.json`。
该 smoke 的用途是证明 runner 会执行、会保存真实三域与 raw/EMA 结果、也会在无权威 grid 时
阻断；A2 在两张图上的下降既不能判定模块失败，也不能被解释为正式涨分。

## A2 raw/EMA 选择

A2 best checkpoint 在完整 258 行 validation 上独立复测：

| 权重 | float SSIM / PSNR | uint8 SSIM / PSNR | JPG SSIM / PSNR |
|---|---:|---:|---:|
| raw | **0.782342 / 24.578094** | **0.781553 / 24.575715** | **0.780959 / 24.574023** |
| EMA | 0.719650 / 22.372465 | 0.718918 / 22.361718 | 0.718424 / 22.361521 |

因此该 screen checkpoint 选择 raw。证据位于
`artifacts/performance_v2/A2_weight_benchmark/baseline_benchmark.json`。EMA 仍保留为可选权重
源，后续每个 epoch 独立报告，不能因这次短程结果静默删除。

## A2 prototype attention 可视化契约

在不重新训练或改写 A2 checkpoint 的前提下，使用 SHA256
`527502f2a6a94ffe8cbb97755f2148256695bc931635e6c1ef0fd9ffe07eb1db` 的 A2 raw/no-TTA
checkpoint 对 4 张 validation 图像执行了真实诊断。固定 seed 2026 按 canonical key 的
SHA256 确定性选择 `colon/00128`–`colon/00131`；每张导出 8 个 shared 和 8 个 CD68 task
prototype attention，共 64 张 256×256 RGB PNG。产物位于
`outputs/performance_v2/prototype_attention_visual_contract/validation/per_image_prototype_diagnostics/prototype_attention_visuals/`，
manifest SHA256 为
`366507a3d2ac3613df9c55c7fd44cc724fd4e0c35fc97680fae9dc0145cbbb5d`，其中逐张记录
relative path、prototype index、尺寸及 PNG SHA256。

这只验证诊断的确定性、可审计性和真实模型输出，不是 prototype 消融或涨分证据。该导出功能
默认 `attention_visuals_enabled=false`；即使显式开启，聚合器也会在写文件前拒绝
`test`/`official_test` 输入（测试覆盖错误文本 `cannot observe test-split`），因此测试图像不能被
用作注意力挑选或人工模型选择。

## A8 ensemble 与 model soup 安全契约

learned ensemble 的公共入口不接受仅由 CLI 声明的 `source=validation/oof`。每个 prediction
和 target `.npy` 必须有 schema-v2 sidecar，但 sidecar 不再是自签名事实来源。CLI 必须实际读取
并 hash 完整 manifest、train/val audited manifests、ROI audit，以及 OOF 的 fold assignment；
随后逐项核对数组 SHA256/content hash、role、shape/dtype、有序 sample keys/organ、JPG domain、
manifest/audit/fold hashes、完整覆盖数、fold/group 与唯一 artifact ID。strict 路径要求 audit 对
全部 train/val 行完成 filename grid、方向、边界连续性和无 ROI/相邻 patch 泄漏验证；test 即使
打开 unsafe 也硬拒绝。grouped OOF 还要求每个样本恰好覆盖、所有 fold 均存在且同一 ROI group
不跨 fold；cross-validation 在其余 fold 拟合非负且和为 1 的权重，只在 held-out fold 计分。
surrogate/debug 只能通过显式 unsafe flag，sidecar 与输出均永久标记，不能重新进入 strict。

model soup 默认同样严格：成员必须有相同 architecture/state schema、精确 initialization
lineage、相同 raw/EMA/SWA weight source。用户给出的排序分数不参与选择；每个成员和每个
greedy candidate 都在同一完整 validation JPG 协议上重新计算，最终 soup 再次全量验证并保存
逐图 CSV。checkpoint 内绑定 validation manifest、train/val audited manifests、现场 ROI audit、
逐图 evidence 的 SHA256、样本数和 metric domain。legacy、截断验证、非 JPG、未验证 ROI 或
缺少 lineage 默认拒绝。唯一绕过方式是显式 unsafe override，文件名、报告、contract 和
checkpoint 均永久写入 unsafe 标志，且不能再作为 strict soup 成员。当前没有运行正式 grouped
OOF、没有生成正式 soup，也没有 A8 性能结果；A8 仍保持未晋级。

## DAPI-only 预训练、迁移与 resume smoke

`dapi_mae_contract_smoke_20260716` 在 CPU 上只读取 train split 的 2 张 DAPI，1 epoch、2
global step，loss=0.519372。预训练 checkpoint 声明 `uses_target_labels=false`，其冻结 DAPI
manifest hash 为 `35ec3b271252ac8b362cf1bf42f13a7e35ed4403`。

随后 `camp_pretrain_transfer_smoke_20260716` 使用同一 DAPI manifest：

- manifest hash 严格相等并验证通过；
- 仅迁移 `local_encoder`，80/80 tensors 成功，shape mismatch、source-only 和缺失 key 均为 0；
- source checkpoint SHA256 为
  `10eb27c705b93b4231994e77f953bdac5f316a610c64ed496de4bd9433615b80`；
- EMA 在迁移后同步；
- 先完成 1 epoch，再从 `last.ckpt` 恢复至 epoch 2，最终 global step=4；
- 两个 epoch 都独立比较 raw/EMA 并选择 raw；最终 JPG smoke 为 raw
  0.399602 / 8.945846，EMA 0.391720 / 8.618183。

这是 2 train / 2 validation、CPU tiny-width 的契约与恢复 smoke，不是 MAE 性能消融，也不
证明预训练有效。完整 fold-local DAPI-MAE 尚未运行。

随后又以 2 train / 2 validation 的 CPU tiny run 验证多阶段父子 lineage：

- `performance_v2_target_finetune_contract_20260716` 完成 1 epoch、global step=2，JPG 为 raw
  0.399616 / 8.956604、EMA 0.399594 / 8.945862，按 JPG 规则选择 raw；
- `performance_v2_organ_finetune_lineage_20260716` 从 target checkpoint 继续，checkpoint 记录
  stage index=2、stage epoch=1、global epoch=2，并保留已完成
  `multitask_pretrain`、`target_finetune` 的 lineage；JPG 为 raw
  0.399569 / 8.963526、EMA 0.399609 / 8.956094，按 SSIM 优先规则选择 EMA。

这两个 run 只验证 stage 转换、父 checkpoint 溯源、冻结范围、global epoch 和 raw/EMA 独立
选择。样本极少且不是 verified ROI fold，不能据此评价 A5/A6 是否提高性能；正式 A5/A6
消融仍未执行。

## D4 门禁、推理与隔离 submission

A7 工程检查在 A2 smoke checkpoint 上请求 D4，并用严格 promotion gate 比较。由于无权威
ROI grid、border 分层不可解析且没有可用的 JPG 晋级证据，
`validation/tta_decisions.json` 写入 `enabled=false`。后续 predict 自动回退到实际
`tta=none`，而不是无条件应用 D4。

该推理链路处理 8 张隔离 smoke 图像，使用 raw 权重；smoke submission validator 的实测为
expected=8、actual=8、validated=8、`valid=true`、`zip_errors=[]`。正确 ZIP 名为
`submission_CD68.zip`，产物位于
`outputs/performance_v2/performance_v2_p0_strict_A2_smoke_seed2026/a7_gate_submission/`。
它不是 official test、不是根 `results/`，不得作为正式提交或 leaderboard 结果。

## 实测资源与复杂度

`artifacts/performance_v2/complexity_report.json` 只汇总可追溯的实测 artifact，整体状态为
`partial`。核心结果如下：

| 阶段 | 参数量 | MACs | 聚合吞吐 / 最后 epoch | 峰值 VRAM | OOM 重试 |
|---|---:|---:|---:|---:|---:|
| A0 screen | 4,107,364 | 9,982,969,856 | 8.298 / 8.078 img/s | 3,062,619,136 B | 0 |
| A1 screen | 4,107,364 | 9,982,969,856 | 7.056 / 7.613 img/s | 3,062,619,136 B | 0 |
| A2 screen | 4,116,774 | 10,589,571,072 | 7.467 / 7.589 img/s | 3,118,437,376 B | 0 |

legacy benchmark 中，D4 相对 none 的实测运行时倍率为 raw 2.151×、EMA 2.098×。DAPI-MAE
条目因其 checkpoint 名为 `dapi_mae_last.ckpt` 而非通用 `last.ckpt`，在复杂度报告中诚实标为
`partial`；报告没有填造缺失的训练吞吐或 context runtime。

## 自动化测试状态

最终实现树的 fail-closed QA 实际通过：`compileall`、`ruff check .`、CLI help、`pip check`、
AST 空实现扫描以及 `pytest` 均返回 0；JUnit 为 **361 passed、0 failed、0 errors、0 skipped**。
机器证据写入 `artifacts/performance_v2/final_qa.json` 和
`artifacts/performance_v2/final_pytest.xml`。冻结脚本会拒绝陈旧 JUnit、tree hash 不一致、缺失
critical artifact 或任何失败/skip，而不是硬编码通过。实际入口为：

```powershell
conda run -n MEDICAL python -m compileall src tests
conda run -n MEDICAL ruff check .
conda run -n MEDICAL pytest -q
pwsh -File scripts/run_performance_v2_final_qa.ps1 `
  -PythonPath C:\Users\16355\miniconda3\envs\MEDICAL\python.exe
```

测试数量只证明工程回归，不替代 verified ROI/JPG performance gate。

## 已执行命令与纠错记录

严格 screen 由持久化 suite、budget、run ID 和 effective config 绑定；shell 没有保存逐字符
argv，因此以下按实际入口和持久化参数列出，不补写不存在的输出：

```powershell
conda run -n MEDICAL python -m virtual_staining.cli run-ablation `
  --suite configs/performance_v2/ablation/p0.yaml `
  --budget screen --from-stage A0 --through-stage A3

conda run -n MEDICAL python -m virtual_staining.cli pretrain-dapi `
  --config configs/performance_v2/dapi_pretrain.yaml `
  --run-id dapi_mae_contract_smoke_20260716 --max-epochs 1 `
  --set data.max_train_samples=2 --set train.device=cpu `
  --set train.batch_size=1 --set train.amp=false `
  --set "pretrain.widths=[8,16,32,64]" `
  --set "pretrain.encoder_depths=[1,1,1,1]"

conda run -n MEDICAL python -m virtual_staining.cli train-v2 `
  --config configs/performance_v2/camp_smoke.yaml `
  --run-id camp_pretrain_transfer_smoke_20260716 `
  --pretrain-checkpoint outputs/performance_v2/dapi_mae_contract_smoke_20260716/checkpoints/dapi_mae_last.ckpt `
  --max-epochs 1 --set data.max_train_samples=2 --set data.max_val_samples=2 `
  --set model.use_sobel_input=true --set train.batch_size=1 `
  --set inference.batch_size=1
```

同一 CAMP run 随后使用 `resume` 恢复到 2 epoch；A2 另执行 raw/EMA benchmark、D4 validate、
8 张 smoke predict、make-submission 和 validate-submission。另实际执行了 seed 2031、CUDA
BF16、2 train / 2 validation、1 epoch 的 A0→A3 runner 契约 smoke，以及 target/organ
fine-tune lineage CPU smoke；其持久化 effective config、metrics、registry 与 checkpoint 是
实际参数和结果的准绳。

已记录的操作/证据错误均被工具显式拒绝或由不可变 companion 披露并纠正：

1. 第一次 submission 校验手工传入了不存在的 `submission.zip`；writer 实际命名为
   `submission_CD68.zip`。改用真实 ZIP 后校验 8/8 通过。
2. 第一次生成复杂度报告时把不存在的 A3 run 目录作为输入；命令失败且未伪造 A3 数据。
   去掉 A3 后重跑成功，报告状态保持 `partial`。
3. schema-v2 baseline companion 在 manifest 重建窗口捕获瞬态 hash；保留失败文件，另建
   schema-v3 冻结 byte copies，不覆盖历史。
4. strict screen 后续路径追加导致当前 hash 漂移；新增 verification companion 精确复核仍可
   恢复的 run trees/registry prefix，并把旧 P0 YAML/ROI audit 标为历史字节不可用。

## 当前保留与未完成项

- 保留：`configs/performance_v2/retained_unpromoted.yaml`，raw/no-TTA 本地回滚基线；新模块
  feature flag 均可关闭。
- A1：样例 screen 退化，禁用。
- A2：样例趋势值得正式复验，但 promotion=false，禁用。
- A3：`blocked_unverified_grid`，没有 checkpoint 或指标。
- A4–A8 与 P1/P2：正式性能消融未通过前置门禁，不进入默认配置；A5/A6 仅完成 tiny lineage
  工程 smoke，不构成晋级；A8 只完成 sidecar/grouped-OOF 和 strict soup 安全契约，没有正式
  ensemble/soup 性能证据。
- official test：为空，未生成正式 submission。

正式 `ROIxxx_row_col` 数据到位后，必须先重跑数据发现、manifest、ROI grid/方向/边界/泄漏
审计，再在同一 verified grouped folds 上重新执行 A0→A3。只有至少两个独立 fold/seed 趋势
一致，最终 JPG 的 ROI-bootstrap 至少一项 95% CI 明确提高、另一项不显著下降且各分层无集中
退化，模块才可晋级。

当前准确表述是：**Performance V2 增量工程、样例 20 epoch screen、CUDA runner 契约 smoke
和多阶段 lineage smoke 已执行；没有 V2 模块获得正式晋级，完整训练、真实 ROI 性能验收和
官方提交仍未完成。**
