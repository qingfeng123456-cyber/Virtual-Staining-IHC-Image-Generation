# AutoDL 运行指南

> 本文档是 AutoDL 上运行本项目的推荐入口。旧 checkpoint 已遗失时，按顺序执行下面两条命令即可重新训练、验证并生成竞赛提交包。

## 前置准备（仅首次）

```bash
cd /root/autodl-tmp/project
conda activate MEDICAL

# 安装项目包（editable 模式，注册 virtual_staining 模块到 Python 路径）
pip install -e .

# 修正 OMP_NUM_THREADS（避免 libgomp 报错）
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=$(nproc)

# 确认 GPU 可用
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

> **如果不想 pip install（怕污染环境）**，用 PYTHONPATH 替代：
> ```bash
> cd /root/autodl-tmp/project
> conda activate MEDICAL
> export PYTHONPATH="$PWD/src:$PYTHONPATH"
> ```
> 加到 `~/.bashrc` 里一劳永逸：
> ```bash
> echo 'export PYTHONPATH="/root/autodl-tmp/project/src:$PYTHONPATH"' >> ~/.bashrc
> source ~/.bashrc
> ```

## 命令 1：训练 + 验证（一条命令完成）

```bash
python -m virtual_staining.cli --log-root log autodl-run \
  --config configs/initial_round_cd68_retrain_v2.yaml \
  --data-root AUTO \
  --target CD68 \
  --run-id cd68_retrain_v2_seed2026
```

**这条命令自动完成：**
1. 环境检测（CUDA/GPU/PyTorch 版本）→ `artifacts/environment.json`
2. 数据发现与 manifest 构建（train/val/test 自动划分）→ `artifacts/data_discovery.json`
3. 数据审计（ROI 对齐、泄漏检查）→ `artifacts/data_audit.json`
4. 训练最多 84 epoch（16.2M 参数的 NAF 风格 MultiMarkerRestorer base64，不是更大的 Transformer）
5. 每 3 epoch 比较 raw/EMA，并以最终 JPG round-trip SSIM 为主选择 checkpoint
6. 训练结束比较无 TTA 与 D4 TTA，写入三域指标和 ROI 分组结果；正式提交配置固定使用 D4 → `validation/metrics.json`
7. 写入汇总报告 → `outputs/initial_round_v2/<run-id>/pipeline_report.json`

**实测耗时：** seed-2026 在 RTX 4090 D 上完成 84 epoch 用时 3 小时 25 分，无 OOM；其他机器仍以实际日志为准。

**快速验证（可选）：** 加 `--max-epochs 3` 跑 3 epoch 确认流程通畅，再跑完整训练。

### 查看验证性能

```bash
python -c "
import json
m = json.load(open('outputs/initial_round_v2/cd68_retrain_v2_seed2026/validation/metrics.json'))
plain = m['domains']['jpg']['macro']
d4 = m['tta_comparison']['d4']['domains']['jpg']['macro']
print('无 TTA:', round(plain['mean_ssim'], 6), round(plain['mean_psnr'], 5))
print('D4:', round(d4['mean_ssim'], 6), round(d4['mean_psnr'], 5))
"
```

关键文件：

| 文件 | 内容 |
| --- | --- |
| `outputs/initial_round_v2/<run-id>/validation/metrics.json` | 完整验证指标（float/uint8/jpg 三域，ROI 分组，分层） |
| `outputs/initial_round_v2/<run-id>/validation/per_image.csv` | 无 TTA 的逐图 SSIM/PSNR |
| `outputs/initial_round_v2/<run-id>/validation/per_image_tta_d4.csv` | D4 TTA 的逐图 SSIM/PSNR |
| `outputs/initial_round_v2/<run-id>/validation/tta_decisions.json` | 保守 promotion 审计；正式配置采用 `tta_policy=configured` 时仅作诊断，不覆盖 D4 |
| `outputs/initial_round_v2/<run-id>/pipeline_report.json` | 汇总报告（环境+训练+验证） |
| `outputs/initial_round_v2/<run-id>/checkpoints/best_ssim.ckpt` | 最佳 checkpoint |

> **重要：** val SSIM/PSNR 是选择 checkpoint 的代理证据，**不是** official test 分数。official 分数只能由上传 ZIP 到竞赛平台后获得。

## 命令 2：打包竞赛提交（一条命令完成）

训练完成后，推荐显式指定本次训练生成的最佳 checkpoint：

```bash
python -m virtual_staining.cli --log-root log autodl-submit \
  --config configs/initial_round_cd68_retrain_v2.yaml \
  --data-root AUTO \
  --target CD68 \
  --checkpoint outputs/initial_round_v2/cd68_retrain_v2_seed2026/checkpoints/best_ssim.ckpt
```

> **不要给 `autodl-submit` 添加 `--run-id`。** 当前 CLI 只有训练命令
> `autodl-run` 支持 `--run-id`；提交命令使用 `--checkpoint` 确定要提交哪个
> run 的模型。错误地添加 `--run-id` 会得到
> `unrecognized arguments: --run-id ...`。

**这条命令自动完成：**
1. 加载明确指定的 `best_ssim.ckpt`，避免存在多个 run 时选错模型
2. 对官方 test 集（1346 张 DAPI）使用最佳 EMA 权重和默认 D4 推理 → `predictions/`
3. 构建竞赛要求的 `results/` 目录结构与 ZIP → `submission/`
4. 严格校验 ZIP（文件名、数量、尺寸、模式）

当前正式配置明确写有：

```yaml
inference:
  weight_source: best_jpg
  tta: d4
  tta_policy: configured
```

`configured` 表示配置中的 D4 是权威设置。即使旧 run 的
`tta_decisions.json` 因器官字段为 `unknown` 而错误记录 `enabled=false`，提交程序也会忽略该错误决策，不需要删除或修改日志文件。

提交完成后必须核对：

```bash
python -c "import json; r=json.load(open('outputs/initial_round_v2/cd68_retrain_v2_seed2026/inference_report.json')); print('TTA=',r['tta'],'策略=',r['tta_policy'],'权重=',r['loaded_weight_source'],'数量=',r['count'])"
```

正确输出应包含：

```text
TTA= d4 策略= configured 权重= ema 数量= 1346
```

### 上传竞赛平台

最终 ZIP 在：

```
outputs/initial_round_v2/cd68_retrain_v2_seed2026/submission/submission_CD68.zip
```

将此 ZIP 上传到竞赛平台，平台返回的分数才是 official test score。

## 为什么训练时不运行 D4、提交时运行 D4

D4 是测试时增强，不是训练模块。训练仍然使用随机同步旋转/翻转，每个 batch 只做一次前后向；提交时则对同一张 DAPI 分别进行 8 种旋转/翻转，预测后恢复方向并平均。它不修改 checkpoint，也不使用测试标签，只增加推理计算量。

本次同一 `best_ssim.ckpt`、同一 1,292 张验证集的实测结果：

| 推理方式 | JPG SSIM | JPG PSNR |
| --- | ---: | ---: |
| 无 TTA | 0.804480 | 25.97599 |
| D4 | **0.809280** | **26.19677** |

D4 在 5 个验证 ROI 上的 SSIM 和 PSNR 都提高，因此正式 CD68 配置默认启用。代价只发生在预测阶段，不会让训练变成 8 倍。

## 可选：仅有一个训练 run 时自动选择 checkpoint

如果 `outputs/` 下确定只有一个可用的 `best_ssim.ckpt`，可以省略
`--checkpoint`：

```bash
python -m virtual_staining.cli --log-root log autodl-submit \
  --config configs/initial_round_cd68_retrain_v2.yaml \
  --data-root AUTO \
  --target CD68
```

此时程序会选择 `outputs/**/best_ssim.ckpt` 中修改时间最新的文件。存在多个
实验时可能选错，因此正式提交仍推荐使用前面的显式 `--checkpoint` 命令。

## 故障排查

| 问题 | 解决 |
| --- | --- |
| `libgomp: Invalid value for OMP_NUM_THREADS` | `unset OMP_NUM_THREADS; export OMP_NUM_THREADS=$(nproc)` |
| `no kernel image available` | PyTorch 版本过低，需 CUDA 12.6+ |
| OOM | 加 `--set train.batch_size=4 --set train.gradient_accumulation=4` |
| 训练时报 run-id 已存在 | 给 `autodl-run` 换 `--run-id`，或确认后删除旧 run |
| 提交时报 `unrecognized arguments: --run-id` | 删除 `autodl-submit` 后面的 `--run-id`，改用 `--checkpoint` |
| 找不到 `best_ssim.ckpt` | 确认训练已完成，并检查 `outputs/initial_round_v2/<run-id>/checkpoints/` |
| 训练中途停止 | 使用同一配置和 `resume` 命令加载 `last.ckpt`；不要从头覆盖已有 run |

## 这次重训具体改了什么

- 主模型仍是上一轮实际达到 JPG SSIM 0.805660、PSNR 26.04039 的 MultiMarkerRestorer base64；不会因追逐新论文而换成未经验证的大模型。
- 使用连续两阶段 MSE/Charbonnier/SSIM 损失：前期学习稳定像素映射，后期增加 MSE 与 SSIM 对齐，兼顾竞赛的 SSIM 和 PSNR。
- 新增小权重的荧光前景辅助损失，只从官方 train 标签动态产生软掩膜，强调 CD68 高表达区域；不会读取 test 标签或引入外部数据。
- validation 从每轮一次改为每 3 轮一次；Linux 下自动启用最多 8 个 DataLoader worker，减轻上一轮 CPU 解码瓶颈。
- checkpoint 和轻量日志不再为每个 epoch 重复保存 1,292 条逐图记录；最终逐图 CSV 仍完整保留。
- 新配置只保留可续训的 `last.ckpt` 和用于提交的 `best_ssim.ckpt`，不再额外复制 best PSNR、best proxy 和 top-k 权重。

这些改动是有文献依据的候选方案，但在新训练完成前不能声称已经涨分。判断是否提升，要把新 run 的 `validation/metrics.json` 与上一轮固定基线 JPG SSIM `0.805660`、PSNR `26.04039` 比较；最终仍以平台分数为准。

## 论文依据与取舍

- 赛事页面给出的本质是成对 DAPI→marker 图像转换，评分由 SSIM 与归一化 PSNR 组合，因此保留全图恢复损失，并显式加入 MSE/SSIM 后期对齐：[赛事任务与评分说明](https://www.aicomp.cn/tracks/tracks-1/3759.html)。
- 2025 年 Spotlight 工作说明荧光前景监督能改善蛋白阳性结构，但也提示只追前景可能牺牲全图指标；本项目因此只加入 0.06 权重的辅助项，不替换全局损失：[Spotlight / foreground-aware virtual staining](https://pmc.ncbi.nlm.nih.gov/articles/PMC12265572/)。
- 2026 年 PGVMS 强调 protein-aware distribution 与 prototype consistency；本工程原本已有 prototype，所以只借鉴“关注蛋白分布”的监督思想，不重复堆第二套原型网络：[PGVMS](https://arxiv.org/abs/2602.23292)。
- ROSIE 在大规模共染数据上使用直接的 MSE 图像恢复目标，支持把显式 MSE 放回训练目标：[ROSIE, Nature Communications 2025](https://www.nature.com/articles/s41467-025-62346-0)。
- 现有主干采用高效 NAF 风格块，上一轮已真实验证有效；因此没有在唯一一次重训中换成更大的 Transformer：[NAFNet 官方实现](https://github.com/megvii-research/NAFNet)。
- DiffVS 的两阶段 latent diffusion 训练与外部依赖成本更高，不符合本轮单次、七小时内的稳健预算，所以未进入默认路径：[DiffVS 官方实现](https://github.com/hvcl/DiffVS)。

以上是工程化改编，不代表复现了这些论文，也不代表在当前数据上已经获得论文所报告的提升。
