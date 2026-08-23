# AutoDL 运行指南

> 本文档是 AutoDL 上运行 CAMP-VS 项目的唯一入口。按顺序执行下面两条命令即可完成训练、验证和竞赛提交打包。

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
  --config configs/initial_round_cd68.yaml \
  --data-root AUTO \
  --target CD68 \
  --run-id cd68_autodl_seed2026
```

**这条命令自动完成：**
1. 环境检测（CUDA/GPU/PyTorch 版本）→ `artifacts/environment.json`
2. 数据发现与 manifest 构建（train/val/test 自动划分）→ `artifacts/data_discovery.json`
3. 数据审计（ROI 对齐、泄漏检查）→ `artifacts/data_audit.json`
4. 训练 120 epoch（CAMP-VS v2 模型：base64 NAF 编码器 + Restormer-lite 6/6 mixer）
5. 验证 best checkpoint（raw+ema，jpg 主域，三域指标 + ROI 分组）→ `validation/metrics.json`
6. 写入汇总报告 → `outputs/initial_round/<run-id>/pipeline_report.json`

**预期耗时：** 约 6-10 小时（视 GPU 型号而定，4090 约 6 小时）。

**快速验证（可选）：** 加 `--max-epochs 3` 跑 3 epoch 确认流程通畅，再跑完整训练。

### 查看验证性能

```bash
python -c "
import json
m = json.load(open('outputs/initial_round/cd68_autodl_seed2026/validation/metrics.json'))
jpg = m['domains']['jpg']['macro']
print('JPG SSIM:', round(jpg['mean_ssim'], 4))
print('JPG PSNR:', round(jpg['mean_psnr'], 2))
print('ROI SSIM:', round(m['domains']['jpg']['macro']['roi_ssim'], 4))
"
```

关键文件：

| 文件 | 内容 |
| --- | --- |
| `outputs/initial_round/<run-id>/validation/metrics.json` | 完整验证指标（float/uint8/jpg 三域，ROI 分组，分层） |
| `outputs/initial_round/<run-id>/validation/per_image.csv` | 逐图 SSIM/PSNR |
| `outputs/initial_round/<run-id>/pipeline_report.json` | 汇总报告（环境+训练+验证） |
| `outputs/initial_round/<run-id>/checkpoints/best_ssim.ckpt` | 最佳 checkpoint |

> **重要：** val SSIM/PSNR 是选择 checkpoint 的代理证据，**不是** official test 分数。official 分数只能由上传 ZIP 到竞赛平台后获得。

## 命令 2：打包竞赛提交（一条命令完成）

训练完成后，推荐显式指定本次训练生成的最佳 checkpoint：

```bash
python -m virtual_staining.cli --log-root log autodl-submit \
  --config configs/initial_round_cd68.yaml \
  --data-root AUTO \
  --target CD68 \
  --checkpoint outputs/initial_round/cd68_autodl_seed2026/checkpoints/best_ssim.ckpt
```

> **不要给 `autodl-submit` 添加 `--run-id`。** 当前 CLI 只有训练命令
> `autodl-run` 支持 `--run-id`；提交命令使用 `--checkpoint` 确定要提交哪个
> run 的模型。错误地添加 `--run-id` 会得到
> `unrecognized arguments: --run-id ...`。

**这条命令自动完成：**
1. 加载明确指定的 `best_ssim.ckpt`，避免存在多个 run 时选错模型
2. 对官方 test 集（1346 张 DAPI）推理，并按 checkpoint/effective config 使用已选择的 raw/EMA 权重和 TTA 设置 → `predictions/`
3. 构建竞赛要求的 `results/` 目录结构与 ZIP → `submission/`
4. 严格校验 ZIP（文件名、数量、尺寸、模式）

### 上传竞赛平台

最终 ZIP 在：

```
outputs/initial_round/cd68_autodl_seed2026/submission/submission_CD68.zip
```

将此 ZIP 上传到竞赛平台，平台返回的分数才是 official test score。

## 可选：仅有一个训练 run 时自动选择 checkpoint

如果 `outputs/` 下确定只有一个可用的 `best_ssim.ckpt`，可以省略
`--checkpoint`：

```bash
python -m virtual_staining.cli --log-root log autodl-submit \
  --config configs/initial_round_cd68.yaml \
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
| 找不到 `best_ssim.ckpt` | 确认训练已完成，并检查 `outputs/initial_round/<run-id>/checkpoints/` |
| 训练中途停止 | 检查 `pipeline_report.json` 的 `epochs_completed` 字段（早停 patience=30） |
