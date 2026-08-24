# CAMP-VS v4：AutoDL 训练、续训与提交完整指南

本指南适用于 AutoDL 的 Linux 终端，默认：

- 项目目录：`/root/autodl-tmp/project`
- Conda 环境：`MEDICAL`
- 官方数据：`/root/autodl-tmp/project/dataset/official`
- 训练目标：`CD68`
- 当前正式配置：`configs/initial_round_cd68_max_v4.yaml`
- 正式 run ID：`cd68_max_v4_seed2026`

如果你的项目目录不同，只修改命令中的 `cd /root/autodl-tmp/project`。

> 注意：本文件代码块中的命令可以直接复制。变量名和文件名中的下划线不能写成 `\_`，命令换行符 `\` 后面也不能再有空格。

## 一、每次打开新终端都要做的准备

如果终端开头已经显示 `(MEDICAL)`，前两行可以跳过；其余设置建议每次打开新终端都重新执行，因为 `export` 只对当前终端会话有效。

~~~bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate MEDICAL

cd /root/autodl-tmp/project

export AIC_DATA_ROOT="$PWD/dataset/official"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
~~~

这些变量的含义：

- `AIC_DATA_ROOT`：告诉程序官方 `train/` 和 `test/` 数据在哪里；
- `PYTHONPATH`：让 Python 直接找到 `src/virtual_staining`，即使没有执行 `pip install -e .` 也能运行；
- `CUDA_VISIBLE_DEVICES=0`：使用第 0 张 GPU；
- `PYTHONUNBUFFERED=1`：让日志立即显示，避免长时间缓存后才输出。

检查 PyTorch、CUDA、GPU 和数据路径：

~~~bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python -c "import os; print('AIC_DATA_ROOT=', os.environ.get('AIC_DATA_ROOT'))"

ls "$AIC_DATA_ROOT/train"
ls "$AIC_DATA_ROOT/test"
~~~

必须看到 `CUDA: True`。如果是 `False`，不要开始 100 轮训练，应先解决 CUDA 版 PyTorch 的安装问题。

## 二、推荐使用 tmux，避免网页断开导致训练停止

新建一个名为 `campvs` 的 tmux 会话：

~~~bash
tmux new -s campvs
~~~

进入 tmux 后，再执行一次环境准备：

~~~bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate MEDICAL

cd /root/autodl-tmp/project

export AIC_DATA_ROOT="$PWD/dataset/official"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
~~~

暂时离开 tmux、让程序继续运行：

```text
按 Ctrl+B，松开后再按 D
```

以后重新进入：

~~~bash
tmux attach -t campvs
~~~

查看已有 tmux 会话：

~~~bash
tmux ls
~~~

## 三、启动 v4 正式训练

在项目根目录、MEDICAL 环境和已经设置好 `export` 的终端中执行：

~~~bash
python -m virtual_staining.cli \
  --log-root log \
  autodl-run \
  --config configs/initial_round_cd68_max_v4.yaml \
  --data-root AUTO \
  --target CD68 \
  --run-id cd68_max_v4_seed2026
~~~

正常情况下会依次进行：

1. 环境与 CUDA 信息记录；
2. 自动发现 `dataset/official`；
3. 构建 train、val、test manifest；
4. 数据与 ROI 邻域审计；
5. 构造 v4 重型三路模型；
6. 训练、保存 checkpoint；
7. raw/EMA 的 float、uint8、JPG 三域验证；
8. 训练结束后的普通和 D4 验证；
9. 汇总可下载日志。

训练进度会显示为：

```text
Train 1/100
Train 2/100
...
```

当前正式配置为：

- 最多 100 epoch；
- 每 5 epoch 验证一次；
- batch size 6；
- gradient accumulation 2；
- effective batch 12；
- raw 和 EMA 分别验证；
- float、uint8、JPG 三域验证；
- 训练期随机 D4 等变；
- 提交期完整 D4 八次推理平均。

`100 epoch` 是最大轮数，不保证一定跑满。以下情况会正常提前结束：

- 连续 5 个验证事件没有改进，触发 early stopping；
- 程序预测继续训练会超过 8.5 小时训练预算；
- 用户主动按 `Ctrl+C` 中断。

前两种情况会保存 checkpoint 并在日志中写明 `stop_reason`，不属于程序报错。

## 四、训练过程中查看 GPU 和日志

另开一个 AutoDL 终端，执行环境准备后查看 GPU：

~~~bash
watch -n 2 nvidia-smi
~~~

按 `Ctrl+C` 退出 `watch`，不会影响另一个 tmux 中的训练。

查看当前 run 的日志文件：

~~~bash
find log/cd68_max_v4_seed2026 -maxdepth 3 -type f
~~~

持续查看主日志末尾：

~~~bash
tail -f log/cd68_max_v4_seed2026/command.log
~~~

如果实际日志文件名不是 `command.log`，先运行前面的 `find`，再对实际 `.log` 文件使用 `tail -f`。

## 五、训练被中断后继续训练

断电、机器重启、关闭实例或主动中断后，不要删除：

```text
outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/last.ckpt
```

先确认最新版断点恢复修复已经同步到 AutoDL：

~~~bash
grep -n "_cpu_byte_rng_state" src/virtual_staining/engine/checkpoint.py
~~~

能够显示代码行才说明当前版本已经修复 CUDA `map_location` 导致的 `RNG state must be a torch.ByteTensor` 问题。

然后执行：

~~~bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate MEDICAL

cd /root/autodl-tmp/project

export AIC_DATA_ROOT="$PWD/dataset/official"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python -m virtual_staining.cli \
  --log-root log \
  resume \
  --config configs/initial_round_cd68_max_v4.yaml \
  --data-root AUTO \
  --checkpoint outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/last.ckpt
~~~

续训会恢复：

- 模型参数；
- AdamW 优化器；
- 学习率调度器；
- AMP scaler；
- EMA；
- 已完成 epoch 和 global step；
- Python、NumPy、CPU/CUDA RNG；
- DataLoader generator 状态；
- 历史最佳指标和 early-stopping 计数。

它不是从头重新训练。

## 六、训练结束后确认最佳权重

检查最佳 checkpoint：

~~~bash
ls -lh outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/best_ssim.ckpt
~~~

也可以检查最后 checkpoint：

~~~bash
ls -lh outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/last.ckpt
~~~

提交优先使用 `best_ssim.ckpt`，不要默认拿 `last.ckpt` 提交。

## 七、执行 D4 推理并生成提交 ZIP

训练结束后执行：

~~~bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate MEDICAL

cd /root/autodl-tmp/project

export AIC_DATA_ROOT="$PWD/dataset/official"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python -m virtual_staining.cli \
  --log-root log \
  autodl-submit \
  --config configs/initial_round_cd68_max_v4.yaml \
  --data-root AUTO \
  --target CD68 \
  --checkpoint outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/best_ssim.ckpt
~~~

该命令会自动完成：

1. 读取官方 test DAPI；
2. 加载验证阶段选出的 raw 或 EMA 权重；
3. 默认执行 D4 八次推理并平均；
4. 生成 `<stem>_fake.jpg`；
5. 构造 `results/test/CD68/`；
6. 生成提交 ZIP；
7. 校验预测数量、文件名、尺寸、颜色模式、ZIP 层级和输入对应关系。

官方 test 只有 DAPI，没有 CD68 真值，因此本地不能计算 test SSIM/PSNR。最终 test 分数只能在竞赛平台上传 ZIP 后得到。

## 八、最终需要下载和上传的文件

最终提交文件位于：

```text
outputs/initial_round_max_v4/cd68_max_v4_seed2026/submission/submission_CD68.zip
```

检查文件：

~~~bash
ls -lh outputs/initial_round_max_v4/cd68_max_v4_seed2026/submission/submission_CD68.zip
~~~

查看 ZIP 前几项：

~~~bash
unzip -l outputs/initial_round_max_v4/cd68_max_v4_seed2026/submission/submission_CD68.zip | head -n 20
~~~

ZIP 内部应类似：

```text
results/test/CD68/ROI025_00_00_fake.jpg
results/test/CD68/ROI025_00_01_fake.jpg
...
```

直接下载并上传下面这个 ZIP：

```text
outputs/initial_round_max_v4/cd68_max_v4_seed2026/submission/submission_CD68.zip
```

不要解压后重新压缩，也不要手工给 ZIP 再套一层文件夹，否则可能破坏竞赛要求的 `results/` 首层目录。

## 九、建议同时下载的训练日志

为了后续分析性能和继续改进，建议同时下载：

```text
log/cd68_max_v4_seed2026/
outputs/initial_round_max_v4/cd68_max_v4_seed2026/effective_config.yaml
outputs/initial_round_max_v4/cd68_max_v4_seed2026/metrics.json
outputs/initial_round_max_v4/cd68_max_v4_seed2026/validation/
outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/best_ssim.ckpt
```

如果只想让我分析日志而不传大权重，优先下载 `log/cd68_max_v4_seed2026/` 中自动生成的日志 ZIP，以及 validation 指标 JSON/CSV。

## 十、最短命令速查

训练：

~~~bash
python -m virtual_staining.cli --log-root log autodl-run --config configs/initial_round_cd68_max_v4.yaml --data-root AUTO --target CD68 --run-id cd68_max_v4_seed2026
~~~

续训：

~~~bash
python -m virtual_staining.cli --log-root log resume --config configs/initial_round_cd68_max_v4.yaml --data-root AUTO --checkpoint outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/last.ckpt
~~~

生成并校验提交 ZIP：

~~~bash
python -m virtual_staining.cli --log-root log autodl-submit --config configs/initial_round_cd68_max_v4.yaml --data-root AUTO --target CD68 --checkpoint outputs/initial_round_max_v4/cd68_max_v4_seed2026/checkpoints/best_ssim.ckpt
~~~
