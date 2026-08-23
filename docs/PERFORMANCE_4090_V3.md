# RTX 4090：9 小时内的 CD68 性能升级指南

## 先读结论

`configs/initial_round_cd68_max_v3.yaml` 是新的**候选配置**，不是已经证明涨分
的默认答案。旧的 `initial_round_cd68_retrain_v2.yaml` 保留不动，随时可以回退。

最新真实日志给出的固定参照是：

| 模型与推理 | JPG SSIM | JPG PSNR |
|---|---:|---:|
| retrain-v2，EMA，无 TTA | 0.804480 | 25.97599 |
| retrain-v2，EMA，D4 | **0.809280** | **26.19677** |

84 epoch 的总命令耗时为 3 小时 25 分，纯训练 2 小时 45 分，峰值显存
11.63 GiB。最佳 checkpoint 出现在约第 60 epoch；之后训练 loss 继续下降，
但验证 SSIM/PSNR 回落。因此 v3 没有换成更大的 Transformer 或无脑增加
epoch，而是把计算分配给互补的卷积、频率和真实 ROI 邻域路径。

## v3 改了什么

所有改动都有 feature flag，关闭后仍走旧路径。

1. **原有 NAF U 形主干**：继续负责 DAPI→CD68 的主体像素映射，不推翻已经
   跑通的 16.23M 参数基线。
2. **轻量 Detail U 分支**：新增四尺度 `(16,24,32,48)` 大核 depthwise U-Net，
   只向主干的 1×、1/2×、1/4× skip 注入细节，1/8×留给频率分支。三个融合
   投影严格零初始化。该支路实测只增加约 7.2 万参数、0.59 GMAC。
3. **空间—频率并行瓶颈**：1/8×特征同时经过局部 NAF 路径和 float32 FFT
   低/高频路径，随后用逐通道 softmax gate 融合，再以零初始化 residual scale
   接回主干。FFT 强制 float32，其他卷积仍使用 BF16。
4. **真实 3×3 ROI 上下文**：官方 `ROI_row_col` 网格已通过 ROI、相邻 patch
   和方向审计；一个窄 context encoder 编码中心及邻居 DAPI，在 1/4、1/8
   尺度用 FiLM，并在 bottleneck 使用仅面对 9 个 token 的轻量 cross-attention。
   若换数据后审计失败，程序会拒绝训练，不会猜坐标。
5. **稳定的 base/detail 输出与原型**：保留旧 direct output logits，并只叠加
   零初始化的 coarse 与有界 detail 残差；因此启用后的第 0 步预测与旧输出头
   逐位一致，同时两个残差投影首个 backward 都能收到梯度。task adapter、
   Sobel 输入和 cosine prototype 也保持启用。
6. **最终输出专用浮点指标代理损失**（配置字段仍叫 `competition_proxy`）：训练后半段才启用，SSIM 使用与
   scikit-image 验证一致的 uniform 7×7、sample covariance、valid crop；
   PSNR 先逐图求 MSE 再取 capped log-MSE。它只约束最终 256×256 输出，
   不污染深监督小图。它没有、也不能在反向传播中模拟 uint8 四舍五入和
   JPEG 编解码；真正是否提高仍由每次验证保存后的 JPG round-trip 指标决定。
7. **结构损失全开**：原有 MSE/Charbonnier/SSIM/MS-SSIM/gradient/statistics
   之外，以小权重启用 Laplacian pyramid 和 log-frequency loss。
8. **D4 等变一致性**：25% 的 batch 额外抽一个旋转/翻转，比较
   `inverse(f(T(x)))` 与停止梯度的 `f(x)`。它针对当前 D4 推理稳定涨分所
   暴露的方向不一致，推理时不增加模型参数。
9. **末段温和难例采样**：最后 22% epoch 才把约 30% 样本槽位分给最高
   activity 桶；总样本数不变。训练标签只来自 official train，不读取 test。
10. **深监督退火**：从 `[1, 0.5, 0.25]` 平滑变为 `[1, 0.1, 0.05]`，后期
   更关注最终分辨率。
11. **AdamW 参数分组**：卷积/线性权重继续 weight decay；bias、归一化、
   NAF beta/gamma 和残差 scale 不衰减。CUDA 上优先 fused AdamW，不支持时
   自动回退。每 3 epoch 同时比较 raw 与 EMA，以 JPG 指标选 checkpoint，
   不把旧模型的 EMA 结论强加给新支路。
12. **4090 吞吐**：BF16、batch 12、accumulation 1、channels-last、TF32、
   cuDNN benchmark、8 workers、prefetch 4。没有默认开启尚未充分验证的
   `torch.compile`。
13. **真实 GPU 日志**：每 2 秒记录 GPU utilization、显存、功耗和温度，汇总
   到 `log/<run-id>/epoch_metrics.jsonl`。
14. **时间保护**：`max_wall_time_hours: 6.5` 会保守估算到下一个可验证停点的
   时间；只在当前 epoch 已验证且 best/last checkpoint 完整保存后停止，为
   最终 D4 验证预留时间。

全开组合的静态实测为 **18,074,199 参数、约 44.11 GMAC/张**；相对上一版
约增加 11.3% 参数和 12.6% 可计数 MAC，并不是再并联一套 42G 的完整模型。
“并行”指网络中并存的多路特征计算和 GPU 批量 kernel；程序不手写 CUDA
streams，因为单卡 autograd 下强行并发通常反而增加同步和显存开销。

### “所有模块都开”的准确含义

本候选已经开启所有与**单目标、已对齐、真实 ROI 网格**相容的模块：NAF、
轻量 U、FFT、ROI context、轻量 cross-attention、prototype、base/detail、
深监督、结构/频率损失、浮点指标代理、D4 等变、难例采样、EMA 和推理 D4。

下面几项故意关闭，不是漏开：foreground loss 在真实日志中没有带来提升；
shift-tolerant loss 与实测零偏移相冲突；correlation/FAMO 只适用于多目标；
DAPI-MAE 需要额外预训练；完整 Restormer/Mamba/扩散/GAN 与现有路径重复且会
显著增加时间或优化风险；`torch.compile` 尚未完成跨 eager checkpoint 回归。

## 第一次推荐只跑一个 seed

上传项目后：

```bash
cd /root/autodl-tmp/project
conda activate MEDICAL
export PYTHONPATH="$PWD/src:$PYTHONPATH"
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m virtual_staining.cli --log-root log autodl-run \
  --config configs/initial_round_cd68_max_v3.yaml \
  --data-root AUTO \
  --target CD68 \
  --run-id cd68_max_v3_seed2026
```

旧 v2 的 4090 D 实测是 3 小时 25 分；v3 全模块版还没有在 4090 上完成正式
训练，按计算量和额外邻域解码保守预计整条命令约 4～8.5 小时，不能把这个
估算冒充实测。训练器的 6.5 小时安全停点保护会为前后数据审计、最终验证和
D4 留出余量；它是预测保护而不是操作系统硬截止，第一次看到的 epoch 吞吐
才是可信依据。

终端有 batch 进度条。轻量日志在：

```text
log/cd68_max_v3_seed2026/cd68_max_v3_seed2026/
```

训练权重和完整验证在：

```text
outputs/initial_round_max_v3/cd68_max_v3_seed2026/
```

后续交给我分析时，优先下载整个轻量日志目录。`module_inventory.json` 会记录
每个顶层模块的参数量和全部 feature flag，`model_stats.json` 记录总参数/MAC，
`epoch_metrics.jsonl` 记录各损失分量、吞吐、显存、GPU 利用率、功耗和温度，
`validation_per_image*.csv` 保存逐图与 ROI 分层指标；不需要下载巨大 checkpoint
我也能先判断数据/优化/吞吐问题。

## 怎样判断是否真的提升

不要看 train loss，也不要看 test 图片“似乎更漂亮”。读取同一个 ROI split、
同一个 JPG round-trip、同一个 D4 协议：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/initial_round_max_v3/cd68_max_v3_seed2026/validation/metrics.json")
metrics = json.loads(path.read_text())
jpg = metrics["tta_comparison"]["d4"]["domains"]["jpg"]["macro"]
print("D4 JPG SSIM =", jpg["mean_ssim"])
print("D4 JPG PSNR =", jpg["mean_psnr"])
print("旧参照       = 0.809280 / 26.19677")
PY
```

只有新 run 的 D4 JPG 指标高于参照、另一项没有明显下降，并且 5 个 ROI、
high/mid/low activity、border/interior 没有集中退化，才能说验证性能提升。
official test 没有标签，最终成绩仍以平台为准。

查看 GPU 是否真正工作：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("log/cd68_max_v3_seed2026/cd68_max_v3_seed2026/epoch_metrics.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
for row in rows[-5:]:
    train = row.get("train", row)
    print(
        "epoch", train.get("epoch"),
        "img/s", train.get("images_per_second"),
        "GPU%", train.get("gpu_monitor/gpu_util_percent_mean"),
        "VRAM MiB", train.get("gpu_monitor/memory_used_mib_max"),
        "power W", train.get("gpu_monitor/power_w_mean"),
    )
PY
```

GPU 利用率偶尔下降很正常；重点看多个 epoch 的均值、吞吐和显存，而不是
AutoDL 网页上某一秒的曲线。若 OOM，命令会自动微批降级；仍失败时使用：

```bash
--set train.batch_size=8 --set train.gradient_accumulation=2
```

## 单模型提交

若 v3 验证通过：

```bash
python -m virtual_staining.cli --log-root log autodl-submit \
  --config configs/initial_round_cd68_max_v3.yaml \
  --data-root AUTO \
  --target CD68 \
  --checkpoint outputs/initial_round_max_v3/cd68_max_v3_seed2026/checkpoints/best_ssim.ckpt
```

上传：

```text
outputs/initial_round_max_v3/cd68_max_v3_seed2026/submission/submission_CD68.zip
```

## 时间允许时再跑第二个 seed

第二个 seed 最稳妥的做法是另开一次租用；只有第一个**整条命令**不超过约
3.5 小时且当前租用剩余时间明确足够，才在同一个 9 小时窗口继续。数据
`split_seed=2026` 固定，因此两个模型使用完全相同的 train/val ROI；模型、
增强和难例采样随机序列则一同改为 3407。

```bash
python -m virtual_staining.cli --log-root log autodl-run \
  --config configs/initial_round_cd68_max_v3.yaml \
  --data-root AUTO \
  --target CD68 \
  --set project.seed=3407 \
  --set data.activity_sampler.seed=3407 \
  --run-id cd68_max_v3_seed3407
```

先分别查看两个 run 的 D4 JPG 指标。两个成员都正常后，再做**均匀预测集成**；
每个成员先做 D4，随后平均 float tensor，最后只编码一次 JPEG：

```bash
python -m virtual_staining.cli --log-root log autodl-ensemble-submit \
  --config configs/initial_round_cd68_max_v3.yaml \
  --data-root AUTO \
  --target CD68 \
  --checkpoints \
    outputs/initial_round_max_v3/cd68_max_v3_seed2026/checkpoints/best_ssim.ckpt \
    outputs/initial_round_max_v3/cd68_max_v3_seed3407/checkpoints/best_ssim.ckpt \
  --output-dir outputs/initial_round_max_v3/ensemble_seed2026_seed3407
```

最终 ZIP：

```text
outputs/initial_round_max_v3/ensemble_seed2026_seed3407/submission/submission_CD68.zip
```

不要用 test 标签学习 ensemble 权重。若以后要使用 `--validation-scores`，分数
必须来自完全相同的本地 val manifest；第一次优先使用默认 1:1 平均。

## 文献依据与未采用路线

- [ConvIR，IEEE TPAMI 2024](https://github.com/c-yn/ConvIR)表明高效卷积
  encoder-decoder 可以在多类复原任务上与 Transformer 竞争；
  [OKNet，AAAI 2024](https://github.com/c-yn/OKNet)采用局部/大核/全局并行
  思路。项目只 clean-room 借鉴窄通道大核细节路径，没有复制其完整网络。
- [ACL，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gu_ACL_Activating_Capability_of_Linear_Attention_for_Image_Restoration_CVPR_2025_paper.html)
  同样把高效全局建模与多尺度卷积局部增强分开；本工程已有 9-token context
  attention，因此没有再堆整套 linear-attention U-Net，而只补轻量局部 U 分支。
- [AdaIR，ICLR 2025](https://github.com/c-yn/AdaIR)从低/高频挖掘和调制说明
  空间域之外的频率结构有用；本项目实现的是更小的单任务 FFT 分支，不声称
  复现 AdaIR。
- [BioIR，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/73ba81c7b25134a559c8a9c39ec1a4c3-Abstract-Conference.html)
  将大范围上下文与细粒度路径区分处理，支持本项目“ROI/频率上下文 + 轻量
  detail U”的分工；没有照搬其显存更高的 dynamic filtering。
- [EQ-Reg，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Bai_A_Regularization-Guided_Equivariant_Approach_for_Image_Restoration_CVPR_2025_paper.pdf)
  及[作者代码](https://github.com/yulu919/EQ-REG)支持用训练期等变正则改善复原网络；
  本项目只做低风险的输出级 D4 consistency，并不声称复现整篇论文。
- [PSPStain，MICCAI 2024](https://papers.miccai.org/miccai-2024/595-Paper2078.html)
  及[官方代码](https://github.com/ccitachi/PSPStain)使用 protein-aware 与 prototype
  学习；本工程原本已有 prototype，故没有重复堆一套 GAN/原型网络。
- [PGVMS，TMI 2026](https://github.com/ccitachi/PGVMS)进一步研究多标记提示学习，
  但其 H&E、CONCH 和生成式依赖与本赛题 DAPI→荧光及 9 小时预算不完全匹配。
- [NTIRE 2025 图像去噪挑战报告](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Sun_The_Tenth_NTIRE_2025_Image_Denoising_Challenge_Report_CVPRW_2025_paper.pdf)
  包含 hard-data-mining fine-tuning 和旋转/翻转 self-ensemble 的高分实践。
- [MambaIRv2，CVPR 2025](https://github.com/csguoh/MambaIR)需要额外 CUDA 扩展，
  完整替换风险较大；[DiffStain，MICCAI 2025](https://papers.miccai.org/miccai-2025/0235-Paper2432.html)
  的扩散推理也不适合本轮严格时间与像素指标目标，所以都未设为默认。

这些论文只提供设计依据。v3 的真实提升必须由你的 AutoDL 日志和同协议验证
证明；在完整训练前，任何人都不应承诺“必涨”或伪造分数。
