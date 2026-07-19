# 模型设计

## 任务定义

给定 DAPI 图像 `x∈[0,1]^(C×256×256)`，学习确定性映射，为 HLA-DR、CD45RO、Vimentin 和 CD68 分别预测与输入空间一致的图像。模型没有随机 latent、采样循环或判别器，因此同一 checkpoint、输入和配置产生相同输出。

## Residual U-Net 基线

基线由四级 encoder-decoder 组成。每级使用 GroupNorm、SiLU 和两层 residual convolution；下采样使用 stride convolution，解码使用 bilinear interpolation 后接 convolution，避免反卷积棋盘格。skip feature 在通道维拼接。输出经 sigmoid 限制到 `[0,1]`。它用于验证 I/O、训练、指标和提交链路，也是 ensemble 的独立成员。

## MultiMarkerRestorer

输入可拼接固定 Sobel x/y 产生的梯度幅值。共享 encoder 采用四尺度 NAF 风格块：1×1 扩张、depthwise 3×3、simple gate、简化 channel attention、1×1 投影和可学习 residual scaling。默认逻辑通道为 `[base,2×base,4×base,8×base]`。

bottleneck 特征先做 L2 normalize，再分别与共享原型 `P_shared∈R^(Ks×C)` 和任务原型 `P_task∈R^(Kt×C)` 计算 cosine similarity：

```text
A = softmax((normalize(F) · normalize(P)^T) / temperature)
F_proto = projection(A · P)
F_out = F + alpha · F_proto
```

共享 decoder 不复制四套 encoder。每个 decoder stage 具有轻量任务 adapter：depthwise 3×3、GELU、pointwise 1×1 和 residual。任务 embedding 生成 FiLM scale/shift。每个任务具有独立 head，并在 64、128、256 三个尺度输出深监督。传入 `task_name` 时只计算指定任务的 adapter/head。

## 通道语义

`ImageSpec` 分开记录文件 storage mode 和模型 logical channels。如果审计证明 RGB 三通道近似相同，模型可使用一个逻辑通道；writer 在保存时仍恢复目标原始 RGB mode。该转换必须记录在审计与 checkpoint 中。

## 损失

每个任务的基础损失为：

```text
L_task = 0.40 L_charb + 0.35 (1-SSIM) + 0.10 (1-MS-SSIM)
       + 0.10 L_grad + 0.05 L_freq
```

Charbonnier 使用 `sqrt(error²+epsilon²)`。结构权重从目标梯度/局部变化得到，不假定亮像素为阳性。梯度损失比较 Sobel x/y；频率损失比较 `log1p(abs(rfft2(.)))`，不直接惩罚相位。深监督权重为 1.0、0.5、0.25。

多任务时可增加预测与 GT 任务相关矩阵差异 `L_corr`。原型 activation loss 鼓励 feature 接近至少一个原型，diversity loss 抑制不同原型余弦相似。它们默认权重很小且可完全关闭：

```text
L_total = mean(L_task) + λcorr Lcorr + λact Lproto_act + λdiv Lproto_div
```

AMP 下 SSIM、FFT、指标和最终 reduction 使用 float32，避免低精度数值溢出。

## 参数量与 FLOPs

运行时记录 trainable parameters，并以 Conv2d/Linear forward hooks 估算固定 256×256 输入下的乘加量。FLOPs 是工程估计值，必须与计算口径一起报告，不能与其他工具的不同口径直接比较。

## 确定性和限制

随机种子覆盖 Python、NumPy、Torch、CUDA 和 DataLoader。可配置 deterministic algorithms。模型不会对测试图进行 blur、resize、配准或颜色后处理。小样本、未知真实 ROI 和可能存在的组织差异仍会限制泛化。

