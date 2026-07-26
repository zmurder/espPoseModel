# 模型提速笔记（ESP32 推理 10s → ~1.4s）

> 记录把 ESP32-S3 推理从 **~10 秒降到 ~1.4 秒**的三条核心改动，
> 以及后续修改的**红线**（不能碰，否则退回 10s）。

## 三条提速建议（已全部实施）

### 1. HardSwish → ReLU（激活硬件加速）
- **原因**：HardSwish 在 ESP-DL 上效率低（可能软件实现 / quant-dequant 反复）；ReLU 有 ESP-NN AI 指令硬件加速。
- **实施**：`InvertedResidual` + `PoseNet` 所有激活用 `nn.ReLU`（`model.py` Hardswish 出现 0 次）。
- **注意**：算子表显示 HardSwish int8 也支持，但 ReLU 更简单 + 硬件加速更好。

### 2. expand_ratio 4 → 2（精简通道膨胀）
- **原因**：expand=4 让 InvertedResidual 中间通道膨胀 4 倍（如 32→128），FLOPs 大。
- **实施**：`InvertedResidual` 默认 `expand_ratio=2`；通道 stem24/s1_24/s2_40/s3_72/s4_96（最大 96 < 128）。
- **效果**：expand 部分 FLOPs 减半。

### 3. 砍 ConvTranspose FPN → Resize 上采样（最关键）
- **原因**：ConvTranspose 在 ESP-DL = **InsertZeros + Conv**（插 0 把特征图扩张成超大中间张量，ESP32 SRAM 装不下溢出 PSRAM）——**这是 10s 慢的主因**。
- **实施**：`FPNUp` 用 `F.interpolate(mode='nearest')` 替代 ConvTranspose（零中间张量）。
- **算子**：Resize int8 支持（nearest），无 InsertZeros 灾难。

## 当前模型状态

| 项 | 值 |
|---|---|
| 参数量 | 0.13M（旧 4 点 0.9M 的 1/7） |
| FLOPs | ~388-514M MACs（方案1 head 普通 Conv 后） |
| 算子 | Conv / ReLU / Resize / Sigmoid / Add（全 int8） |
| ESP32 推理 | 1.38s（方案1 后约 1.5-1.8s，<3s 达标）|
| 无 | ConvTranspose / Hardswish |

## 🔴 红线（后续修改不能碰，否则退回 10s）

| ❌ 不能加回 | 原因 |
|---|---|
| ConvTranspose | InsertZeros 大张量 → PSRAM 溢出 → 10s |
| HardSwish | 软件实现，慢 |
| expand_ratio=4 | FLOPs 翻倍 |

| ✅ 只能用 | |
|---|---|
| 上采样 | Resize(nearest) 或 PixelShuffle(DepthToSpace) |
| 激活 | ReLU / ReLU6 |
| depthwise | backbone InvertedResidual 可用（量化误差可接受）；**head 输出层用普通 Conv**（depthwise 量化压峰值，见 quant_analysis.md）|

## 速度 vs 精度权衡

- **提速**靠：ReLU + expand2 + Resize（backbone/FPN，10s→1.4s）。
- **量化 conf**靠：sigma 宽峰 + head 普通 Conv（方案1，解决 int8 峰值崩）。
- 两者不冲突：backbone 提速，head 量化友好，各管一摊。

## 附：为什么旧 4 点模型 10s

旧模型用 ConvTranspose（3 个，FPN）+ HardSwish + expand=4 + 0.9M：
- ConvTranspose InsertZeros 产生超大中间张量溢出 PSRAM（主因）。
- HardSwish 软件实现慢。
- expand=4 + 0.9M FLOPs 大（1.78G MACs）。
重构后（ReLU + expand2 + Resize + 0.13M = 388M MACs）→ 1.38s。
