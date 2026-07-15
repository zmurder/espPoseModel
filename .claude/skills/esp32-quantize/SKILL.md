---
name: esp32-quantize
description: 将 PyTorch/ONNX 模型转换为 ESP32 可部署的 .espdl 格式。支持 PTQ 量化（默认、混合精度、层间均衡）和 QAT 量化感知训练。当需要量化模型到 ESP32 部署时使用。
argument-hint: [model_path] [target_platform]
disable-model-invocation: true
---

# ESP32 模型量化与部署转换工具

将自定义模型转换为 ESP32 可部署的 `.espdl` 格式，支持两种量化方式：**PTQ（训练后量化）**和 **QAT（量化感知训练）**。

## 快速开始

### 1. 安装依赖

```bash
pip install esp-ppq torch torchvision onnx
```

### 2. 目标平台选择

| 平台 | ROUND 策略 | target 参数值 |
| :--- | :--------- | :------------ |
| ESP32 | ROUND_HALF_UP | `c`（编译用 `esp32`） |
| ESP32S3 | ROUND_HALF_UP | `esp32s3` |
| ESP32P4 | ROUND_HALF_EVEN | `esp32p4` |

---

## 方式一：PTQ（训练后量化）

PTQ 是在模型训练完成后进行的量化，不需要修改训练代码，是最简单的量化方式。

### 步骤 1：准备模型

**PyTorch 模型** → 参考 [reference_torch.md](reference_torch.md)
**ONNX 模型** → 参考 [reference_onnx.md](reference_onnx.md)

### 步骤 2：准备校准数据集

```python
from torch.utils.data import DataLoader, TensorDataset

# 数据集需要覆盖模型输入的各种情况
# shuffle 必须为 False（计算量化误差时会多次遍历）
dataset = TensorDataset(x_test, y_test)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 定义 collate_fn：TensorDataset 返回 (x, y)，量化只需要 x
def collate_fn(batch):
    return batch[0].to(DEVICE)
```

### 步骤 3：量化并导出

**PyTorch 模型**：
```python
from esp_ppq.api import espdl_quantize_torch

quant_ppq_graph = espdl_quantize_torch(
    model=model,
    espdl_export_file='your_model.espdl',
    calib_dataloader=dataloader,
    calib_steps=32,
    input_shape=[1, 3, 224, 224],
    target='esp32',  # 或 'esp32s3', 'esp32p4', 'c'
    num_of_bits=8,
    collate_fn=collate_fn,
    device='cpu',
    export_test_values=True,
    verbose=1,
)
```

**ONNX 模型**：
```python
from esp_ppq.api import espdl_quantize_onnx

quant_ppq_graph = espdl_quantize_onnx(
    onnx_import_file='model.onnx',
    espdl_export_file='your_model.espdl',
    calib_dataloader=dataloader,
    calib_steps=32,
    input_shape=[1, 3, 224, 224],
    target='esp32',
    num_of_bits=8,
    collate_fn=collate_fn,
    device='cpu',
    export_test_values=True,
    verbose=1,
)
```

### 高级 PTQ 方法

如果默认 8bit 量化精度损失较大，尝试以下方法：

#### 1. 混合精度量化

将误差较大的层使用 int16 量化：

```python
from esp_ppq.api import get_target_platform

dispatching_override = [
    ("/path/to/layer1", get_target_platform(TARGET, 16)),
    ("/path/to/layer2", get_target_platform(TARGET, 16)),
]
```

#### 2. 层间均衡量化

适用于 ReLU6 激活的模型：

```python
from esp_ppq.layers.equalization import layerwise_equalization

# 1. 将 ReLU6 替换为 ReLU
model = convert_relu6_to_relu(model)

# 2. 应用层间均衡
model = layerwise_equalization(model, iterations=4, value_threshold=0.4, opt_level=2)
```

### 量化误差分析

- **累计误差（Graphwise Error）**：最后一层的误差应小于 10%
- **逐层误差（Layerwise Error）**：大部分层应小于 1%

---

## 方式二：QAT（量化感知训练）

当 PTQ 无法达到满意精度时使用。详见 [reference_advanced.md](reference_advanced.md#_4)。

---

## 输出文件

| 文件 | 说明 |
| :--- | :--- |
| `*.espdl` | ESPDL 模型二进制文件，可直接用于芯片推理 |
| `*.info` | 模型文本文件，用于调试 |
| `*.json` | 量化信息文件 |

---

## 完整参考示例

| 文件 | 说明 |
| :--- | :--- |
| [reference_torch.md](reference_torch.md) | PyTorch 模型量化完整示例 - 来自官方 sin_model 教程 |
| [reference_onnx.md](reference_onnx.md) | ONNX 模型量化完整示例 - 包含多框架转换指南 |
| [reference_advanced.md](reference_advanced.md) | 高级量化方法（混合精度、层间均衡、QAT）与模型部署 |
| [reference_pose.md](reference_pose.md) | 姿态检测模型量化参考 - 针对 LitePoseNet 7关键点 |

---

## 常见问题

1. **量化后精度下降太多** → 尝试混合精度量化或 QAT
2. **不同平台的 .espdl 不能混用** → 确保量化 target 和部署 target 匹配
3. **batch_size 必须为 1** → 当前 ESP-DL 仅支持 batch_size=1
4. **其他框架模型** → 需先转换为 ONNX（见 reference_onnx.md）
