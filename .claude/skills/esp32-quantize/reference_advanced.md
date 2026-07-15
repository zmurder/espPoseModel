# 高级量化方法与模型部署

> **版本信息**：`last_updated: 2026-04-15`，`source: ESP-DL how_to_quantize_model.html`

来自 ESP-DL 官方文档的高级量化方法和部署指南。

## 高级量化方法

### 1. 混合精度量化

当默认 8bit 量化精度损失较大时，可以将误差较大的层使用 int16 量化。

**分析方法**：
运行量化后会输出 **Graphwise Error（累计误差）** 和 **Layerwise Error（逐层误差）**：
- 最后一层的累计误差应小于 10%
- 大部分层的逐层误差应小于 1%
- 误差较大的层建议使用更高精度

**量化设置**：
```python
from esp_ppq.api import get_target_platform

TARGET = 'esp32p4'

# 指定某些层使用 16bit 量化
dispatching_override = [
    ("/features/features.1/conv/conv.0/conv.0.0/Conv", get_target_platform(TARGET, 16)),
    ("/features/features.1/conv/conv.0/conv.0.2/Clip", get_target_platform(TARGET, 16)),
]

quant_ppq_graph = espdl_quantize_torch(
    model=model,
    espdl_export_file='your_model.espdl',
    calib_dataloader=dataloader,
    calib_steps=32,
    input_shape=[1, 3, 224, 224],
    target=TARGET,
    num_of_bits=8,
    collate_fn=collate_fn,
    dispatching_override=dispatching_override,
    device=DEVICE,
    export_test_values=True,
    verbose=1,
)
```

### 2. 层间均衡量化

来自论文 [Data-Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721)。

**要求**：将 MobilenetV2 模型中原来的 ReLU6 替换为 ReLU。

```python
import torch.nn as nn

def convert_relu6_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_relu6_to_relu(child)
    return model

# 将 ReLU6 替换为 ReLU
model = convert_relu6_to_relu(model)

# 使用层间均衡
from esp_ppq.layers.equalization import layerwise_equalization
model = layerwise_equalization(
    model,
    iterations=4,
    value_threshold=0.4,
    opt_level=2,
)
```

**量化结果示例（MobileNetV2）**：
- 默认 8bit 量化：Prec@1 60.500%
- 混合精度量化：Prec@1 69.550%
- 层间均衡量化：Prec@1 69.800%
- Float 模型：Prec@1 71.878%

### 3. 算子分裂量化（Horizontal Layer Split）

将某些算子分裂为多个子算子分别量化，可以提高精度。

### 4. 量化感知训练（QAT）

当 PTQ 无法达到满意精度时，在训练过程中模拟量化效果。

参考 [PPQ QAT Example](https://github.com/OpenPPL/ppq/blob/master/ppq/samples/TensorRT/Example_QAT.py)：

```python
from esp_ppq.api import export_qat_torch_to_onnx, QuantizationSettingFactory
from esp_ppq.quantization.quantizer import TensorRTCUPQ

# 1. 加载预训练模型并替换 ReLU6 为 ReLU
model = convert_relu6_to_relu(model)

# 2. 创建 QAT 量化器
TensorRT_quantizer = TensorRTCUPQ(
    graph=model,
    input_shapes=input_shapes,
    quant_min=MIN_INT,
    quant_max=MAX_INT,
)

# 3. 开始 QAT 训练（使用你的训练数据）

# 4. 导出 QAT 模型
export_qat_torch_to_onnx(
    model=model,
    graph=graph,
    save_path='qat_model.onnx',
    input_shapes=input_shapes,
)

# 5. 使用 ONNX 量化流程
```

---

## 模型部署

部署到 ESP32 芯片时，需要进行图像预处理和后处理。

### 前处理（ImagePreprocessor）

ESP-DL 提供了 `ImagePreprocessor` 类，封装的图像处理流程包括：
- Color conversion
- Crop
- Resize
- Normalization
- Quantize

**参考文件**：
- `esp-dl/vision/image/dl_image_preprocessor.hpp`
- `esp-dl/vision/image/dl_image_preprocessor.cpp`

### 后处理

根据任务类型（分类、检测、分割等）进行对应的后处理。

**图像分类后处理参考文件**：
- `esp-dl/vision/classification/dl_cls_postprocessor.hpp`
- `esp-dl/vision/classification/dl_cls_postprocessor.cpp`
- `esp-dl/vision/classification/imagenet_cls_postprocessor.hpp`
- `esp-dl/vision/classification/imagenet_cls_postprocessor.cpp`

### 部署检查清单

1. **.espdl 文件**：确保量化 target 和部署 target 匹配
   - ESP32 使用 ROUND_HALF_UP
   - ESP32S3 使用 ROUND_HALF_UP
   - ESP32P4 使用 ROUND_HALF_EVEN

2. **batch_size**：当前 ESP-DL 仅支持 batch_size=1

3. **export_test_values**：启用后可在 .info 文件中查看测试输入/输出，用于板端验证

4. **不同平台的 .espdl 不能混用**：量化时选择的 target 必须与部署平台一致
