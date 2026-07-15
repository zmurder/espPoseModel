# ONNX 模型量化参考

> **版本信息**：`last_updated: 2026-04-15`，`source: ESP-DL how_to_quantize_model.html`

完整的 ONNX 模型量化示例，来自 ESP-DL 官方教程。

## 完整脚本

```python
from sin_model import generate_data, SinPredictor
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx


def collate_fn(batch):
    # TensorDataset 迭代的时候返回的是 Tuple(x, y), 量化的时候只需要x, 不需要label y。
    batch = batch[0].to(DEVICE)
    return batch


if __name__ == "__main__":

    ONNX_MODEL_PATH = "sin_model.onnx"
    ESPDL_MODEL_PATH = "sin_model.espdl"
    INPUT_SHAPE = [1, 1]  # 1 个输入特征
    TARGET = "esp32s3"  # 量化目标类型，可选 'c', 'esp32s3' or 'esp32p4'
    NUM_OF_BITS = 8  # 量化位数
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available

    x, y = generate_data()
    # dataloader shuffle必须设置为False。
    # 因为计算量化误差的时候会多次遍历数据集，如果shuffle是True的话，会得到错误的量化误差。
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # 校准的步数
        input_shape=INPUT_SHAPE,  # 输入形状，批次为 1
        inputs=None,
        target=TARGET,  # 量化目标类型
        num_of_bits=NUM_OF_BITS,  # 量化位数
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # 输出详细日志信息
    )
```

## 关键参数说明

| 参数 | 说明 |
| :--- | :--- |
| `onnx_import_file` | 输入的 ONNX 模型路径 |
| `espdl_export_file` | 输出的 .espdl 文件路径 |
| `calib_dataloader` | 校准数据加载器 |
| `calib_steps` | 校准步数，通常等于 batch_size |
| `input_shape` | 输入形状，batch size 必须为 1 |
| `target` | 目标平台：'c', 'esp32s3', 'esp32p4' |
| `num_of_bits` | 量化位数，通常为 8 |
| `collate_fn` | 数据整理函数，从 batch 中提取输入 |
| `export_test_values` | 是否导出测试输入/输出用于板端验证 |
| `verbose` | 日志详细程度，1 为详细 |

## PyTorch 转 ONNX

如果你的模型是 PyTorch 格式，需要先转换为 ONNX：

```python
import torch

# 加载模型
model = YourModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# 创建 dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

## 其他框架转 ONNX

| 源框架 | 转换工具 |
| :--- | :--- |
| TensorFlow | [tf2onnx](https://github.com/onnx/tensorflow-onnx) |
| TFLite | [tflite2onnx](https://github.com/zhenhuaw-me/tflite2onnx) |
| PaddlePaddle | [paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX) |
