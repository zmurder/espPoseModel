# PyTorch 模型量化参考

> **版本信息**：`last_updated: 2026-04-15`，`source: ESP-DL how_to_quantize_model.html`

完整的 PyTorch 模型量化示例，来自 ESP-DL 官方教程。

## 完整脚本

```python
from sin_model import generate_data, SinPredictor
from torch.utils.data import DataLoader, TensorDataset
import torch
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor


def collate_fn(batch):
    # When iterating over TensorDataset, it returns Tuple(x, y). For quantization, only x is needed, not label y.
    batch = batch[0].to(DEVICE)
    return batch


if __name__ == "__main__":

    ESPDL_MODEL_PATH = "sin_model.espdl"
    INPUT_SHAPE = [1, 1]  # 1 input feature
    TARGET = (
        "esp32s3"  # Quantization target type, options: 'c', 'esp32s3', or 'esp32p4'
    )
    NUM_OF_BITS = 8  # Number of quantization bits
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available

    x_test, y_test = generate_data()
    # Dataloader shuffle must be set to False.
    # Because when calculating quantization error, the dataset will be traversed multiple times. If shuffle is True, the quantization error will be incorrect.
    dataset = TensorDataset(x_test, y_test)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = SinPredictor()
    model.load_state_dict(torch.load("sin_model.pth"))
    model.eval()

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # Number of calibration steps
        input_shape=INPUT_SHAPE,  # Input shape, batch size is 1
        inputs=None,
        target=TARGET,  # Quantization target type
        num_of_bits=NUM_OF_BITS,  # Quantization bits
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # Output detailed log information
    )

    criterion = torch.nn.MSELoss()
    # Test accuracy of the original model on the test set
    loss = 0
    for batch_x, batch_y in dataloader:
        y_pred = model(batch_x)
        loss += criterion(y_pred, batch_y)
    loss /= len(dataloader)
    print(f"origin model loss: {loss.item():.5f}")

    # Test accuracy of the quantized model on the test set
    executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)
    loss = 0
    for batch_x, batch_y in dataloader:
        y_pred = executor(batch_x)
        loss += criterion(y_pred[0], batch_y)
    loss /= len(dataloader)
    print(f"quant model loss: {loss.item():.5f}")
```

## 关键参数说明

| 参数 | 说明 |
| :--- | :--- |
| `model` | PyTorch 模型，必须是 eval 模式 |
| `espdl_export_file` | 输出的 .espdl 文件路径 |
| `calib_dataloader` | 校准数据加载器 |
| `calib_steps` | 校准步数，通常等于 batch_size |
| `input_shape` | 输入形状，batch size 必须为 1 |
| `target` | 目标平台：'c', 'esp32s3', 'esp32p4' |
| `num_of_bits` | 量化位数，通常为 8 |
| `collate_fn` | 数据整理函数，从 batch 中提取输入 |
| `export_test_values` | 是否导出测试输入/输出用于板端验证 |
| `verbose` | 日志详细程度，1 为详细 |

## 注意事项

1. **Dataloader shuffle 必须为 False**：计算量化误差时会多次遍历数据集，shuffle 为 True 会导致错误的结果

2. **collate_fn 的作用**：TensorDataset 迭代返回 `(x, y)` 元组，但量化只需要输入 `x`

3. **精度验证**：使用 `TorchExecutor` 可以直接用量化后的图进行推理，验证精度
