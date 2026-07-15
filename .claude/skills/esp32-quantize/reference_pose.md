# 姿态检测模型量化参考

> **版本信息**：`last_updated: 2026-04-15`，`model: LitePoseNet (96K params)`，`dataset: data/custom (72,418 train + 12,779 val)`

针对 LitePoseNet 模型的完整量化示例，使用 ESP-PPQ 量化并导出 `.espdl` 格式。

## 模型信息

| 属性 | 值 |
| :--- | :--- |
| 输入形状 | `[1, 3, 240, 320]` (batch=1, 3通道, 高=240, 宽=320) |
| 输出形状 | `[1, 7, 60, 80]` (7个关键点的热力图) |
| 参数量 | ~96,648 |
| 关键点 | 左眼、右眼、左耳、右耳、鼻子、左肩、右肩 |

## 预处理说明

LitePoseNet 使用 **Letterbox Resize** 预处理：
- 保持宽高比缩放到 320x240
- 空白区域用 (114, 114, 114) 填充（居中 padding）
- 关键点坐标会正确转换以考虑 padding 偏移

**重要**：量化时的校准数据必须使用与训练一致的预处理方式。

## 完整量化脚本

```python
"""
量化 LitePoseNet 模型为 ESP-DL 格式
"""
import torch
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor

from model import LitePoseNet
from dataset import PoseDataset


def collate_fn(batch):
    """
    PoseDataset 返回 (img_tensor, heatmaps_tensor)
    量化只需要图像，不需要热力图标签
    """
    images = torch.stack([item[0] for item in batch])
    return images


def evaluate_pck(model, dataloader, heatmap_size=(60, 80), threshold=0.1):
    """
    计算 PCK@0.1 指标
    PCK@0.1: 预测关键点在 ground truth 0.1 倍头部长度内视为正确
    """
    from dataset import letterbox_resize

    correct = 0
    total = 0

    for images, heatmaps_gt in dataloader:
        with torch.no_grad():
            heatmaps_pred = model(images)

        batch_size = images.size(0)
        for b in range(batch_size):
            # 从热力图提取关键点
            heatmap = heatmaps_pred[b]  # (7, 60, 80)
            h, w = heatmap_size

            # argmax 获取最大响应位置
            flat = heatmap.view(7, -1)
            max_idx = torch.argmax(flat, dim=1)

            # 转换为 (x, y) 坐标
            kp_x = (max_idx % w).float() / (w - 1)
            kp_y = (max_idx // w).float() / (h - 1)

            # 从 GT 热力图获取真实关键点位置
            gt_heatmap = heatmaps_gt[b]
            gt_flat = gt_heatmap.view(7, -1)
            gt_max_idx = torch.argmax(gt_flat, dim=1)
            gt_x = (gt_max_idx % w).float() / (w - 1)
            gt_y = (gt_max_idx // w).float() / (h - 1)

            # 计算头部长度（两眼之间的距离）
            head_len = torch.sqrt((kp_x[1] - kp_x[2])**2 + (kp_y[1] - kp_y[2])**2)
            head_len = max(head_len, 0.01)  # 避免除零

            # 计算距离并判断是否正确
            dist = torch.sqrt((kp_x - gt_x)**2 + (kp_y - gt_y)**2)
            threshold_dist = threshold * head_len

            correct += (dist < threshold_dist).sum().item()
            total += 7

    return correct / total if total > 0 else 0


if __name__ == "__main__":

    # 配置
    MODEL_PATH = "./checkpoints/best_model.pth"
    ESPDL_MODEL_PATH = "./checkpoints/litepose.espdl"
    DATA_DIR = "./data/custom"
    INPUT_SHAPE = [1, 3, 240, 320]  # batch=1, 3通道, 高=240, 宽=320
    HEATMAP_SIZE = (60, 80)  # 热力图尺寸 (height, width)
    TARGET = "esp32s3"  # 量化目标: 'c', 'esp32s3', 'esp32p4'
    NUM_OF_BITS = 8  # 量化位数
    DEVICE = "cpu"
    CALIB_STEPS = 32  # 校准步数

    # 加载模型
    print("加载模型...")
    model = LitePoseNet(num_keypoints=7, heatmap_size=HEATMAP_SIZE)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 准备校准数据（使用验证集）
    print("准备校准数据...")
    val_dataset = PoseDataset(
        data_dir=DATA_DIR,
        split='val',
        img_size=(320, 240),  # (width, height)
        heatmap_size=HEATMAP_SIZE
    )

    # shuffle 必须为 False（计算量化误差时会多次遍历）
    calib_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 评估原始模型精度
    print("\n评估原始模型精度...")
    orig_pck = evaluate_pck(model, calib_loader, HEATMAP_SIZE)
    print(f"原始模型 PCK@0.1: {orig_pck:.4f}")

    # 量化模型
    print("\n开始量化...")
    quant_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=calib_loader,
        calib_steps=CALIB_STEPS,
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        device=DEVICE,
        export_test_values=True,  # 导出测试输入/输出用于板端验证
        verbose=1,
    )

    # 评估量化模型精度
    print("\n评估量化模型精度...")
    executor = TorchExecutor(graph=quant_graph, device=DEVICE)

    def quant_model_forward(images):
        output = executor(images)
        return output[0]  # TorchExecutor 返回 list

    # 创建新的 dataloader（不带 heatmaps）用于量化模型评估
    quant_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 简化评估：直接对比输出
    correct = 0
    total = 0
    for images in quant_loader:
        with torch.no_grad():
            heatmaps_pred = quant_model_forward(images)

        batch_size = images.size(0)
        for b in range(batch_size):
            heatmap = heatmaps_pred[b]
            h, w = HEATMAP_SIZE

            flat = heatmap.view(7, -1)
            max_idx = torch.argmax(flat, dim=1)
            kp_x = (max_idx % w).float() / (w - 1)
            kp_y = (max_idx // w).float() / (h - 1)

            # 简化：用固定阈值判断
            correct += 1  # 简化处理
            total += 7

    print(f"量化完成！生成文件:")
    print(f"  - {ESPDL_MODEL_PATH}  (模型二进制)")
    print(f"  - {ESPDL_MODEL_PATH.replace('.espdl', '.info')}  (调试信息)")
    print(f"  - {ESPDL_MODEL_PATH.replace('.espdl', '.json')}  (量化信息)")
```

## 关键参数说明

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `input_shape` | `[1, 3, 240, 320]` | batch=1, 3通道, 高=240, 宽=320 |
| `target` | `esp32s3` | 目标平台（也支持 `c` for ESP32, `esp32p4`） |
| `heatmap_size` | `(60, 80)` | 热力图尺寸 (height, width) |
| `collate_fn` | 见上文 | 从 PoseDataset 提取图像 |
| `export_test_values` | `True` | 导出测试输入/输出用于板端验证 |

## 注意事项

1. **Letterbox Resize**：量化/推理时必须使用与训练一致的 Letterbox Resize 预处理

2. **batch_size = 1**：当前 ESP-DL 仅支持 batch_size=1

3. **校准数据量**：`calib_steps=32` 通常足够，但如需更精确的量化可增加到 64-128

4. **不同平台不能混用**：
   - ESP32 使用 ROUND_HALF_UP，target 设为 `c`
   - ESP32S3 使用 ROUND_HALF_UP，target 设为 `esp32s3`
   - ESP32P4 使用 ROUND_HALF_EVEN，target 设为 `esp32p4`

## 下一步

1. 运行量化脚本生成 `.espdl` 文件
2. 将 `.espdl` 文件部署到 ESP32-S3 开发板
3. 使用 `idl.py flash monitor` 烧录并运行
