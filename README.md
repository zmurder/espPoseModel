# ESP32-S3 坐姿检测模型

在 ESP32-S3 上实时检测坐姿:输入 OV3660 摄像头 320×240 图像,输出双眼+双肩 4 个关键点,用于算距离/倾斜度判断坐姿是否端正。

端到端流程:**数据核验 → 训练 → 评测 → 导出 → 量化 → ESP32 部署**。

## 指标

| 项 | 值 |
|---|---|
| 模型 | PoseNet,0.84M 参数,纯 heatmap(120×160) |
| 算子 | 仅 5 种 ESP-DL 支持:`Conv/ConvTranspose/HardSwish/Add/Sigmoid`(无 Resize) |
| 量化 | int8 对称,`pose_model.espdl` 1.2M |
| PCK@0.1 (cam 真实域) | 95.06%(量化后,损失 0.68%) |
| conf≥0.6 时 | 眼睛 PCK 100%,肩膀 97–99.7%(中位偏差 3–5px) |
| 双肩倾斜角中位误差 | 2.09°(坐姿判断阈值通常 10–15°,远够用) |

## 目录结构

```
esp32_pose_model/
├── README.md                   # 本文件(全流程)
├── PLAN.md                     # 实现计划
│
├── kp_config.py                # 全局配置(关键点/数据源/尺寸/混合比例)
├── model.py                    # PoseNet 模型 + decode
├── dataset.py                  # 三源统一加载 + heatmap 生成 + 增强
├── train.py                    # 分阶段训练(quick→full)
├── evaluate.py                 # PCK@0.1 全量评测
├── export_onnx.py              # ONNX 导出 + onnxsim 融合 BN + 算子检查
├── quantize_espdl.py           # ESP-PPQ int8 量化 + 量化前后 PCK 对比
│
├── verify_data.py              # MediaPipe 核验 custom 真值顺序
├── gen_cam_labels.py           # cam_data 真值生成(删头朝下图)
├── visualize_data.py           # 4 点数据集可视化
├── compare_mediapipe.py        # test_img 上对比模型 vs MediaPipe
│
├── diag_pose.py                # 坐姿指标诊断(倾斜角/肩宽/偏差方向)
├── diag_hard.py                # 困难样本坐标对比(左右肩是否搞反)
├── conf_pck.py                 # 置信度分层 PCK(验证 conf 阈值有效性)
├── verify_onnx.py              # 验证 onnx == best.pth(量化链路源头)
│
├── data/                       # 三源数据
│   ├── train2017/ val2017/     # COCO(过滤单人+4点可见)
│   ├── custom/                 # bilibili 视频截图(MediaPipe 核验为 COCO 序)
│   ├── cam_data/               # ESP32 实拍(MediaPipe 生成真值,已删头朝下)
│   └── annotations/
│
├── checkpoints/                # best.pth / latest.pth / epoch_*.pth
├── output/                     # pose_model.{onnx, espdl, info, json}
├── test_img/                   # 对比测试图
├── comparison*/                # 对比可视化输出
└── esp32_deploy/               # ESP32-S3 部署工程(见 esp32_deploy/README.md)
```

## 环境依赖

```bash
# Python 训练环境(WSL/Linux + NVIDIA GPU)
pip install torch torchvision opencv-python numpy
pip install mediapipe onnx onnxsim onnxruntime
pip install esp-ppq            # ESP-PPQ 量化
pip install tensorboard tqdm
```

> RTX 5060(Blackwell)注意:`train.py` 已设 `cudnn.benchmark=False`(benchmark 会卡住),batch_size 默认 16(8GB 显存)。

## 快速开始(端到端)

### 0. 数据准备

三源数据放到 `data/`:
- **COCO**:`train2017/` `val2017/` + `annotations/person_keypoints_{train,val}2017.json`
- **custom**(bilibili 截图):`custom/images/` + `custom/annotations/`
- **cam_data**(ESP32 实拍):`cam_data/*.jpg`

**核验 custom 真值**(关键点顺序存疑,必须先验):
```bash
python3 verify_data.py --num_samples 300
# 预期: COCO 序假设胜, 中位距离 ~1.4px, 异常率 <5%
```

**生成 cam_data 真值**(真值已删,图片可能上下颠倒):
```bash
python3 gen_cam_labels.py
# 自动: MediaPipe 检测 -> 几何判断头朝下则删图 -> 保留正向图生成标注
```

### 1. 训练(分阶段)

```bash
# Phase 1: 快速验证(5 epoch)
python3 train.py --phase quick --epochs 5

# 可视化确认效果(用 best.pth)
python3 compare_mediapipe.py --model_path checkpoints/best.pth

# Phase 2: 完整训练(resume from quick)
python3 train.py --phase full --epochs 100 --resume
```

### 2. 评测

```bash
# 浮点模型逐点 PCK
python3 evaluate.py --model_path checkpoints/best.pth --source cam
python3 evaluate.py --model_path checkpoints/best.pth --source custom_val

# 坐姿指标诊断(倾斜角/肩宽误差,判断坐姿可用性)
python3 diag_pose.py --source cam
```

### 3. 导出 ONNX

```bash
python3 export_onnx.py --model_path checkpoints/best.pth
# 预期: onnxsim 融合 BN, 算子全在 ESP-DL 支持列表, 无 BN 残留
```

### 4. 量化

```bash
python3 quantize_espdl.py --onnx_path output/pose_model.onnx
# 预期: 生成 pose_model.espdl, 量化损失 <3%
```

### 5. ESP32 部署

详见 [`esp32_deploy/README.md`](esp32_deploy/README.md)。核心:
```bash
cd esp32_deploy
idf.py set-target esp32s3
idf.py add-dependency "espressif/esp-dl" "espressif/esp32-camera"
# 拷 output/pose_model.{espdl,info} 到 SD 卡
idf.py build flash monitor
```

## 模型架构(PoseNet)

高分辨率纯 Heatmap,从零设计(不参考历史 V2):

```
输入 (B,3,240,320)
  ├─ Stem: Conv(3→32,s2)+BN+HardSwish                    → (32,120,160)
  ├─ Backbone 4 级下采样:
  │    stage1 (32,120×160) → stage2 (64,60×80)
  │    → stage3 (128,30×40) → stage4 (256,15×20)
  ├─ Top-down FPN(ConvTranspose 逐级上采样融合多尺度):
  │    c4 → up3+c3 → up2+c2 → up1+c1                    → (32,120×160)
  └─ Heatmap head: Conv+BN+HardSwish+Conv+Sigmoid        → (4,120×160)
```

- **高分辨率 heatmap**(1/2 输入,120×160):定位精度高,后处理只需 argmax
- **无 Resize**:所有上采样用 `ConvTranspose2d(k2,s2)`,符合 ESP-DL 算子约束
- **多尺度 FPN**:融合 4 层特征,这是 V2 缺失、导致肩膀定位差的关键改进
- decode:argmax 取归一化坐标(ESP32 C++ 后处理简单)

## 关键设计决策

| 问题 | 方案 |
|---|---|
| custom 关键点顺序存疑 | MediaPipe 几何核验,确认 COCO 序 `[1,2,5,6]`(中位 1.4px) |
| cam_data 头朝下 | 几何判断 `eye_y<shoulder_y`,头朝下图直接删除(42 张),保留 666 张正向 |
| 三源数据混合 | `WeightedRandomSampler`:custom 50% + COCO 30% + cam 20% |
| MediaPipe 索引 | 修正为 `LEFT_EYE=2,RIGHT_EYE=5,LEFT_SHOULDER=11,RIGHT_SHOULDER=12` |
| 肩膀定位差 | 诊断发现 conf<0.6 的 13% 难检样本 argmax 跑偏 → 后处理 conf 阈值过滤 |
| 量化精度 | int8 对称,损失 0.68%(95.74%→95.06%) |

## 脚本速查

| 脚本 | 用途 | 关键命令 |
|---|---|---|
| `verify_data.py` | 核验 custom 真值 | `--num_samples 300` |
| `gen_cam_labels.py` | 生成 cam 真值 | (无参) |
| `train.py` | 训练 | `--phase quick/full --resume` |
| `compare_mediapipe.py` | 对比 MediaPipe | `--model_path best.pth --img_dir <dir>` |
| `evaluate.py` | PCK 评测 | `--source cam/custom_val/coco_val` |
| `export_onnx.py` | 导出 ONNX | `--model_path best.pth` |
| `quantize_espdl.py` | int8 量化 | `--onnx_path output/pose_model.onnx` |
| `diag_pose.py` | 坐姿指标诊断 | `--source cam` |
| `conf_pck.py` | conf 分层 PCK | (无参) |
| `verify_onnx.py` | 验证 onnx==best | (无参) |

## 常见问题

**Q: 训练卡在第一个 batch 无输出?**
A: RTX 5060(Blackwell)上 `cudnn.benchmark=True` 会卡,`train.py` 已设 False。若仍卡,检查 batch_size 是否超显存(8GB 用 16)。

**Q: 可视化对比效果差?**
A: 先确认用的是 `best.pth` 而非 `latest.pth`/旧 checkpoint。肩膀偏差大的样本看 `diag_hard.py` 输出的 conf——conf 低是难检样本,用后处理阈值过滤即可,不影响坐姿判断。

**Q: ESP32 实测 conf 普遍偏低?**
A: 99% 是前处理没对齐。检查:颜色(RGB)/归一化(ImageNet)/裁剪(center crop 4:3)。对照表见 `esp32_deploy/README.md`。

**Q: 想换关键点/改尺寸?**
A: 改 `kp_config.py`(`TARGET_KP_NAMES`/`IMG_SIZE`/`HEATMAP_SIZE`/`SIGMA`),其余脚本自动适配。改尺寸需同步 `model.py` 的 Stem 下采样比例。

## 文档

- [`PLAN.md`](PLAN.md) — 实现计划与设计依据
- [`docs/说明.txt`](docs/说明.txt) — 原始需求
- [`docs/how_to_quantize_model.rst`](docs/how_to_quantize_model.rst) — ESP-PPQ 量化教程
- [`docs/how_to_deploy_mobilenetv2.rst`](docs/how_to_deploy_mobilenetv2.rst) — ESP-DL 部署教程
- [`docs/operator_support_state.md`](docs/operator_support_state.md) — ESP-DL 算子支持
- [`esp32_deploy/README.md`](esp32_deploy/README.md) — 部署工程说明
