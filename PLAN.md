# ESP32-S3 姿态检测模型 — 从零重写实现计划

## Context

用户要在 ESP32-S3 上训练一个轻量姿态检测模型，输入 OV3660 摄像头 320×240 图像，输出双眼+双肩 4 个关键点坐标（用于算距离/倾斜度判断坐姿）。项目已有 V1/V2 历史实现（git 历史中，工作区已删），V1 量化后 PCK@0.1=95.18%、模型 402KB，验证了技术路线可行。

用户要求**从零重写全套**程序（不恢复 git 历史），原因是 V2 代码存在多个未修复缺陷（见下）。训练前必须先用 MediaPipe 核验 custom 与 cam_data 两个自建数据集的真值——cam_data 真值已被删除、图片可能上下颠倒，需重新生成。WSL 环境有 NVIDIA GPU（RTX 5060）。数据用 COCO+custom+cam_data 三者混合。模型参考 MediaPipe(BlazePose) 设计。

## 用户决策与约束

- **代码起点**：从零重写全套（参考 V2 已验证的设计决策，但不照抄）
- **硬件**：有 NVIDIA GPU（RTX 5060, torch 2.10+cu128, CUDA 可用），脚本仍需自动检测并自适应降级
- **数据**：COCO + custom + cam_data 三者混合
- **算子约束**（docs/operator_support_state.md）：仅 batch=1；Conv 仅 int8/int16（groups=1 或 input_channels）；Resize 仅 int8 且不支持 antialias；ConvTranspose 经 InsertZeros+Conv 实现；ESP32-S3 用 ROUND_HALF_UP
- **量化**（docs/how_to_quantize_model.rst）：ESP-PPQ `espdl_quantize_onnx`，target=esp32s3，8-bit，calib_steps=32
- **分阶段训练**：少量 epoch → 可视化 test_img 对比 MediaPipe → 确认效果再继续，避免白训

## V2 遗留 bug（重写必须修复）

1. **MediaPipe 索引错误**：V2 `visualize_comparison_v2.py` 用 `{1:左眼,2:右眼,5:左肩,6:右肩}`，实际 MP PoseLandmark 是 `LEFT_EYE=2, RIGHT_EYE=5, LEFT_SHOULDER=11, RIGHT_SHOULDER=12`。重写修正为 `{2:0, 5:1, 11:2, 12:3}`。
2. **模型含 Resize 算子**：V2 `model_v2.py` forward 有 `F.interpolate(mode='bilinear')`，与"ConvTranspose 替代 Resize"目标矛盾，且 Resize 仅 int8 限制量化灵活性。重写完全消除，所有上采样用 `ConvTranspose2d(k2,s2)`。
3. **custom 关键点顺序**：`categories.keypoints` 声明 LitePose 序，但数据实际是 COCO 序（已几何验证：index 0 的点在双眼之间、眼下方肩上方 = nose）。提取索引应用 `[1,2,5,6]`（COCO 序），**不依赖 categories 名称查询**。
4. **offset loss 未做空间 mask**：V2 对全图算 L1，offset target 在远点处达 ±80，淹没关键点信号。重写 mask 到 `heatmap>0.3` 邻域。
5. **PCK target 取自 heatmap argmax**：V2 从 gt heatmap argmax 取 target，引入 ~1/60 离散化误差。重写直接用原始归一化坐标。

## 文件清单（11 个新文件，根目录扁平结构）

tools/ 下三个脚本保留不复用（内嵌 categories 名称 bug、按 7 点设计）。

| 文件 | 职责 |
|---|---|
| `kp_config.py` | 全局常量：4 关键点名称/顺序/权重、各数据集源 keypoint 索引映射、修正后的 MP landmark 映射、骨架、颜色 |
| `model.py` | BlazePose-lite 模型 + decode + 参数统计 |
| `dataset.py` | 统一数据集类 + DataLoader 工厂 + heatmap/offset/mask 生成 + 增强 |
| `train.py` | 训练循环：GPU 自适应、分阶段、checkpoint、TensorBoard、早停、OOM 恢复 |
| `verify_data.py` | MediaPipe 核验 custom 真值：抽样可视化 + MP 重检对比 + 顺序判定 + 异常检出 |
| `gen_cam_labels.py` | 为 cam_data 生成 COCO 真值：MP 检测 + 上下颠倒判断 + 4 点输出 |
| `visualize_data.py` | 可视化任意 4 点 COCO 数据集 |
| `compare_mediapipe.py` | test_img 上对比训练模型 vs MediaPipe |
| `evaluate.py` | PCK@0.1 全量评测（浮点模型） |
| `export_onnx.py` | 导出 ONNX(opset18) + onnxsim 融合 BN + 算子检查 |
| `quantize_espdl.py` | ESP-PPQ 量化 + 量化前后 PCK 对比 |

## 数据流程（最关键）

### 关键点索引统一（kp_config.py）

```python
TARGET_KP_NAMES = ['left_eye', 'right_eye', 'left_shoulder', 'right_shoulder']
TARGET_KP_WEIGHTS = [1.0, 1.0, 1.4, 1.4]  # 肩膀权重高
DATASET_KP_INDEX = {
    'coco':   [1, 2, 5, 6],   # COCO 17点中提取
    'custom': [1, 2, 5, 6],   # custom 7点(COCO序)，非 categories 名称查得
    'cam':    [0, 1, 2, 3],   # 我们自生成的4点，顺序即 TARGET_KP_NAMES
}
MP_KP_INDEX = {'left_eye':2, 'right_eye':5, 'left_shoulder':11, 'right_shoulder':12}
```

### custom 真值核验（verify_data.py）

1. 从 `data/custom/annotations/person_keypoints_val.json` 抽样 300 张
2. 每张跑 `mp_pose.Pose(static_image_mode=True, model_complexity=2)`，取 MP 4 点（索引 2,5,11,12）转像素坐标
3. 对 custom 7 点分别用两种索引假设提取 4 点：A=COCO序`[1,2,5,6]`、B=LitePose序`[0,1,5,6]`
4. 比较两种假设与 MP 检测的逐点欧氏距离中位数，统计哪种更小 → 确认真实顺序
5. 输出异常样本（距离>50px）列表；抽样 20 张可视化（custom 点+MP 点不同颜色）存 `verify_output/`
6. **预期**：假设 A 距离显著更小，确认 custom 为 COCO 序。此步通过后才训练

### cam_data 真值生成（gen_cam_labels.py）

1. 遍历 `data/cam_data/*.jpg`（800 张 320×240）
2. 每张：原图跑 MP + 垂直翻转图(`cv2.flip(img,0)`)跑 MP，比较 4 点 visibility 之和，取高者
3. flip 胜则用翻转图结果（坐标已是翻转后）、记 `flipped=True`；两者都无姿态则跳过
4. 提取 4 点（MP 索引 2,5,11,12），visibility≥0.5 → v=2 否则 v=0
5. 过滤：4 点中 ≥3 点 v=2 才保留
6. 输出 `data/cam_data/annotations/person_keypoints.json`（categories.keypoints=4 点名，与数据一致）
7. MP 实例循环外创建复用；单张失败 try/except 跳过；每 100 张存部分结果

### 三数据集统一加载（dataset.py）

- `PoseDataset(source=...)`，数据源配置含 ann_file/img_dir/kp_indices/split
- 加载：读 COCO JSON → 仅留单人图 → 用 kp_indices 提取 4 点 → 过滤 4 点中 v≥1 的 ≥3 个
- `__getitem__`：读图→`center_crop_resize`到 320×240（避免 letterbox 黑边）→ 关键点坐标转换(减crop偏移→乘缩放→归一化)→ 训练增强(scale0.8-1.2/rot±15°/shift±10%/hflip0.5/ColorJitter，hflip 时交换 (0,1)(2,3))→ ToTensor+Normalize(ImageNet)→ 生成 heatmap+offset+mask
- 几何 sanity check：前 100 样本检查 eyes_y<shoulders_y，违反率>20% 则警告
- DataLoader 用 `WeightedRandomSampler` 控制比例：custom 50% + coco 30% + cam 20%（cam 过采样，若过拟合降 to 10%）

### heatmap/offset/mask 生成（修正 V2）

```python
# heatmap (4,60,80): 高斯核 exp(-dist²/2σ²), σ=3
# offset (8,60,80): [kp0_dx,kp0_dy,...] = (hx - xs, hy - ys)
# mask (4,60,80): heatmap>0.3 的关键点邻域，用于 offset loss
```

## 模型设计（model.py）

BlazePose-lite，仅用 ESP-DL 支持算子（Conv/BN/HardSwish/ConvTranspose/Add/Sigmoid），**完全消除 Resize**：

```
Stem:    Conv(3→32,k3,s2,p1,bias=False)+BN+HardSwish          → (32,120,160)
Backbone:
  block1: InvertedResidual(32→32,s1,expand=4)                 → (32,120,160)
  block2: InvertedResidual(32→64,s2,expand=4)                 → (64,60,80)
  block3: InvertedResidual(64→64,s1,expand=4)                 → (64,60,80)
  block4: InvertedResidual(64→64,s2,expand=4)                 → (64,30,40)
FPN:
  up:    ConvTranspose2d(64→64,k2,s2)                         → (64,60,80)
  fuse:  Add(up(x4), x2) → Conv(64→64,k1)+BN+HardSwish        → (64,60,80)
Heads (60×80):
  heatmap: Conv(64→64,k3,p1)+BN+HardSwish + Conv(64→4,k1)+Sigmoid → (4,60,80)
  offset:  Conv(64→64,k3,p1)+BN+HardSwish + Conv(64→8,k1)        → (8,60,80)
```

- InvertedResidual：1×1升维→3×3 depthwise(groups=hidden_dim,符合 groups 约束)→1×1降维，s1且通道匹配时残差
- 用 `nn.Hardswish()` 替代手写 clamp
- decode（仅推理/评测，不导出）：heatmap argmax + offset → 归一化坐标，含 max/gather 不导出，ESP32 C++ 后处理
- 参数量 ~250-300K

## 训练设计（train.py）

- batch=32(GPU)/8(CPU 自适应), epochs=100, lr=1e-3, wd=1e-4, AdamW, ReduceLROnPlateau(mode=max,factor=0.5,patience=10,min_lr=1e-5), grad_clip=1.0
- Loss：heatmap MSE(带 kp 权重 [1,1,1.4,1.4]) + offset L1(带空间 mask，仅 heatmap>0.3 邻域，按 mask 归一化)
- 数据混合：custom 50% + coco 30% + cam 20%（WeightedRandomSampler）
- 验证集：custom val split（12779 张，同分布，作训练监控）
- **分阶段**：`--phase quick --epochs 5` 训练后提示运行 compare_mediapipe.py；`--phase full --epochs 100 --resume` 继续
- Checkpoint：latest.pth(每epoch)/best.pth(PCK最优)/epoch_N.pth(每10epoch)；--resume 恢复 epoch+optimizer+scheduler
- TensorBoard：train/loss、hm_loss、offset_loss、lr；val/loss、pck_10、kp_accuracy/{name}
- 早停：patience=20（val PCK 连续 20 epoch 无提升）
- OOM 恢复：catch OutOfMemoryError，empty_cache + 跳过该 batch

## 验证与评测

- **compare_mediapipe.py**：test_img 5 张，各自 center_crop_resize→模型得 4 点；原图送 MP（**修正索引 2,5,11,12**）得 4 点；并排画（MP 绿/模型蓝+骨架）存 `comparison/`；打印逐点坐标+距离。判断标准：平均距离<30px 则可接受进 phase 2。
- **evaluate.py**：PCK@0.1，threshold=0.1×sqrt(320²+240²)≈40px；target 用原始归一化坐标（非 heatmap argmax）；总体+4 点分别统计。
- **quantize_espdl.py**：浮点 PCK → `espdl_quantize_onnx`(calib=cam_data val, shuffle=False, collate 只返图像, calib_steps=32, input_shape=[1,3,240,320], target=esp32s3, 8-bit) → 量化 PCK；打印精度损失（预期<3%）。

## 执行顺序（每步带验证）

0. 环境已确认（torch/CUDA/mediapipe/esp-ppq/onnxsim 齐全）
1. 写 `kp_config.py`+`model.py` → `python model.py` 验证输出形状(4,60,80)+(8,60,80)、参数量
2. 写 `verify_data.py` → `python verify_data.py --num_samples 300` → 确认 COCO 序假设胜、异常率<5%、可视化对齐 **（通过后才继续）**
3. 写 `gen_cam_labels.py` → `python gen_cam_labels.py` → 成功率>70%、翻转比例合理、`visualize_data.py` 抽样确认
4. 写 `dataset.py`+`visualize_data.py` → `python dataset.py` 测试三源加载、可视化 4 点位置正确（眼上肩下）
5. 写 `train.py` → `python train.py --phase quick --epochs 5` → loss 下降、无 OOM
6. 写 `compare_mediapipe.py` → `python compare_mediapipe.py --model_path checkpoints/latest.pth` → **肉眼判断 5 张 test_img**：距离<30px 进 step 7，否则分析（数据顺序/增强/学习率）
7. `python train.py --phase full --epochs 100 --resume` → val PCK 提升、早停或训完（预期 PCK@0.1>90%）
8. 写 `evaluate.py` → `python evaluate.py --model_path checkpoints/best.pth` → 总体+4 点 PCK
9. 写 `export_onnx.py` → `python export_onnx.py` → onnxsim 融合成功、无 BN 残留、算子检查通过
10. 写 `quantize_espdl.py` → `python quantize_espdl.py` → 生成 .espdl/.info/.json、量化 PCK 损失<3%

## WSL 防崩要点

- DataLoader：num_workers=4(GPU)/0(CPU)、pin_memory、persistent_workers、prefetch_factor=2
- 显存监控：每 50 batch 打印；OOM 跳过 batch 清缓存
- 数据量：COCO 过滤后 ~1.1 万、custom 7.2 万，JSON 一次读入；`--max_samples` 可抽样降载
- `cudnn.benchmark=True`、每 epoch `empty_cache()`、不用 AMP（模型小，保稳定）
- MediaPipe：实例循环外创建复用、单张失败 try/except 跳过、定期存部分结果
- KeyboardInterrupt 时保存 latest.pth

## 关键复用点

| V2 代码 | 复用 | 改进 |
|---|---|---|
| model_v2.py | InvertedResidual、Stem+Backbone+双头、decode | 消除 F.interpolate；nn.Hardswish；FPN 简化为单层 ConvTranspose |
| dataset.py | center_crop_resize、增强管线 | offset 加 mask；索引不依赖 categories；WeightedRandomSampler |
| train_v2.py | loss 结构、AdamW+scheduler、PCK、checkpoint | offset loss mask；分阶段；OOM 恢复；早停；target 取原始坐标 |
| export_v2.py | onnxsim、opset18、算子检查 | 清理结构 |
| quantize_v2.py | 量化模型评估、PCK 对比 | 改用 espdl_quantize_onnx（ONNX 路径更稳） |
| visualize_comparison_v2.py | 对比图绘制 | **修正 MP 索引**(2,5,11,12) |
| tools/video_to_dataset.py | 正确的 MP 索引 [2,5,7,8,0,11,12] | 不复用脚本，参考索引 |
