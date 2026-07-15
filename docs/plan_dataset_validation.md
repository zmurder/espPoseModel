# Plan: 验证数据集并修复关键点顺序问题

## Context

训练前验证三个数据集（COCO、custom、cam_output）的正确性。发现 **custom 数据集有关键点顺序不一致的 bug**：
- COCO 和 cam_output 的关键点顺序：`[nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder]`
- custom 的关键点顺序：`[left_eye, right_eye, left_ear, right_ear, nose, left_shoulder, right_shoulder]`
- `dataset.py` 用 `coco_indices=[1,2,5,6]` 提取 4 个关键点
- 对 COCO/cam_output：正确提取 left_eye, right_eye, left_shoulder, right_shoulder
- 对 custom：错误提取了 **right_eye, left_ear, left_shoulder, right_shoulder**（left_eye 被映射成了 right_eye，right_eye 被映射成了 left_ear）

## 任务 1：修复 dataset.py 的关键点索引问题 ✅

**文件：** `dataset.py`

在 `load_coco_annotations` 中，从 categories 的 keypoint 名称动态确定索引：

```python
target_names = ['left_eye', 'right_eye', 'left_shoulder', 'right_shoulder']
kp_names = coco_data['categories'][0]['keypoints']
self.coco_indices = [kp_names.index(name) for name in target_names]
```

同时在 `__getitem__` 中将硬编码的 `coco_indices = [1, 2, 5, 6]` 改为 `coco_indices = self.coco_indices`。

修复后各数据集的索引映射：
- custom: `[0, 1, 5, 6]` → left_eye, right_eye, left_shoulder, right_shoulder ✅
- cam_output: `[1, 2, 5, 6]` → left_eye, right_eye, left_shoulder, right_shoulder ✅
- COCO: `[1, 2, 5, 6]` → left_eye, right_eye, left_shoulder, right_shoulder ✅

## 任务 2：验证三个数据集都能正确加载 ✅

验证结果：
1. ✅ 三个数据集都能被 PoseDatasetHeatmap 正确加载
2. ✅ 提取的 4 个关键点名称全部正确
3. ✅ 关键点位置合理：眼睛在上方(Y≈0.4)，肩膀在下方(Y≈0.8)
4. ✅ cam_output 的 flip_y 处理正确
5. ✅ 数据量：custom 72418 样本，cam_output 696 样本，COCO 11924 样本

## 任务 3：确认训练流程完整可运行

修复后用少量数据快速测试 train.py 能正常启动训练。
