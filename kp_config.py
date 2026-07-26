"""
姿态检测模型全局配置
6 关键点: left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder
用于 ESP32-S3 坐姿检测（算距离/倾斜度，低头时用耳朵替代眼睛）
"""
import os

# ============ 关键点定义 ============
NUM_KEYPOINTS = 6
# 顺序: left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder
TARGET_KP_NAMES = ['left_eye', 'right_eye', 'left_ear', 'right_ear',
                   'left_shoulder', 'right_shoulder']
TARGET_KP_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 2.5, 2.5]  # 肩膀权重最高(定位优先)

# 骨架连接 (关键点索引对)
SKELETON = [[0, 1], [2, 3], [4, 5], [0, 2], [1, 3]]  # 双眼/双耳/双肩/同侧眼-耳

# BGR 颜色 (用于可视化，OpenCV 默认 BGR)
KP_COLORS = [
    (0, 0, 255),      # left_eye 红
    (255, 0, 255),    # right_eye 粉
    (0, 255, 255),    # left_ear 黄
    (255, 255, 0),    # right_ear 青
    (0, 165, 255),    # left_shoulder 橙
    (255, 165, 0),    # right_shoulder 黄
]

# ============ 各数据集关键点提取索引 ============
# 从源 keypoints 数组中提取 6 个目标点的索引
# COCO 17点标准序: [nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4),
#                   left_shoulder(5), right_shoulder(6), left_elbow(7), ...]
# custom 实际是 COCO 序 (已几何验证)
# cam 是我们自生成的 6 点, 顺序即 TARGET_KP_NAMES
DATASET_KP_INDEX = {
    'coco':   [1, 2, 3, 4, 5, 6],   # 眼1,2 + 耳3,4 + 肩5,6 (COCO序)
    'custom': [1, 2, 3, 4, 5, 6],   # custom 同 COCO 序
    'cam':    [0, 1, 2, 3, 4, 5],   # cam 自生成 6 点, 顺序即 TARGET_KP_NAMES
}

# ============ MediaPipe PoseLandmark 索引 ============
# MP PoseLandmark: LEFT_EYE=2/RIGHT_EYE=5/LEFT_EAR=7/RIGHT_EAR=8/LEFT_SHOULDER=11/RIGHT_SHOULDER=12
MP_KP_INDEX = {
    'left_eye':       2,    # PoseLandmark.LEFT_EYE
    'right_eye':      5,    # PoseLandmark.RIGHT_EYE
    'left_ear':       7,    # PoseLandmark.LEFT_EAR
    'right_ear':      8,    # PoseLandmark.RIGHT_EAR
    'left_shoulder':  11,   # PoseLandmark.LEFT_SHOULDER
    'right_shoulder': 12,   # PoseLandmark.RIGHT_SHOULDER
}
# MediaPipe landmark 索引 -> 我们的 6 点索引
MP_TO_OUR = {2: 0, 5: 1, 7: 2, 8: 3, 11: 4, 12: 5}

# ============ 图像与热图尺寸 ============
IMG_WIDTH = 320
IMG_HEIGHT = 240
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)            # (W, H)

# 高分辨率纯 Heatmap：1/2 输入分辨率，定位精度高、后处理只需 argmax
HEATMAP_WIDTH = 160
HEATMAP_HEIGHT = 120
HEATMAP_SIZE = (HEATMAP_HEIGHT, HEATMAP_WIDTH)  # (H, W)

SIGMA = 5.0  # 高斯热图标准差: 宽峰量化友好(int8 不压垮峰值), 兼顾定位精度

# ============ 归一化 ============
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ============ PCK ============
PCK_THRESHOLD_RATIO = 0.1  # PCK@0.1: 距离 < 0.1 * 对角线 视为正确

# ============ 数据集路径 ============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

DATASET_SOURCES = {
    'coco_train': {
        'ann_file': os.path.join(DATA_DIR, 'annotations', 'person_keypoints_train2017.json'),
        'img_dir': os.path.join(DATA_DIR, 'train2017'),
        'kp_indices': DATASET_KP_INDEX['coco'],
        'source_type': 'coco',
        'split': 'train',
    },
    'coco_val': {
        'ann_file': os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2017.json'),
        'img_dir': os.path.join(DATA_DIR, 'val2017'),
        'kp_indices': DATASET_KP_INDEX['coco'],
        'source_type': 'coco',
        'split': 'val',
    },
    'custom_train': {
        'ann_file': os.path.join(DATA_DIR, 'custom', 'annotations', 'person_keypoints_train.json'),
        'img_dir': os.path.join(DATA_DIR, 'custom', 'images'),
        'kp_indices': DATASET_KP_INDEX['custom'],
        'source_type': 'custom',
        'split': 'train',
    },
    'custom_val': {
        'ann_file': os.path.join(DATA_DIR, 'custom', 'annotations', 'person_keypoints_val.json'),
        'img_dir': os.path.join(DATA_DIR, 'custom', 'images'),
        'kp_indices': DATASET_KP_INDEX['custom'],
        'source_type': 'custom',
        'split': 'val',
    },
    'cam': {
        'ann_file': os.path.join(DATA_DIR, 'cam_data', 'annotations', 'person_keypoints.json'),
        'img_dir': os.path.join(DATA_DIR, 'cam_data'),
        'kp_indices': DATASET_KP_INDEX['cam'],
        'source_type': 'cam',
        'split': 'all',
    },
}

# 训练数据混合比例 (WeightedRandomSampler 用)
DATA_MIX_RATIO = {
    'custom_train': 0.5,
    'coco_train': 0.2,   # 原0.3, 站姿全身域差大降权
    'cam': 0.3,          # 原0.2, cam 是部署域提权
}

# ============ 输出目录 ============
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
