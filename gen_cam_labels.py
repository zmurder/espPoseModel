"""
为 cam_data 生成 COCO 格式 4 关键点真值
==================================
cam_data 是 ESP32 实拍图 (320x240)，真值已删除，图片可能上下颠倒。
本脚本用 MediaPipe 检测，自动判断上下颠倒（原图 vs 垂直翻转图，取 4 点 visibility 之和更高者），
输出 4 点 COCO 标注。annotation 中 flipped 字段记录是否需翻转，dataset 读取时据此翻转图片
（坐标已对应翻转后的图，原图不破坏）。

用法:
    python3 gen_cam_labels.py
"""
import os
import json
import glob
import argparse

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

import kp_config
from kp_config import MP_KP_INDEX, TARGET_KP_NAMES, SKELETON

CAM_DIR = kp_config.DATASET_SOURCES['cam']['img_dir']
OUT_ANN = kp_config.DATASET_SOURCES['cam']['ann_file']


def save_coco(path, images, anns):
    coco = {
        'info': {'description': 'cam_data pose labels (MediaPipe generated)'},
        'licenses': [],
        'images': images,
        'annotations': anns,
        'categories': [{
            'supercategory': 'person', 'id': 1, 'name': 'person',
            'keypoints': list(TARGET_KP_NAMES), 'skeleton': SKELETON,
        }],
    }
    with open(path, 'w') as f:
        json.dump(coco, f)


def detect(pose, img_bgr):
    """MediaPipe 检测 4 关键点，返回像素坐标 (4,3) [x,y,vis]，失败返回 None"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    lm = res.pose_landmarks.landmark
    kps = np.zeros((4, 3), dtype=np.float32)
    for i, name in enumerate(TARGET_KP_NAMES):
        l = lm[MP_KP_INDEX[name]]
        kps[i] = [l.x * w, l.y * h, l.visibility]
    return kps


def is_upright(kps, min_vis):
    """几何判断姿态是否正向：至少3点可见且眼睛 y < 肩膀 y（避免误判上下颠倒）"""
    if kps is None:
        return False
    if int((kps[:, 2] >= min_vis).sum()) < 3:
        return False
    eye_y = (kps[0, 1] + kps[1, 1]) / 2
    shoulder_y = (kps[2, 1] + kps[3, 1]) / 2
    return eye_y < shoulder_y


def main():
    parser = argparse.ArgumentParser(description='为 cam_data 生成 MediaPipe 真值')
    parser.add_argument('--cam_dir', default=CAM_DIR)
    parser.add_argument('--out', default=OUT_ANN)
    parser.add_argument('--min_vis', type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(args.cam_dir, '*.jpg')))
    print(f'找到 {len(imgs)} 张 cam 图片')

    pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    coco_images, coco_anns = [], []
    img_id = 0
    ann_id = 0
    n_flip = 0
    n_fail = 0
    n_discard = 0

    for p in tqdm(imgs, desc='生成 cam 标注'):
        img = cv2.imread(p)
        if img is None:
            n_fail += 1
            continue
        try:
            kps_orig = detect(pose, img)
            if kps_orig is None or int((kps_orig[:, 2] >= args.min_vis).sum()) < 3:
                # MediaPipe 检测失败/关键点不足：跳过，保留图片不生成标注
                n_fail += 1
                continue
            eye_y = (kps_orig[0, 1] + kps_orig[1, 1]) / 2
            shoulder_y = (kps_orig[2, 1] + kps_orig[3, 1]) / 2
            if eye_y < shoulder_y:
                # 正向：保留并生成标注
                kps, flipped = kps_orig, False
            else:
                # 头朝下：直接删除该图片
                try:
                    os.remove(p)
                except OSError:
                    pass
                n_discard += 1
                continue
        except Exception:
            n_fail += 1
            continue

        kp_flat = []
        for i in range(4):
            x, y, vis = kps[i]
            v = 2 if vis >= args.min_vis else 0
            kp_flat.extend([float(x), float(y), int(v)])
        n_vis = sum(1 for i in range(4) if kp_flat[i * 3 + 2] == 2)
        if n_vis < 3:
            n_fail += 1
            continue

        xs = [kp_flat[i * 3] for i in range(4) if kp_flat[i * 3 + 2] == 2]
        ys = [kp_flat[i * 3 + 1] for i in range(4) if kp_flat[i * 3 + 2] == 2]
        bx, by = min(xs), min(ys)
        bw, bh = max(xs) - bx, max(ys) - by

        coco_images.append({
            'id': img_id, 'file_name': os.path.basename(p),
            'width': 320, 'height': 240,
        })
        coco_anns.append({
            'id': ann_id, 'image_id': img_id, 'category_id': 1,
            'keypoints': kp_flat, 'num_keypoints': n_vis,
            'bbox': [bx, by, bw, bh], 'area': float(bw * bh),
            'iscrowd': 0, 'flipped': flipped,
        })
        img_id += 1
        ann_id += 1

        if img_id % 100 == 0:
            save_coco(args.out, coco_images, coco_anns)

    pose.close()
    save_coco(args.out, coco_images, coco_anns)

    n = len(coco_anns)
    print(f'\n完成: 成功 {n}, 删除头朝下 {n_discard}, 失败 {n_fail}')
    print(f'输出: {args.out}')
    print('已删除头朝下图片, 保留的均为正向图, 标注用原图坐标(无翻转)。')


if __name__ == '__main__':
    main()
