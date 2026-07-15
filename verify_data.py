"""
MediaPipe 核验 custom 数据集真值
=============================
目的：经验性确认 custom 7点标注的真实关键点顺序。
  custom 的 categories.keypoints 声明 LitePose 序，但数据实际是 COCO 序（已几何验证）。
  本脚本用 MediaPipe 重新检测，对比两种索引假设的逐点距离，确认真实顺序，并检出异常样本。

两种假设（从 custom 7点 keypoints 数组中提取 4 目标点的索引）：
  A: COCO 序    [1,2,5,6]  (left_eye,right_eye,left_shoulder,right_shoulder)
  B: LitePose 序 [0,1,5,6]  (categories.keypoints 声称的顺序)

用法:
    python3 verify_data.py --num_samples 300
"""
import os
import json
import random
import argparse

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

import kp_config
from kp_config import MP_KP_INDEX, TARGET_KP_NAMES, KP_COLORS

CUSTOM_ANN = kp_config.DATASET_SOURCES['custom_val']['ann_file']
CUSTOM_IMG_DIR = kp_config.DATASET_SOURCES['custom_val']['img_dir']

HYPOTHESES = {
    'COCO_seq(A)': [1, 2, 5, 6],
    'LitePose_seq(B)': [0, 1, 5, 6],
}


def get_mp_keypoints(pose, img_bgr):
    """MediaPipe 检测 4 关键点，返回像素坐标 (4,3) [x,y,vis]，检测失败返回 None"""
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


def main():
    parser = argparse.ArgumentParser(description='MediaPipe 核验 custom 真值顺序')
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--num_vis', type=int, default=20)
    parser.add_argument('--out_dir', type=str, default='verify_output')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(CUSTOM_ANN) as f:
        coco = json.load(f)
    img_info = {im['id']: im for im in coco['images']}
    ann_map = {}
    for ann in coco['annotations']:
        ann_map.setdefault(ann['image_id'], []).append(ann)

    img_ids = [iid for iid, a in ann_map.items() if len(a) == 1]  # 仅单人
    random.seed(42)
    samples = random.sample(img_ids, min(args.num_samples, len(img_ids)))
    print(f'抽样 {len(samples)} 张单人图核验...')

    pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    win_counts = {k: 0 for k in HYPOTHESES}
    dist_summary = {k: [] for k in HYPOTHESES}
    anomalies = []
    vis_count = 0
    valid = 0

    for img_id in tqdm(samples, desc='核验'):
        ann = ann_map[img_id][0]
        info = img_info[img_id]
        img_path = os.path.join(CUSTOM_IMG_DIR, info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        mp_kps = get_mp_keypoints(pose, img)
        if mp_kps is None:
            continue
        vis = mp_kps[:, 2] >= 0.5
        if vis.sum() < 3:
            continue
        valid += 1

        custom_kps = np.array(ann['keypoints']).reshape(-1, 3)  # (7,3)

        dists = {}
        for name, idx in HYPOTHESES.items():
            extracted = custom_kps[idx]  # (4,3) 像素
            d = np.linalg.norm(extracted[:, :2] - mp_kps[:, :2], axis=1)
            d[~vis] = np.nan
            dists[name] = float(np.nanmedian(d))
            dist_summary[name].append(dists[name])

        winner = min(dists, key=dists.get)
        win_counts[winner] += 1
        if dists[winner] > 50:
            anomalies.append((img_id, info['file_name'], dists[winner]))

        if vis_count < args.num_vis:
            vis_img = img.copy()
            for i, idx in enumerate(HYPOTHESES[winner]):
                x, y = int(custom_kps[idx, 0]), int(custom_kps[idx, 1])
                cv2.circle(vis_img, (x, y), 8, KP_COLORS[i], -1)
                cv2.putText(vis_img, 'C', (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, KP_COLORS[i], 2)
            for i in range(4):
                if vis[i]:
                    x, y = int(mp_kps[i, 0]), int(mp_kps[i, 1])
                    cv2.circle(vis_img, (x, y), 8, (0, 255, 0), -1)
                    cv2.putText(vis_img, 'M', (x + 10, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.out_dir, f'vis_{img_id:06d}.jpg'), vis_img)
            vis_count += 1

    pose.close()

    print('\n========== 核验结果 ==========')
    print(f'有效样本: {valid}')
    for name in HYPOTHESES:
        c = win_counts[name]
        med = float(np.median(dist_summary[name])) if dist_summary[name] else float('nan')
        print(f'  {name}: 距离更小 {c} 张, 中位距离中位数 {med:.1f}px')
    winner = max(win_counts, key=win_counts.get)
    print(f'\n-> 确认 custom 真实顺序: {winner}, 提取索引 {HYPOTHESES[winner]}')
    print(f'异常样本(距离>50px): {len(anomalies)} ({100*len(anomalies)/max(valid,1):.1f}%)')
    for img_id, fn, d in anomalies[:10]:
        print(f'  {fn}: {d:.1f}px')
    print(f'\n可视化: {args.out_dir}/ ({vis_count} 张, 绿M=MediaPipe 彩C=custom标注)')

    if 'A' in winner:
        print('\n[结论] custom 数据为 COCO 序, kp_config.py 中 DATASET_KP_INDEX["custom"]=[1,2,5,6] 正确。')
    else:
        print('\n[警告] LitePose 序假设胜出, 需检查 kp_config.py 索引!')


if __name__ == '__main__':
    main()
