"""
可视化 4 关键点 COCO 数据集
读取标注，抽样画关键点+骨架+图例，用于确认 custom/cam 标注是否正确。

用法:
    python3 visualize_data.py --annotation data/custom/annotations/person_keypoints_val.json \
        --images data/custom/images --output vis_custom --num_samples 20
    python3 visualize_data.py --annotation data/cam_data/annotations/person_keypoints.json \
        --images data/cam_data --output vis_cam --num_samples 20
"""
import os
import json
import random
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import kp_config
from kp_config import KP_COLORS, SKELETON, TARGET_KP_NAMES


def draw_keypoints(img, kps, vis_thr=2):
    """kps: (4,3) [x,y,v] 像素坐标"""
    for a, b in SKELETON:
        if kps[a][2] >= vis_thr and kps[b][2] >= vis_thr:
            cv2.line(img, (int(kps[a][0]), int(kps[a][1])),
                     (int(kps[b][0]), int(kps[b][1])), (255, 255, 255), 2)
    for i, (x, y, v) in enumerate(kps):
        if v >= vis_thr:
            cv2.circle(img, (int(x), int(y)), 8, KP_COLORS[i], -1)
            cv2.circle(img, (int(x), int(y)), 8, (255, 255, 255), 2)
            cv2.putText(img, TARGET_KP_NAMES[i][:3], (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def draw_legend(img):
    h, w = img.shape[:2]
    x0, y0 = w - 200, 15
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 185, y0 + 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, 'Keypoints', (x0 + 10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for i, name in enumerate(TARGET_KP_NAMES):
        y = y0 + 50 + i * 22
        cv2.circle(img, (x0 + 20, y), 7, KP_COLORS[i], -1)
        cv2.putText(img, name, (x0 + 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description='可视化 4 点 COCO 数据集')
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--flipped_field', action='store_true',
                        help='标注含 flipped 字段(cam)，读取时按需翻转图片')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with open(args.annotation) as f:
        coco = json.load(f)
    ann_map = {a['image_id']: a for a in coco['annotations']}

    imgs = coco['images']
    random.seed(42)
    samples = random.sample(imgs, min(args.num_samples, len(imgs)))
    print(f'数据集 {len(imgs)} 图, 可视化 {len(samples)} 张')

    n = 0
    for info in tqdm(samples, desc='可视化'):
        ann = ann_map.get(info['id'])
        if not ann:
            continue
        img = cv2.imread(os.path.join(args.images, info['file_name']))
        if img is None:
            continue
        if args.flipped_field and ann.get('flipped', False):
            img = cv2.flip(img, 0)
        kps = np.array(ann['keypoints']).reshape(-1, 3)
        img = draw_keypoints(img, kps)
        img = draw_legend(img)
        cv2.imwrite(os.path.join(args.output, f'vis_{info["id"]:06d}.jpg'), img)
        n += 1
    print(f'完成 {n} 张 -> {args.output}')


if __name__ == '__main__':
    main()
