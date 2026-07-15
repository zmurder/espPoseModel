#!/usr/bin/env python3
"""
可视化验证生成的COCO数据集

Usage:
    python visualize_extracted.py \
        --annotation ./data_custom/annotations/instances.json \
        --images ./data_custom/images \
        --output ./data_custom/vis
"""

import os
import json
import cv2
import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


# 关键点名称和颜色 (BGR格式)
KP_INFO = {
    'left_eye':     {'idx': 0, 'color': (0, 0, 255)},    # 红
    'right_eye':    {'idx': 1, 'color': (255, 0, 255)},  # 粉
    'left_ear':     {'idx': 2, 'color': (255, 0, 0)},    # 蓝
    'right_ear':    {'idx': 3, 'color': (0, 255, 255)},  # 青
    'nose':         {'idx': 4, 'color': (0, 255, 0)},    # 绿
    'left_shoulder':{'idx': 5, 'color': (0, 165, 255)},  # 橙
    'right_shoulder':{'idx': 6, 'color': (255, 165, 0)},  # 黄
}

# 连接线 - 已禁用
SKELETON = []


def draw_legend(img, kp_names):
    """在图片右上角绘制颜色含义（放大版）"""
    h, w = img.shape[:2]

    # 图例背景 - 放大一倍
    legend_x = w - 200
    legend_y = 15
    legend_h = 340
    legend_w = 185

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # 标题
    cv2.putText(img, "Legend", (legend_x + 10, legend_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 绘制每个关键点的颜色
    y_offset = legend_y + 70
    for name, info in kp_names.items():
        color = info['color']
        # 绘制小圆点 - 放大一倍
        cv2.circle(img, (legend_x + 25, y_offset), 12, color, -1)
        cv2.circle(img, (legend_x + 25, y_offset), 12, (255, 255, 255), 2)
        # 绘制文字 - 放大一倍
        display_name = name.replace('_', ' ').title()
        cv2.putText(img, display_name, (legend_x + 50, y_offset + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 44

    return img


def draw_keypoints(img, keypoints, kp_names, visibility_threshold=2):
    """在图像上绘制关键点（不画连线，放大版）"""
    h, w = img.shape[:2]

    # 绘制关键点 - 放大一倍
    for name, info in kp_names.items():
        kp = keypoints[info['idx']]
        x, y, v = kp

        if v >= visibility_threshold:
            x, y = int(x), int(y)
            color = info['color']
            cv2.circle(img, (x, y), 12, color, -1)
            cv2.circle(img, (x, y), 12, (255, 255, 255), 2)
            cv2.putText(img, name.replace('_', ' '), (x+15, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 添加图例
    img = draw_legend(img, kp_names)

    return img


def visualize_sample(img_path, keypoints, output_path=None):
    """可视化单个样本"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None

    # 绘制关键点
    img = draw_keypoints(img, keypoints, KP_INFO)

    # 添加信息 - 左下角（放大版）
    num_visible = sum(1 for kp in keypoints if kp[2] >= 2)
    h, w = img.shape[:2]
    text = f"Visible: {num_visible}/7"
    cv2.putText(img, text, (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if output_path:
        cv2.imwrite(output_path, img)

    return img


def main():
    parser = argparse.ArgumentParser(description="可视化COCO姿态数据集")
    parser.add_argument('--annotation', type=str, required=True,
                       help='COCO标注文件路径')
    parser.add_argument('--images', type=str, required=True,
                       help='图片目录路径')
    parser.add_argument('--output', type=str, required=True,
                       help='可视化输出目录')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='随机采样的样本数')
    parser.add_argument('--show_all', action='store_true',
                       help='可视化所有样本(可能很多)')
    parser.add_argument('--indices', type=str, default=None,
                       help='指定要可视化的图片ID，逗号分隔')

    args = parser.parse_args()

    # 加载标注
    with open(args.annotation, 'r') as f:
        coco = json.load(f)

    images = coco.get('images', [])
    annotations = coco.get('annotations', [])

    print(f"数据集: {len(images)} 图片, {len(annotations)} 标注")

    # 创建图片ID到标注的映射
    ann_map = {ann['image_id']: ann for ann in annotations}

    # 确定要可视化的图片
    if args.indices:
        target_ids = [int(x.strip()) for x in args.indices.split(',')]
        samples = [img for img in images if img['id'] in target_ids]
    elif args.show_all:
        samples = images
    else:
        samples = random.sample(images, min(args.num_samples, len(images)))

    print(f"将可视化 {len(samples)} 张图片")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 可视化
    success_count = 0
    for img_info in tqdm(samples, desc="可视化"):
        img_id = img_info['id']
        filename = img_info['file_name']

        # 查找标注
        ann = ann_map.get(img_id)
        if not ann:
            print(f"警告: 图片 {img_id} 没有标注")
            continue

        # 获取关键点
        keypoints = []
        kps = ann['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]
        for i in range(7):
            keypoints.append([kps[i*3], kps[i*3+1], kps[i*3+2]])

        # 读取并绘制
        img_path = os.path.join(args.images, filename)
        output_path = os.path.join(args.output, f"vis_{img_id:06d}_{filename}")

        result = visualize_sample(img_path, keypoints, output_path)
        if result is not None:
            success_count += 1

    print(f"\n完成! 成功可视化 {success_count}/{len(samples)} 张图片")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
