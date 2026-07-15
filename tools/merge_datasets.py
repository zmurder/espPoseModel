#!/usr/bin/env python3
"""
合并多个COCO格式的姿态检测数据集

Usage:
    python merge_datasets.py --datasets ./data,./data_custom --output ./data_merged

    # 验证合并结果
    python merge_datasets.py --datasets ./data,./data_custom --output ./data_merged --verify
"""

import os
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def load_coco_annotation(ann_path):
    """加载COCO标注文件"""
    with open(ann_path, 'r') as f:
        return json.load(f)


def merge_coco_annotations(datasets, output_dir, copy_images=True, val_split=0.1):
    """
    合并多个COCO数据集

    Args:
        datasets: 数据集路径列表 (逗号分隔)
        output_dir: 输出目录
        copy_images: 是否复制图片到输出目录
        val_split: 验证集比例（默认10%）
    """
    # 收集所有数据
    all_images = []
    all_annotations = []
    all_categories = None

    image_id_offset = 0
    ann_id_offset = 0

    dataset_stats = {}

    for dataset_path in datasets:
        dataset_path = dataset_path.strip()
        if not dataset_path:
            continue

        print(f"\n处理数据集: {dataset_path}")

        # 查找标注文件（COCO格式和自定义格式）
        ann_paths = [
            os.path.join(dataset_path, 'annotations.json'),
            os.path.join(dataset_path, 'annotations', 'person_keypoints_train2017.json'),
            os.path.join(dataset_path, 'annotations', 'person_keypoints_val2017.json'),
            os.path.join(dataset_path, 'annotations', 'person_keypoints.json'),
            os.path.join(dataset_path, 'person_keypoints_train.json'),
            os.path.join(dataset_path, 'person_keypoints_val.json'),
            os.path.join(dataset_path, 'person_keypoints.json'),
        ]

        ann_path = None
        for p in ann_paths:
            if os.path.exists(p):
                ann_path = p
                break

        if not ann_path:
            print(f"  警告: 未找到标注文件，跳过")
            dataset_stats[dataset_path] = {'images': 0, 'annotations': 0}
            continue

        coco = load_coco_annotation(ann_path)

        # 记录类别（使用第一个数据集的类别）
        if all_categories is None:
            all_categories = coco.get('categories', [])

        # 构建图像ID到文件名的映射
        image_info = {img['id']: img for img in coco['images']}

        # 重新映射ID
        images = coco.get('images', [])
        annotations = coco.get('annotations', [])

        # 数据清洗：只保留符合坐姿检测标准的样本
        # 关键点顺序（根据categories定义）: left_eye, right_eye, left_ear, right_ear, nose, left_shoulder, right_shoulder
        # COCO原始顺序: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4), left_shoulder(5), right_shoulder(6)
        filtered_images = []
        filtered_annotations = []
        filtered_stats = {'total': 0, '7_kp_visible': 0, 'has_ears': 0, 'no_ears': 0}

        for ann in annotations:
            filtered_stats['total'] += 1
            kps = ann['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]

            # 检查7个关键点是否都可见 (v >= 2 表示可见)
            visible_count = sum(1 for i in range(7) if kps[i*3+2] >= 2)

            if visible_count < 7:
                continue  # 跳过关键点不足7个的样本

            filtered_stats['7_kp_visible'] += 1

            # 检查耳朵是否可见 (左耳=索引2, 右耳=索引3 in our 7-kp order)
            # 在COCO顺序中，左耳=索引3，右耳=索引4
            left_ear_visible = kps[3*3+2] >= 2  # COCO: left_ear
            right_ear_visible = kps[4*3+2] >= 2  # COCO: right_ear

            if left_ear_visible and right_ear_visible:
                filtered_stats['has_ears'] += 1
                priority = 0  # 优先
            else:
                filtered_stats['no_ears'] += 1
                priority = 1  # 次优先

            filtered_annotations.append((priority, ann))
            filtered_images.append((priority, image_info[ann['image_id']]))

        # 按优先级排序（优先的在前）
        filtered_annotations.sort(key=lambda x: x[0])
        filtered_images.sort(key=lambda x: x[0])

        # 取前N个样本（避免数据集过大）
        max_samples = 50000  # 最多保留5万个样本
        filtered_annotations = [ann for _, ann in filtered_annotations[:max_samples]]
        filtered_images = [img for _, img in filtered_images[:max_samples]]

        print(f"  数据清洗: 总共{filtered_stats['total']}个样本, 7个关键点可见: {filtered_stats['7_kp_visible']}")
        print(f"  其中耳朵可见: {filtered_stats['has_ears']}, 耳朵不可见: {filtered_stats['no_ears']}")
        print(f"  最终保留: {len(filtered_annotations)} 个样本")

        # 更新image_id
        for img in filtered_images:
            img['id'] += image_id_offset
        for ann in filtered_annotations:
            ann['id'] += ann_id_offset
            ann['image_id'] += image_id_offset

        all_images.extend(filtered_images)
        all_annotations.extend(filtered_annotations)

        image_id_offset = max([img['id'] for img in all_images] or [0]) + 1
        ann_id_offset = max([ann['id'] for ann in all_annotations] or [0]) + 1

        # 复制图片（只复制过滤后的图片）
        if copy_images:
            img_dir = os.path.join(dataset_path, 'images')
            # 也尝试train2017目录（COCO格式）
            train2017_dir = os.path.join(dataset_path, 'train2017')
            if os.path.exists(train2017_dir):
                img_dir = train2017_dir
            if os.path.exists(img_dir):
                dst_dir = os.path.join(output_dir, 'train2017')
                os.makedirs(dst_dir, exist_ok=True)
                filtered_img_set = set(img['id'] for img in filtered_images)
                for img in tqdm(images, desc=f"  复制图片"):
                    if img['id'] not in filtered_img_set:
                        continue
                    src = os.path.join(img_dir, img['file_name'])
                    dst = os.path.join(dst_dir, img['file_name'])
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)

        dataset_stats[dataset_path] = {
            'images': len(filtered_images),
            'annotations': len(filtered_annotations)
        }

    # 保存
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    # 划分训练集和验证集 (90%/10%)
    from collections import defaultdict
    image_to_anns = defaultdict(list)
    for ann in all_annotations:
        image_to_anns[ann['image_id']].append(ann)

    all_image_ids = list(image_to_anns.keys())
    np.random.seed(42)
    np.random.shuffle(all_image_ids)

    val_split = 0.1
    val_image_ids = set(all_image_ids[:int(len(all_image_ids) * val_split)])
    train_image_ids = set(all_image_ids[int(len(all_image_ids) * val_split):])

    train_images = [img for img in all_images if img['id'] in train_image_ids]
    train_anns = [ann for ann in all_annotations if ann['image_id'] in train_image_ids]
    val_images = [img for img in all_images if img['id'] in val_image_ids]
    val_anns = [ann for ann in all_annotations if ann['image_id'] in val_image_ids]

    print(f"Train/Val split: {len(train_anns)} train, {len(val_anns)} val")

    def create_coco_format(images, annotations, categories):
        return {
            "info": {"description": "Merged pose dataset", "version": "1.0"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories or [],
        }

    # 保存完整标注
    merged_ann_path = os.path.join(output_dir, 'annotations', 'instances.json')
    with open(merged_ann_path, 'w') as f:
        json.dump(create_coco_format(all_images, all_annotations, all_categories), f, indent=2)

    # 保存训练集标注（兼容dataset.py的COCO格式）
    train_coco = create_coco_format(train_images, train_anns, all_categories)
    pk_train_path = os.path.join(output_dir, 'annotations', 'person_keypoints_train2017.json')
    with open(pk_train_path, 'w') as f:
        json.dump(train_coco, f, indent=2)

    # 保存验证集标注（兼容dataset.py的COCO格式）
    val_coco = create_coco_format(val_images, val_anns, all_categories)
    pk_val_path = os.path.join(output_dir, 'annotations', 'person_keypoints_val2017.json')
    with open(pk_val_path, 'w') as f:
        json.dump(val_coco, f, indent=2)

    # 合并后完整数据（包含train和val的组合）用于参考
    merged_path = os.path.join(output_dir, 'person_keypoints.json')
    with open(merged_path, 'w') as f:
        json.dump(create_coco_format(all_images, all_annotations, all_categories), f, indent=2)

    return train_coco, val_coco, dataset_stats


def verify_merged_dataset(merged_coco, output_dir):
    """验证合并后的数据集"""
    print("\n=== 数据集验证 ===")

    images = merged_coco.get('images', [])
    annotations = merged_coco.get('annotations', [])

    print(f"总图片数: {len(images)}")
    print(f"总标注数: {len(annotations)}")

    # 检查图片是否存在
    missing_images = []
    for img in tqdm(images, desc="检查图片"):
        img_path = os.path.join(output_dir, 'images', img['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img['file_name'])

    if missing_images:
        print(f"警告: {len(missing_images)} 张图片缺失")
    else:
        print("所有图片文件完整")

    # 检查标注与图片对应关系
    img_ids = set(img['id'] for img in images)
    ann_img_ids = set(ann['image_id'] for ann in annotations)
    orphan_anns = ann_img_ids - img_ids

    if orphan_anns:
        print(f"警告: {len(orphan_anns)} 个标注没有对应的图片")
    else:
        print("所有标注都有对应的图片")

    # 按图片统计关键点数量
    kp_counts = defaultdict(int)
    for ann in annotations:
        kp_counts[ann['num_keypoints']] += 1

    print("\n关键点数量分布:")
    for kp_count in sorted(kp_counts.keys()):
        print(f"  {kp_count} 个关键点: {kp_counts[kp_count]} 张图")

    # 计算平均关键点数
    if annotations:
        avg_kp = sum(ann['num_keypoints'] for ann in annotations) / len(annotations)
        print(f"\n平均关键点数: {avg_kp:.2f}")

    return len(missing_images) == 0 and len(orphan_anns) == 0


def analyze_dataset(merged_coco):
    """分析数据集详情"""
    annotations = merged_coco.get('annotations', [])

    # 关键点可见度统计
    kp_names = ['left_eye', 'right_eye', 'left_ear', 'right_ear', 'nose', 'left_shoulder', 'right_shoulder']

    print("\n=== 数据集分析 ===")

    # 每张图的关键点数量
    kp_counts = [ann['num_keypoints'] for ann in annotations]
    print(f"关键点数量: min={min(kp_counts)}, max={max(kp_counts)}, avg={sum(kp_counts)/len(kp_counts):.2f}")

    # 计算每个关键点的可见率
    print("\n各关键点可见率:")
    kp_visibility = {name: [] for name in kp_names}

    for ann in annotations:
        kps = ann['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]
        for i, name in enumerate(kp_names):
            v = kps[i * 3 + 2]  # visibility
            kp_visibility[name].append(1 if v >= 2 else 0)

    for name in kp_names:
        vis_rate = sum(kp_visibility[name]) / len(kp_visibility[name]) * 100
        print(f"  {name}: {vis_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="合并多个COCO姿态数据集")
    parser.add_argument('--datasets', type=str, required=True,
                       help='数据集路径，逗号分隔，如: ./data,./data_custom')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--copy_images', action='store_true', default=True,
                       help='复制图片到输出目录(默认True)')
    parser.add_argument('--no_copy_images', action='store_true',
                       help='不复制图片，仅合并标注')
    parser.add_argument('--verify', action='store_true',
                       help='验证合并后的数据集')

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]

    print(f"将合并 {len(datasets)} 个数据集:")
    for d in datasets:
        print(f"  - {d}")

    copy_images = args.copy_images and not args.no_copy_images

    # 合并
    train_coco, val_coco, stats = merge_coco_annotations(
        datasets,
        args.output,
        copy_images=copy_images
    )

    print("\n=== 合并统计 ===")
    total_images = 0
    total_annotations = 0
    for path, stat in stats.items():
        print(f"{path}: {stat['images']} 图片, {stat['annotations']} 标注")
        total_images += stat['images']
        total_annotations += stat['annotations']

    print(f"\n总计: {total_images} 图片, {total_annotations} 标注")
    print(f"训练集: {len(train_coco['annotations'])} 标注")
    print(f"验证集: {len(val_coco['annotations'])} 标注")

    # 分析
    analyze_dataset(train_coco)

    # 验证
    if args.verify:
        verify_merged_dataset(train_coco, args.output)

    print(f"\n输出目录: {args.output}")


if __name__ == "__main__":
    main()
