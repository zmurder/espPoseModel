#!/usr/bin/env python3
"""
从视频提取帧并使用MediaPipe生成COCO格式数据集（流式处理版）

特性：
- 流式处理：边读边检测边保存，内存占用低
- 均匀采样：按固定间隔采样，不判断相似度
- 严格过滤：只保留7个关键点都可见的帧
- 批量处理：支持多视频合并

Usage:
    # 处理目录下的所有视频
    python video_to_dataset.py --input_dir ./res/videos --output ./data/custom

    # 处理单个视频
    python video_to_dataset.py --input video.mp4 --output ./data/custom

    # 调整采样参数
    python video_to_dataset.py --input_dir ./res/videos --output ./data/custom \
        --sample_interval 30 --min_keypoints 5 --val_split 0.15
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from datetime import datetime

import mediapipe as mp
import numpy as np
from tqdm import tqdm

# MediaPipe Pose 关键点索引
MP_POSE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}

# MediaPipe → LitePoseNet 关键点映射
# LitePoseNet: [left_eye, right_eye, left_ear, right_ear, nose, left_shoulder, right_shoulder]
MP_TO_LITEPOSE = [
    2,   # left_eye → LEFT_EYE (MediaPipe index 2)
    5,   # right_eye → RIGHT_EYE (MediaPipe index 5)
    7,   # left_ear → LEFT_EAR (MediaPipe index 7)
    8,   # right_ear → RIGHT_EAR (MediaPipe index 8)
    0,   # nose → NOSE (MediaPipe index 0)
    11,  # left_shoulder → LEFT_SHOULDER (MediaPipe index 11)
    12,  # right_shoulder → RIGHT_SHOULDER (MediaPipe index 12)
]

LITEPOSE_NAMES = ['left_eye', 'right_eye', 'left_ear', 'right_ear', 'nose', 'left_shoulder', 'right_shoulder']


def filter_keypoints_quality(landmarks, mp_indices, min_visibility=0.5):
    """
    检查7个关键点的质量
    返回: (is_valid, visible_count, keypoints_list)
    """
    keypoints = []
    visible_count = 0

    for mp_idx in mp_indices:
        lm = landmarks[mp_idx]
        visibility = lm.visibility
        if visibility >= min_visibility:
            visible_count += 1
            keypoints.append({
                'x': lm.x,
                'y': lm.y,
                'visibility': visibility,
            })
        else:
            keypoints.append({
                'x': lm.x,
                'y': lm.y,
                'visibility': visibility,
            })

    # 只有7个关键点都可见才保留
    is_valid = visible_count == 7

    return is_valid, visible_count, keypoints


def process_video_streaming(video_path, output_dir, args, video_name, global_frame_id, annotationsAccumulator):
    """
    流式处理视频：边读边检测边保存，内存占用低

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        args: 命令行参数
        video_name: 视频名称（用于文件名前缀）
        global_frame_id: 全局帧ID计数器
        annotationsAccumulator: 外部的annotations列表引用，用于定期保存

    Returns:
        (annotations, new_global_frame_id)
    """
    # 确保images目录存在
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\n处理视频: {video_path}")
    print(f"  FPS: {fps:.1f}, 总帧数: {total_frames}, 时长: {duration:.1f}s")

    # 初始化MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )

    annotations = []
    frame_idx = 0
    saved_count = 0
    failed_count = 0
    not_enough_kp_count = 0

    pbar = tqdm(total=total_frames, desc="  流式处理", leave=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 均匀采样
        if frame_idx % args.sample_interval == 0:
            img_h, img_w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if not results.pose_landmarks:
                failed_count += 1
            else:
                landmarks = results.pose_landmarks.landmark

                # 检查7个关键点是否都可见
                is_valid, visible_count, keypoints = filter_keypoints_quality(
                    landmarks, MP_TO_LITEPOSE, args.min_visibility
                )

                if not is_valid:
                    not_enough_kp_count += 1
                else:
                    # 转换为像素坐标
                    keypoints_pixel = []
                    for kp in keypoints:
                        keypoints_pixel.append({
                            'x': kp['x'] * img_w,
                            'y': kp['y'] * img_h,
                            'visibility': kp['visibility'],
                        })

                    # 保存图片
                    img_name = f"{video_name}_frame_{global_frame_id:06d}.jpg"
                    img_path = os.path.join(images_dir, img_name)
                    cv2.imwrite(img_path, frame)

                    # 计算边界框
                    x_coords = [kp['x'] for kp in keypoints_pixel]
                    y_coords = [kp['y'] for kp in keypoints_pixel]
                    bbox_x = min(x_coords)
                    bbox_y = min(y_coords)
                    bbox_w = max(x_coords) - bbox_x
                    bbox_h = max(y_coords) - bbox_y

                    annotations.append({
                        'frame_id': global_frame_id,
                        'original_frame': frame_idx,
                        'filename': img_name,
                        'keypoints': keypoints_pixel,
                        'img_width': img_w,
                        'img_height': img_h,
                        'bbox': [bbox_x, bbox_y, bbox_w, bbox_h],
                    })

                    global_frame_id += 1
                    saved_count += 1

                    # 每100张图保存一次标注
                    if saved_count % 100 == 0:
                        save_partial_annotations(annotationsAccumulator, output_dir, Path(output_dir).name)

            pbar.set_postfix({
                'saved': saved_count,
                'no_pose': failed_count,
                'low_kp': not_enough_kp_count
            })

        pbar.update(1)
        frame_idx += 1

    cap.release()
    pose.close()
    pbar.close()

    print(f"  有效帧: {saved_count}, 无姿态: {failed_count}, 关键点不足: {not_enough_kp_count}")

    return annotations, global_frame_id


def save_partial_annotations(annotations, output_dir, dataset_name="custom"):
    """保存部分标注（每处理完一个视频后调用）"""
    if not annotations:
        return
    now = datetime.now()

    images = []
    for ann in annotations:
        images.append({
            "id": ann['frame_id'],
            "license": 1,
            "file_name": ann['filename'],
            "width": ann['img_width'],
            "height": ann['img_height'],
            "date_captured": now.strftime("%Y-%m-%d %H:%M:%S"),
            "coco_url": "",
            "flickr_url": "",
        })

    LITEPOSE_TO_COCO = [4, 0, 1, 2, 3, 5, 6]
    coco_annotations = []
    for ann in annotations:
        keypoints_flat = []
        kps = ann['keypoints']
        for coco_idx in LITEPOSE_TO_COCO:
            kp = kps[coco_idx]
            v = 2 if kp['visibility'] >= 0.5 else 1
            keypoints_flat.extend([kp['x'], kp['y'], v])

        coco_annotations.append({
            "id": ann['frame_id'],
            "image_id": ann['frame_id'],
            "category_id": 1,
            "keypoints": keypoints_flat,
            "num_keypoints": 7,
            "bbox": ann['bbox'],
            "area": ann['bbox'][2] * ann['bbox'][3],
            "iscrowd": 0,
        })

    categories = [{
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": LITEPOSE_NAMES,
        "skeleton": [
            [5, 6], [5, 2], [6, 3], [2, 0], [3, 1], [0, 1], [0, 4], [1, 4],
        ]
    }]

    coco_format = {
        "info": {"description": f"Custom pose dataset - {dataset_name}", "version": "1.0", "year": now.year, "date_created": now.strftime("%Y-%m-%d %H:%M:%S")},
        "licenses": [{"url": "", "id": 1, "name": "Custom License"}],
        "images": images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    partial_path = os.path.join(output_dir, 'annotations', 'partial.json')
    with open(partial_path, 'w') as f:
        json.dump(coco_format, f, indent=2)


def create_coco_annotation(annotations, output_dir, dataset_name="custom"):
    """创建COCO格式的annotation文件"""
    now = datetime.now()

    # 创建images列表
    images = []
    for ann in annotations:
        images.append({
            "id": ann['frame_id'],
            "license": 1,
            "file_name": ann['filename'],
            "width": ann['img_width'],
            "height": ann['img_height'],
            "date_captured": now.strftime("%Y-%m-%d %H:%M:%S"),
            "coco_url": "",
            "flickr_url": "",
        })

    # 创建annotations列表 (COCO person_keypoints格式)
    # COCO关键点顺序: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder]
    # 我们的顺序: [left_eye, right_eye, left_ear, right_ear, nose, left_shoulder, right_shoulder]
    # 所以 COCO[0]=LITEPOSE[4], COCO[1]=LITEPOSE[0], ... 映射: [4, 0, 1, 2, 3, 5, 6]
    LITEPOSE_TO_COCO = [4, 0, 1, 2, 3, 5, 6]

    coco_annotations = []
    for ann in annotations:
        # 转换为COCO关键点格式: [x1, y1, v1, x2, y2, v2, ...]
        # v=0: 不可见, v=1: 遮挡, v=2: 可见
        keypoints_flat = []
        kps = ann['keypoints']
        for coco_idx in LITEPOSE_TO_COCO:
            kp = kps[coco_idx]
            v = 2 if kp['visibility'] >= 0.5 else 1
            keypoints_flat.extend([kp['x'], kp['y'], v])

        coco_annotations.append({
            "id": ann['frame_id'],
            "image_id": ann['frame_id'],
            "category_id": 1,
            "keypoints": keypoints_flat,
            "num_keypoints": 7,
            "bbox": ann['bbox'],
            "area": ann['bbox'][2] * ann['bbox'][3],
            "iscrowd": 0,
        })

    # COCO categories
    categories = [{
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": LITEPOSE_NAMES,
        "skeleton": [
            [5, 6],  # shoulders
            [5, 2],  # left shoulder to left ear
            [6, 3],  # right shoulder to right ear
            [2, 0],  # left ear to left eye
            [3, 1],  # right ear to right eye
            [0, 1],  # eyes
            [0, 4],  # left eye to nose
            [1, 4],  # right eye to nose
        ]
    }]

    coco_format = {
        "info": {
            "description": f"Custom pose dataset from video - {dataset_name}",
            "url": "",
            "version": "1.0",
            "year": now.year,
            "contributor": "",
            "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [{"url": "", "id": 1, "name": "Custom License"}],
        "images": images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    # 保存
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    ann_path = os.path.join(output_dir, 'annotations', 'instances.json')
    with open(ann_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    # 同时保存person_keypoints格式（兼容dataset.py）
    pk_path = os.path.join(output_dir, 'annotations', 'person_keypoints.json')
    with open(pk_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    # 保存到根目录
    root_pk_path = os.path.join(output_dir, 'person_keypoints.json')
    with open(root_pk_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"  COCO标注已保存: {ann_path}")

    return coco_format


def main():
    parser = argparse.ArgumentParser(
        description="从视频生成COCO格式姿态数据集（流式处理版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理目录下的所有视频
  python video_to_dataset.py --input_dir ./res/videos --output ./data/custom

  # 处理单个视频
  python video_to_dataset.py --input video.mp4 --output ./data/custom

  # 调整采样参数
  python video_to_dataset.py --input_dir ./res/videos --output ./data/custom \\
      --sample_interval 30 --min_keypoints 5 --val_split 0.15
        """
    )

    parser.add_argument('--input', type=str, help='单个视频文件路径')
    parser.add_argument('--input_dir', type=str, help='包含多个视频的目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--sample_interval', type=int, default=10,
                       help='采样间隔(帧)，默认10')
    parser.add_argument('--min_keypoints', type=int, default=5,
                       help='最小关键点数（当前必须7个全可见），默认5')
    parser.add_argument('--min_visibility', type=float, default=0.5,
                       help='关键点最小可见度，默认0.5')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                       help='MediaPipe最小置信度，默认0.5')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='验证集划分比例，默认0.15')

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("必须指定 --input 或 --input_dir")

    # 确定要处理的视频
    if args.input:
        video_files = [args.input]
    else:
        video_dir = args.input_dir
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for f in os.listdir(video_dir):
            if any(f.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(video_dir, f))
        video_files.sort()

    if not video_files:
        print(f"错误: 在 {args.input_dir} 中未找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")
    print(f"采样参数:")
    print(f"  - sample_interval: {args.sample_interval}")
    print(f"  - min_visibility: {args.min_visibility}")
    print(f"  - min_confidence: {args.min_confidence}")
    print(f"  - val_split: {args.val_split}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 流式处理所有视频
    all_annotations = []
    global_frame_id = 0

    for video_path in video_files:
        video_name = Path(video_path).stem
        annotations, global_frame_id = process_video_streaming(
            video_path, args.output, args, video_name, global_frame_id, all_annotations
        )
        # 每个视频处理完后再保存一次
        all_annotations.extend(annotations)
        save_partial_annotations(all_annotations, args.output, Path(args.output).name)
        print(f"  [进度] 已保存 {len(all_annotations)} 张图片到标注文件")

    if not all_annotations:
        print("错误: 没有检测到有效姿态（7个关键点都可见）")
        return

    print(f"\n总计: 有效 {len(all_annotations)} 帧")

    # 创建COCO标注
    dataset_name = Path(args.output).name
    coco_format = create_coco_annotation(all_annotations, args.output, dataset_name)

    # 划分训练集和验证集并保存
    val_count = int(len(all_annotations) * args.val_split)
    train_annotations = all_annotations[val_count:]
    val_annotations = all_annotations[:val_count]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_annotations)} 帧")
    print(f"  验证集: {len(val_annotations)} 帧")

    # 保存划分后的标注
    train_images = [img for img in coco_format['images'] if img['id'] in [a['frame_id'] for a in train_annotations]]
    val_images = [img for img in coco_format['images'] if img['id'] in [a['frame_id'] for a in val_annotations]]

    def create_split_annotation(images, annotations, categories, split_name):
        split_format = coco_format.copy()
        split_format['images'] = images
        split_format['annotations'] = [ann for ann in coco_format['annotations'] if ann['id'] in [a['frame_id'] for a in annotations]]

        split_path = os.path.join(args.output, 'annotations', f'person_keypoints_{split_name}.json')
        with open(split_path, 'w') as f:
            json.dump(split_format, f, indent=2)
        return split_format

    create_split_annotation(train_images, train_annotations, coco_format['categories'], 'train')
    create_split_annotation(val_images, val_annotations, coco_format['categories'], 'val')

    print(f"\n完成!")
    print(f"  输出目录: {args.output}")
    print(f"  图片数量: {len(coco_format['images'])}")
    print(f"  标注数量: {len(coco_format['annotations'])}")


if __name__ == "__main__":
    main()
