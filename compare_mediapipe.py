"""
test_img 上对比训练模型 vs MediaPipe
=================================
在 test_img 每张图上：模型推理 + MediaPipe 检测（修正索引 2,5,11,12），
在原图上并排画（模型蓝/MediaPipe 绿+骨架），打印逐点距离，判断效果是否可接受。

用法:
    python3 compare_mediapipe.py --model_path checkpoints/latest.pth
"""
import os
import argparse
import glob

import cv2
import numpy as np
import torch
import mediapipe as mp

import kp_config
from kp_config import (MP_KP_INDEX, TARGET_KP_NAMES, IMG_MEAN, IMG_STD,
                       IMG_WIDTH, IMG_HEIGHT, KP_COLORS, SKELETON)
from model import PoseNet
from dataset import center_crop_resize


def get_mp_kps(pose, img_bgr):
    """MediaPipe 检测 4 关键点，返回原图像素坐标 (4,2)，失败返回 None"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    lm = res.pose_landmarks.landmark
    kps = np.zeros((4, 2), dtype=np.float32)
    for i, name in enumerate(TARGET_KP_NAMES):
        l = lm[MP_KP_INDEX[name]]
        kps[i] = [l.x * w, l.y * h]
    return kps


def main():
    parser = argparse.ArgumentParser(description='test_img 对比 MediaPipe')
    parser.add_argument('--model_path', default='checkpoints/latest.pth')
    parser.add_argument('--img_dir', default='test_img')
    parser.add_argument('--output', default='comparison')
    parser.add_argument('--threshold', type=float, default=30.0, help='可接受平均距离(px)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PoseNet().to(device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval()

    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0)
    imgs = sorted(glob.glob(os.path.join(args.img_dir, '*.jpg')))
    print(f'对比 {len(imgs)} 张图片 (device={device})')

    ok_count = 0
    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp, crop_x, crop_y, crop_w, crop_h = center_crop_resize(rgb, IMG_WIDTH, IMG_HEIGHT)
        x = inp.astype(np.float32) / 255.0
        x = (x - np.array(IMG_MEAN, dtype=np.float32)) / np.array(IMG_STD, dtype=np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            hm = model(x)
        pred_kps, conf = model.decode(hm)
        pred_kps = pred_kps[0].cpu().numpy()   # (4,2) 归一化(resized 320x240)
        conf = conf[0].cpu().numpy()
        # 模型坐标转回原图：归一化 -> 裁剪区域像素 -> 原图像素
        pred_orig = np.zeros_like(pred_kps)
        pred_orig[:, 0] = pred_kps[:, 0] * crop_w + crop_x
        pred_orig[:, 1] = pred_kps[:, 1] * crop_h + crop_y

        mp_kps = get_mp_kps(pose, img)

        vis = img.copy()
        # 模型(蓝)+骨架
        for a, b in SKELETON:
            cv2.line(vis, tuple(pred_orig[a].astype(int)),
                     tuple(pred_orig[b].astype(int)), (255, 0, 0), 2)
        for i in range(6):
            xp, yp = int(pred_orig[i, 0]), int(pred_orig[i, 1])
            cv2.circle(vis, (xp, yp), 8, (255, 0, 0), -1)
            cv2.putText(vis, f'{conf[i]:.2f}', (xp + 10, yp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # MediaPipe(绿)+骨架
        if mp_kps is not None:
            for a, b in SKELETON:
                cv2.line(vis, tuple(mp_kps[a].astype(int)),
                         tuple(mp_kps[b].astype(int)), (0, 255, 0), 2)
            for i in range(6):
                cv2.circle(vis, (int(mp_kps[i, 0]), int(mp_kps[i, 1])), 8, (0, 255, 0), -1)
            dist = np.linalg.norm(pred_orig - mp_kps, axis=1)
            print(f'{os.path.basename(img_path)}: 平均距离 {dist.mean():.1f}px '
                  f'逐点 {[round(d, 1) for d in dist]}')
            if dist.mean() < args.threshold:
                ok_count += 1
        else:
            print(f'{os.path.basename(img_path)}: MediaPipe 未检测到人')

        cv2.putText(vis, 'Blue=Model  Green=MediaPipe', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(args.output, os.path.basename(img_path)), vis)

    pose.close()
    print(f'\n完成: {ok_count}/{len(imgs)} 张平均距离 < {args.threshold}px')
    print(f'可视化: {args.output}/')
    if ok_count >= max(1, len(imgs)) * 0.6:
        print('[结论] 效果可接受, 可继续 Phase 2 完整训练')
    else:
        print('[警告] 效果不佳, 请分析: 数据顺序/增强强度/学习率/heatmap是否收敛')


if __name__ == '__main__':
    main()
