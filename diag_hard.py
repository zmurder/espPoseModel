"""
诊断困难样本：打印模型预测坐标 vs MediaPipe(complexity=2,同训练真值)坐标
判断右肩严重偏差是: 左右搞反 / 定位飞 / MediaPipe(0)假偏差
"""
import os
import numpy as np
import torch
import cv2
import mediapipe as mp

import kp_config
from kp_config import MP_KP_INDEX, TARGET_KP_NAMES, IMG_MEAN, IMG_STD, IMG_WIDTH, IMG_HEIGHT
from model import PoseNet
from dataset import center_crop_resize


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PoseNet(num_keypoints=4).to(device)
    sd = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval()
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)

    paths = [
        'test_img/2.jpg',
        'test_img_cam/image_20260427_211416_952.jpg',
        'test_img_cam/image_20260427_211429_666.jpg',
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp, cx, cy, cw, ch = center_crop_resize(rgb, IMG_WIDTH, IMG_HEIGHT)
        x = inp.astype(np.float32) / 255.0
        x = (x - np.array(IMG_MEAN, dtype=np.float32)) / np.array(IMG_STD, dtype=np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            hm = model(x)
        pk, conf = model.decode(hm)
        pk = pk[0].cpu().numpy()
        conf = conf[0].cpu().numpy()
        po = np.zeros_like(pk)
        po[:, 0] = pk[:, 0] * cw + cx
        po[:, 1] = pk[:, 1] * ch + cy

        res = pose.process(rgb)
        print(f'\n=== {os.path.basename(path)} (原图 {w}x{h}, crop x={cx} y={cy} w={cw} h={ch}) ===')
        if not res.pose_landmarks:
            print('  MediaPipe 未检测到人')
            continue
        lm = res.pose_landmarks.landmark
        mk = np.zeros((4, 2))
        mv = np.zeros(4)
        for i, name in enumerate(TARGET_KP_NAMES):
            l = lm[MP_KP_INDEX[name]]
            mk[i] = [l.x * w, l.y * h]
            mv[i] = l.visibility
        print(f'  {"关键点":<16}{"模型(原图px)":<20}{"conf":<8}{"MP(原图px)":<20}{"MPvis":<8}{"偏差px"}')
        for i, name in enumerate(TARGET_KP_NAMES):
            d = np.linalg.norm(po[i] - mk[i])
            print(f'  {name:<16}{str(po[i].astype(int)):<20}{conf[i]:<8.3f}'
                  f'{str(mk[i].astype(int)):<20}{mv[i]:<8.3f}{d:.1f}')
        # 检查左右肩是否搞反
        ls_pred, rs_pred = po[2], po[3]
        ls_mp, rs_mp = mk[2], mk[3]
        d_swap = np.linalg.norm(ls_pred - rs_mp) + np.linalg.norm(rs_pred - ls_mp)
        d_norm = np.linalg.norm(ls_pred - ls_mp) + np.linalg.norm(rs_pred - rs_mp)
        print(f'  [左右肩搞反检查] 交换后总偏差={d_swap:.0f}px, 不交换={d_norm:.0f}px '
              f'-> {"疑似搞反!" if d_swap < d_norm * 0.6 else "未搞反"}')

    pose.close()


if __name__ == '__main__':
    main()
