"""
诊断坐姿指标误差：双肩/双眼连线倾斜角、肩宽/眼距、偏差方向
判断模型对"坐姿判断(倾斜度/距离)"是否真正可用。
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import kp_config
from kp_config import IMG_WIDTH, IMG_HEIGHT
from model import PoseNet
from dataset import PoseDataset


def ang(a, b):
    return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))


def stat(name, arr, fmt='{:.2f}'):
    arr = np.array(arr)
    print(f'{name}: 中位{fmt.format(np.median(arr))} 均值{fmt.format(arr.mean())} '
          f'90%分位{fmt.format(np.percentile(arr, 90))}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PoseNet(num_keypoints=4).to(device)
    sd = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval()

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='cam')
    args = p.parse_args()

    ds = PoseDataset(args.source, training=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        num_workers=4 if device == 'cuda' else 0)

    sh_ang_err, eye_ang_err = [], []
    width_ratio, eye_dist_ratio = [], []
    sh_off, eye_off = [], []

    with torch.no_grad():
        for img, hm, kps in loader:
            pred_hm = model(img.to(device))
            pred_kps, _ = model.decode(pred_hm)
            pred_kps = pred_kps.cpu().numpy()
            target = kps[..., :2].numpy()
            vis = kps[..., 2].numpy() >= 1
            S = np.array([IMG_WIDTH, IMG_HEIGHT])
            for b in range(pred_kps.shape[0]):
                p = pred_kps[b] * S
                t = target[b] * S
                for i in range(4):
                    if vis[b, i]:
                        if i < 2:
                            eye_off.append(p[i] - t[i])
                        else:
                            sh_off.append(p[i] - t[i])
                if vis[b, 2] and vis[b, 3]:
                    sh_ang_err.append(abs(ang(t[2], t[3]) - ang(p[2], p[3])))
                    w = np.linalg.norm(t[3] - t[2])
                    if w > 1:
                        width_ratio.append(np.linalg.norm(p[3] - p[2]) / w)
                if vis[b, 0] and vis[b, 1]:
                    eye_ang_err.append(abs(ang(t[0], t[1]) - ang(p[0], p[1])))
                    d = np.linalg.norm(t[1] - t[0])
                    if d > 1:
                        eye_dist_ratio.append(np.linalg.norm(p[1] - p[0]) / d)

    print(f'\n== {args.source} 坐姿指标诊断 ==')
    stat('双肩倾斜角误差(°)', sh_ang_err)
    stat('双眼倾斜角误差(°)', eye_ang_err)
    stat('肩宽比(预/真)', width_ratio, '{:.3f}')
    stat('眼距比(预/真)', eye_dist_ratio, '{:.3f}')
    sh_off = np.array(sh_off)
    eye_off = np.array(eye_off)
    print(f'肩膀偏差中位: dx={np.median(sh_off[:, 0]):.1f}px dy={np.median(sh_off[:, 1]):.1f}px')
    print(f'眼睛偏差中位: dx={np.median(eye_off[:, 0]):.1f}px dy={np.median(eye_off[:, 1]):.1f}px')


if __name__ == '__main__':
    main()
