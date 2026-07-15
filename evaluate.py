"""
PCK@0.1 全量评测（浮点模型）
=========================
target 用原始归一化坐标（非 heatmap argmax，避免离散化误差）。总体 + 4 关键点分别统计。

用法:
    python3 evaluate.py --model_path checkpoints/best.pth --source custom_val
    python3 evaluate.py --model_path checkpoints/best.pth --source coco_val
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import kp_config
from kp_config import PCK_THRESHOLD_RATIO, IMG_WIDTH, IMG_HEIGHT, TARGET_KP_NAMES
from model import PoseNet
from dataset import PoseDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    kp_correct = np.zeros(4)
    kp_total = np.zeros(4)
    threshold = PCK_THRESHOLD_RATIO * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5
    print(f'PCK 阈值: {threshold:.1f}px (0.1 * 对角线)')

    for img, hm, kps in loader:
        img = img.to(device)
        pred_hm = model(img)
        pred_kps, _ = model.decode(pred_hm)
        pred_kps = pred_kps.cpu().numpy()
        target = kps[..., :2].numpy()
        vis = kps[..., 2].numpy() >= 1
        dx = (pred_kps[..., 0] - target[..., 0]) * IMG_WIDTH
        dy = (pred_kps[..., 1] - target[..., 1]) * IMG_HEIGHT
        dist = (dx ** 2 + dy ** 2) ** 0.5
        correct_mask = (dist < threshold) & vis
        correct += int(correct_mask.sum())
        total += int(vis.sum())
        kp_correct += correct_mask.sum(axis=0)
        kp_total += vis.sum(axis=0)

    pck = correct / max(total, 1)
    per_kp = kp_correct / np.clip(kp_total, 1, None)
    return pck, per_kp, total


def main():
    parser = argparse.ArgumentParser(description='PCK 评测')
    parser.add_argument('--model_path', default='checkpoints/best.pth')
    parser.add_argument('--source', default='custom_val',
                        choices=['custom_val', 'coco_val', 'cam'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PoseNet(num_keypoints=4).to(device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    print(f'模型: {args.model_path} (device={device})')

    ds = PoseDataset(args.source, training=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4 if torch.cuda.is_available() else 0)

    pck, per_kp, total = evaluate(model, loader, device)
    print(f'\n== {args.source} ({total} 关键点) ==')
    print(f'PCK@0.1 = {pck:.4f} ({pck*100:.2f}%)')
    for i, name in enumerate(TARGET_KP_NAMES):
        print(f'  {name}: {per_kp[i]:.4f}')


if __name__ == '__main__':
    main()
