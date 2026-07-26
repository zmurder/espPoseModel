"""统计 cam 上关键点置信度分层 PCK：验证 conf 阈值过滤能否去掉灾难样本"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import kp_config
from kp_config import IMG_WIDTH, IMG_HEIGHT, TARGET_KP_NAMES
from model import PoseNet
from dataset import PoseDataset


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PoseNet().to(device)
    sd = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval()
    ds = PoseDataset('cam', training=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    TH = 0.1 * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5

    # 每个关键点收集 (conf, dist, vis)
    data = {i: [] for i in range(6)}
    with torch.no_grad():
        for img, hm, kps in loader:
            pred_hm = model(img.to(device))
            pk, conf = model.decode(pred_hm)
            pk = pk.cpu().numpy()
            conf = conf.cpu().numpy()
            target = kps[..., :2].numpy()
            vis = kps[..., 2].numpy() >= 1
            dx = (pk[..., 0] - target[..., 0]) * IMG_WIDTH
            dy = (pk[..., 1] - target[..., 1]) * IMG_HEIGHT
            dist = (dx ** 2 + dy ** 2) ** 0.5
            for b in range(pk.shape[0]):
                for i in range(6):
                    if vis[b, i]:
                        data[i].append((conf[b, i], dist[b, i]))

    bins = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.1)]
    for i in range(6):
        arr = np.array(data[i])
        confs, dists = arr[:, 0], arr[:, 1]
        n = len(arr)
        overall = (dists < TH).mean()
        print(f'\n== {TARGET_KP_NAMES[i]} (n={n}, 整体PCK={overall:.3f}) ==')
        print(f'  {"conf区间":<14}{"数量":<8}{"占比":<8}{"PCK":<8}{"中位偏差px"}')
        for lo, hi in bins:
            m = (confs >= lo) & (confs < hi)
            if m.sum() == 0:
                continue
            pck = (dists[m] < TH).mean()
            print(f'  [{lo:.1f},{hi:.1f})    {m.sum():<8}{m.sum()/n:<8.2%}{pck:<8.3f}{np.median(dists[m]):<.1f}')
        for t in [0.3, 0.4]:
            m = confs >= t
            if m.sum():
                print(f'  conf>={t}: {m.sum()}/{n} ({m.sum()/n:.1%}), PCK={((dists[m]<TH).mean()):.3f}, 中位偏差={np.median(dists[m]):.1f}px')


if __name__ == '__main__':
    main()
