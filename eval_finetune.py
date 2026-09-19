"""fine-tune 效果对比: 旧模型(best.pth) vs 新模型(latest.pth) 在各数据源的 PCK + conf。
决策依据: 新数据域提升 & 旧域不退 -> 用新模型; 否则保持旧模型。"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import kp_config
from kp_config import (PCK_THRESHOLD_RATIO, IMG_WIDTH, IMG_HEIGHT, TARGET_KP_NAMES,
                       DATASET_SOURCES)
from model import PoseNet
from dataset import PoseDataset

TH = PCK_THRESHOLD_RATIO * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SOURCES = ['cam_data_20260816', 'cam']  # cam_data_20260816=新数据, cam=旧cam

models = {}
for name, path in [('旧(best)', 'checkpoints/best.pth'), ('新(latest)', 'checkpoints/latest.pth')]:
    m = PoseNet().to(device); m.eval()
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd['model'] if 'model' in sd else sd)
    models[name] = m

for src in SOURCES:
    loader = DataLoader(PoseDataset(src, training=False), batch_size=1,
                        shuffle=False, num_workers=0)
    print(f'\n== {src} ({len(loader)} 张) ==')
    print(f'{"模型":<12}{"总PCK":<10}{"眼PCK":<8}{"耳PCK":<8}{"肩PCK":<8}{"conf≥0.6":<10}{"平均conf"}')
    for name, m in models.items():
        qc = np.zeros(6); tot = np.zeros(6); confs = [[] for _ in range(6)]
        with torch.no_grad():
            for img, hm, kps in loader:
                vis = (kps[0, :, 2] >= 1).numpy()
                pk, pc = m.decode(m(img.to(device)))
                pk = pk[0].cpu().numpy(); pc = pc[0].cpu().numpy()
                tgt = kps[0, :, :2].numpy()
                for i in range(6):
                    if vis[i]:
                        d = (((pk[i, 0] - tgt[i, 0]) * IMG_WIDTH) ** 2 +
                             ((pk[i, 1] - tgt[i, 1]) * IMG_HEIGHT) ** 2) ** 0.5
                        qc[i] += d < TH; tot[i] += 1
                        confs[i].append(pc[i])
        allc = np.concatenate(confs)
        eye = qc[:2].sum() / tot[:2].sum(); ear = qc[2:4].sum() / tot[2:4].sum()
        sh = qc[4:].sum() / tot[4:].sum()
        print(f'{name:<12}{qc.sum()/tot.sum():<10.4f}{eye:<8.3f}{ear:<8.3f}{sh:<8.3f}'
              f'{(allc>=0.6).mean()*100:<10.1f}{allc.mean():.3f}')
