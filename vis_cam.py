"""最新浮点模型(best.pth, 6点)可视化 cam_data: 预测(红圈+骨架+conf) vs 真值(绿×)。"""
import os
import json
import numpy as np
import torch
import cv2

import kp_config
from kp_config import (IMG_MEAN, IMG_STD, IMG_WIDTH, IMG_HEIGHT,
                       SKELETON, DATASET_SOURCES)
from model import PoseNet
from dataset import center_crop_resize

device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = PoseNet().to(device); m.eval()
sd = torch.load('checkpoints/best.pth', map_location=device)
m.load_state_dict(sd['model'] if 'model' in sd else sd)

cam = DATASET_SOURCES['cam']
img_dir = cam['img_dir']
coco = json.load(open(cam['ann_file']))
id2ann = {a['image_id']: a for a in coco['annotations']}

os.makedirs('output/vis_cam', exist_ok=True)
N = 60
done = 0
for im in coco['images']:
    if done >= N:
        break
    ann = id2ann.get(im['id'])
    if ann is None:
        continue
    img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, im['file_name'])), cv2.COLOR_BGR2RGB)
    inp, cx, cy, cw, ch = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
    x = inp.astype(np.float32) / 255.0
    x = (x - np.array(IMG_MEAN, np.float32)) / np.array(IMG_STD, np.float32)
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        pk, conf = m.decode(m(xt))
    pk = pk[0].cpu().numpy()
    conf = conf[0].cpu().numpy()
    # 真值 原图像素 -> inp(320x240) 坐标(与 dataset.py 一致的 crop+resize 变换)
    kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32)
    kp[:, 0] = (kp[:, 0] - cx) * (IMG_WIDTH / cw)
    kp[:, 1] = (kp[:, 1] - cy) * (IMG_HEIGHT / ch)

    vis = cv2.cvtColor(inp, cv2.COLOR_RGB2BGR).copy()
    for a, b in SKELETON:
        if conf[a] > 0.3 and conf[b] > 0.3:
            cv2.line(vis, (int(pk[a, 0] * 320), int(pk[a, 1] * 240)),
                     (int(pk[b, 0] * 320), int(pk[b, 1] * 240)), (255, 255, 255), 1)
    for i in range(6):
        px, py = int(pk[i, 0] * 320), int(pk[i, 1] * 240)
        cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)              # 预测 红
        cv2.putText(vis, f'{conf[i]:.2f}', (px + 6, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # conf 黄
        if kp[i, 2] >= 1:                                            # 真值 绿×
            cv2.drawMarker(vis, (int(kp[i, 0]), int(kp[i, 1])),
                           (0, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.putText(vis, 'red=pred  yellow=conf  greenX=gt', (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(os.path.join('output/vis_cam', im['file_name']), vis)
    done += 1

print(f'完成 {done} 张 -> output/vis_cam/  (红圈=预测  黄字=conf  绿×=真值)')
print('关键点顺序: 左眼/右眼/左耳/右耳/左肩/右肩')
