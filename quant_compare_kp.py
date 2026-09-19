"""量化损失归因：分关键点 PCK + 量化前后 argmax 位置漂移 + 逐层误差报告。
回答'左肩量化为什么掉点'：机制(峰值漂移) + 元凶层(逐层误差)。"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import kp_config
from kp_config import PCK_THRESHOLD_RATIO, IMG_WIDTH, IMG_HEIGHT, TARGET_KP_NAMES
from model import PoseNet
from dataset import PoseDataset

TH = PCK_THRESHOLD_RATIO * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PoseNet().to(device)
model.eval()
sd = torch.load('checkpoints/best.pth', map_location=device)
model.load_state_dict(sd['model'] if 'model' in sd else sd)

ds = PoseDataset('cam', training=False)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import TorchExecutor

calib_loader = DataLoader(PoseDataset('cam', training=False), batch_size=1,
                          shuffle=False, num_workers=0)
print('量化中 (calib_steps=32, error_report=True)...')
quant_graph = espdl_quantize_onnx(
    onnx_import_file='output/current.onnx',
    espdl_export_file='output/_qcmp.espdl',
    calib_dataloader=calib_loader, calib_steps=32,
    input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3', num_of_bits=8,
    collate_fn=lambda b: b[0], dispatching_override=None, device=device,
    error_report=True, skip_export=True, export_test_values=False, verbose=0,
)
executor = TorchExecutor(quant_graph)
print('量化完成, 开始 cam 全量推理对比\n')

f_correct = np.zeros(6); q_correct = np.zeros(6); total = np.zeros(6)
f_dist = [[] for _ in range(6)]; q_dist = [[] for _ in range(6)]
shift = [[] for _ in range(6)]  # 量化引入的 argmax 漂移(px)

with torch.no_grad():
    for img, hm, kps in loader:
        target = kps[0, :, :2].numpy()
        vis = (kps[0, :, 2] >= 1).numpy()
        pf, _ = model.decode(model(img.to(device)))
        pf = pf[0].cpu().numpy()
        out = executor.forward(img.to(device))
        out = out[0] if isinstance(out, (list, tuple)) else out
        out = torch.as_tensor(out).detach().cpu().float()
        if out.dim() == 3:
            out = out.unsqueeze(0)
        pq, _ = model.decode(out)
        pq = pq[0].numpy()
        for i in range(6):
            if vis[i]:
                df = ((pf[i, 0] - target[i, 0]) * IMG_WIDTH) ** 2 + ((pf[i, 1] - target[i, 1]) * IMG_HEIGHT) ** 2
                df = df ** 0.5
                dq = ((pq[i, 0] - target[i, 0]) * IMG_WIDTH) ** 2 + ((pq[i, 1] - target[i, 1]) * IMG_HEIGHT) ** 2
                dq = dq ** 0.5
                f_correct[i] += df < TH
                q_correct[i] += dq < TH
                total[i] += 1
                f_dist[i].append(df); q_dist[i].append(dq)
                sx = (pf[i, 0] - pq[i, 0]) * IMG_WIDTH
                sy = (pf[i, 1] - pq[i, 1]) * IMG_HEIGHT
                shift[i].append((sx ** 2 + sy ** 2) ** 0.5)

print(f'{"关键点":<16}{"浮点PCK":<10}{"量化PCK":<10}{"损失":<8}{"浮点偏差":<10}{"量化偏差":<10}{"漂移中位":<10}{"漂移90%"}')
for i, name in enumerate(TARGET_KP_NAMES):
    fp = f_correct[i] / total[i]; qp = q_correct[i] / total[i]
    s = np.array(shift[i])
    print(f'{name:<16}{fp:<10.3f}{qp:<10.3f}{fp-qp:<+8.3f}{np.median(f_dist[i]):<10.1f}'
          f'{np.median(q_dist[i]):<10.1f}{np.median(s):<10.2f}{np.percentile(s, 90):.2f}')
