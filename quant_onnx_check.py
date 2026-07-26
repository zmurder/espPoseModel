"""测三种 int16 配置在 ESP32 测试图 + cam 整体：
  纯int8 / head-only(Conv_514+conv2d_32) / stem+head
确认 head-only 既救敏感左肩、又不损害整体、且 int16 层最少。"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import TargetPlatform, TorchExecutor

import kp_config
from kp_config import IMG_MEAN, IMG_STD, IMG_WIDTH, IMG_HEIGHT, PCK_THRESHOLD_RATIO, TARGET_KP_NAMES
from model import PoseNet
from dataset import PoseDataset, center_crop_resize
from quant_fix import build_calib

TH = PCK_THRESHOLD_RATIO * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIGS = [
    ('纯int8', None),
    ('head-only', {'node_Conv_514': TargetPlatform.ESPDL_S3_INT16.value,
                   'node_conv2d_32': TargetPlatform.ESPDL_S3_INT16.value}),
    ('stem+head', {'node_Conv_452': TargetPlatform.ESPDL_S3_INT16.value,
                   'node_Conv_514': TargetPlatform.ESPDL_S3_INT16.value,
                   'node_conv2d_32': TargetPlatform.ESPDL_S3_INT16.value}),
]


def quant_graph(disp):
    cl = DataLoader(build_calib('cam'), batch_size=1, shuffle=False, num_workers=0)
    return espdl_quantize_onnx(
        onnx_import_file='output/pose_model_6kp.onnx', espdl_export_file='output/_qo.espdl',
        calib_dataloader=cl, calib_steps=64, input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3',
        num_of_bits=8, collate_fn=lambda b: b, dispatching_override=disp, device=device,
        error_report=False, skip_export=True, export_test_values=False, verbose=0)


print('量化三种配置...')
execs = [(name, TorchExecutor(quant_graph(disp))) for name, disp in CONFIGS]
m = PoseNet().to(device); m.eval()
sd = torch.load('checkpoints/best.pth', map_location=device)
m.load_state_dict(sd['model'] if 'model' in sd else sd)

# --- 单图 320240 ---
img = cv2.cvtColor(cv2.imread('test_img/320240.jpg'), cv2.COLOR_BGR2RGB) if False else None
import cv2
img = cv2.cvtColor(cv2.imread('test_img/320240.jpg'), cv2.COLOR_BGR2RGB)
inp, *_ = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
x = inp.astype(np.float32) / 255.0
x = (x - np.array(IMG_MEAN, np.float32)) / np.array(IMG_STD, np.float32)
xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()

def decode_one(executor):
    with torch.no_grad():
        out = executor.forward(xt)
        out = out[0] if isinstance(out, (list, tuple)) else out
        hm = torch.as_tensor(out).detach().cpu().float()[0].numpy()
    f = hm.reshape(4, -1)
    return f.argmax(1) % 160, f.argmax(1) // 120, f.max(1)

with torch.no_grad():
    hmf = m(xt)[0].cpu().numpy()
ff = hmf.reshape(4, -1)
cf, rf, mvf = ff.argmax(1) % 160, ff.argmax(1) // 120, ff.max(1)
print(f'\n=== 单图 320240 左肩(浮点 hm=121,88 c=0.841; ESP32 hm=33,95 c=0.266) ===')
print(f'{"配置":<12}{"左肩 hm":<16}{"左肩 conf":<12}{"右肩 conf"}')
for name, ex in execs:
    c, r, mv = decode_one(ex)
    print(f'{name:<12}({c[2]},{r[2]})       {mv[2]:<12.3f}{mv[3]:.3f}')

# --- cam 整体 ---
el = DataLoader(PoseDataset('cam', training=False), batch_size=1, shuffle=False, num_workers=0)
print(f'\n=== cam 整体 ===')
print(f'{"配置":<12}{"左肩conf≥0.6":<14}{"右肩conf≥0.6":<14}{"左肩PCK":<10}{"右肩PCK":<10}{"总PCK"}')
for name, ex in execs:
    qc = np.zeros(6); tot = np.zeros(6); qconf = [[] for _ in range(6)]
    with torch.no_grad():
        for im, hm, kps in el:
            tgt = kps[0, :, :2].numpy(); vis = (kps[0, :, 2] >= 1).numpy()
            out = ex.forward(im.to(device)); out = out[0] if isinstance(out, (list, tuple)) else out
            out = torch.as_tensor(out).detach().cpu().float()
            if out.dim() == 3:
                out = out.unsqueeze(0)
            pq, pconf = m.decode(out); pq = pq[0].numpy(); pconf = pconf[0].numpy()
            for i in range(6):
                if vis[i]:
                    dq = (((pq[i, 0] - tgt[i, 0]) * IMG_WIDTH) ** 2 + ((pq[i, 1] - tgt[i, 1]) * IMG_HEIGHT) ** 2) ** 0.5
                    qc[i] += dq < TH; tot[i] += 1; qconf[i].append(pconf[i])
    c6 = [np.mean(np.array(qconf[i]) >= 0.6) * 100 for i in range(6)]
    print(f'{name:<12}{c6[2]:<14.2f}{c6[3]:<14.2f}{qc[2]/tot[2]:<10.3f}{qc[3]/tot[3]:<10.3f}{qc.sum()/tot.sum():.4f}')
