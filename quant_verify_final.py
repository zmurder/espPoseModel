"""cam 整体对比 纯int8 vs 混合精度(stem+head int16)：conf分布+PCK+坐姿。
确认混合精度整体优于纯int8，决定导出哪个 espdl。"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import TargetPlatform, TorchExecutor

import kp_config
from kp_config import IMG_WIDTH, IMG_HEIGHT, PCK_THRESHOLD_RATIO, TARGET_KP_NAMES
from model import PoseNet
from dataset import PoseDataset
from quant_fix import build_calib

TH = PCK_THRESHOLD_RATIO * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
S = np.array([IMG_WIDTH, IMG_HEIGHT])
MIX = ['node_Conv_452', 'node_Conv_514', 'node_conv2d_32']
disp_mix = {L: TargetPlatform.ESPDL_S3_INT16.value for L in MIX}


def ang(a, b):
    return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))


def quant_graph(disp):
    cl = DataLoader(build_calib('cam'), batch_size=1, shuffle=False, num_workers=0)
    return espdl_quantize_onnx(
        onnx_import_file='output/pose_model_6kp.onnx', espdl_export_file='output/_qv.espdl',
        calib_dataloader=cl, calib_steps=64, input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3',
        num_of_bits=8, collate_fn=lambda b: b, dispatching_override=disp, device=device,
        error_report=False, skip_export=True, export_test_values=False, verbose=0)


print('量化 纯int8 + 混合精度...')
ex8 = TorchExecutor(quant_graph(None))
exm = TorchExecutor(quant_graph(disp_mix))
m = PoseNet().to(device); m.eval()
sd = torch.load('checkpoints/best.pth', map_location=device)
m.load_state_dict(sd['model'] if 'model' in sd else sd)
el = DataLoader(PoseDataset('cam', training=False), batch_size=1, shuffle=False, num_workers=0)


def run(executor, label):
    qc = np.zeros(6); tot = np.zeros(6); qconf = [[] for _ in range(6)]
    shang = []
    with torch.no_grad():
        for img, hm, kps in el:
            tgt = kps[0, :, :2].numpy(); vis = (kps[0, :, 2] >= 1).numpy()
            pf, _ = m.decode(m(img.to(device))); pf = pf[0].cpu().numpy()
            out = executor.forward(img.to(device))
            out = out[0] if isinstance(out, (list, tuple)) else out
            out = torch.as_tensor(out).detach().cpu().float()
            if out.dim() == 3:
                out = out.unsqueeze(0)
            pq, pconf = m.decode(out); pq = pq[0].numpy(); pconf = pconf[0].numpy()
            for i in range(6):
                if vis[i]:
                    dq = (((pq[i, 0] - tgt[i, 0]) * IMG_WIDTH) ** 2 + ((pq[i, 1] - tgt[i, 1]) * IMG_HEIGHT) ** 2) ** 0.5
                    qc[i] += dq < TH; tot[i] += 1; qconf[i].append(pconf[i])
            if vis[2] and vis[3]:
                p = pq * S
                shang.append(ang(p[2], p[3]))
    print(f'\n== {label} ==')
    print(f'{"关键点":<16}{"conf≥0.6占":<12}{"PCK":<10}')
    for i, n in enumerate(TARGET_KP_NAMES):
        c = np.array(qconf[i])
        print(f'{n:<16}{(c >= 0.6).mean() * 100:<12.2f}{qc[i] / tot[i]:<10.3f}')
    print(f'总 PCK={qc.sum() / tot.sum():.4f}  双肩倾斜角中位={np.median(shang):.1f}°')
    return qc / tot, [(np.array(qconf[i]) >= 0.6).mean() for i in range(6)]


run(ex8, '纯 int8')
run(exm, '混合精度(stem+head int16)')
