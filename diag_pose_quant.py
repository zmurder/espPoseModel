"""量化对坐姿指标的影响：浮点 vs 纯int8 vs 混合精度 的 双肩倾斜角/肩宽 对比。
坐姿判断用双肩相对关系，比单点PCK更贴近真实用途——直接回答'量化后坐姿是否可用'。"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import kp_config
from kp_config import IMG_WIDTH, IMG_HEIGHT
from model import PoseNet
from dataset import PoseDataset
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import TargetPlatform, TorchExecutor
from quant_fix import build_calib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
S = np.array([IMG_WIDTH, IMG_HEIGHT])


def ang(a, b):
    return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))


def stat(name, arr, fmt='{:.2f}'):
    arr = np.array(arr)
    if len(arr) == 0:
        print(f'  {name}: 无数据'); return
    print(f'  {name}: 中位{fmt.format(np.median(arr))} 均值{fmt.format(arr.mean())} 90%={fmt.format(np.percentile(arr, 90))}')


def build_qgraph(disp):
    cl = DataLoader(build_calib('cam'), batch_size=1, shuffle=False, num_workers=0)
    return espdl_quantize_onnx(
        onnx_import_file='output/pose_model.onnx', espdl_export_file='output/_dp.espdl',
        calib_dataloader=cl, calib_steps=64, input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3',
        num_of_bits=8, collate_fn=lambda b: b, dispatching_override=disp, device=device,
        error_report=False, skip_export=True, export_test_values=False, verbose=0)


MIX = ['node_Conv_452', 'node_Conv_514', 'node_conv2d_32']
disp_mix = {L: TargetPlatform.ESPDL_S3_INT16.value for L in MIX}

model = PoseNet(4).to(device); model.eval()
sd = torch.load('checkpoints/best.pth', map_location=device)
model.load_state_dict(sd['model'] if 'model' in sd else sd)
el = DataLoader(PoseDataset('cam', training=False), batch_size=1, shuffle=False, num_workers=0)


def pose_metrics(pred_fn, label):
    shang, width, sherr = [], [], []
    with torch.no_grad():
        for img, hm, kps in el:
            p = pred_fn(img) * S
            vis = (kps[0, :, 2] >= 1).numpy()
            tgt = kps[0, :, :2].numpy() * S
            if vis[2] and vis[3]:
                shang.append(ang(p[2], p[3]))
                w = np.linalg.norm(tgt[3] - tgt[2])
                if w > 1:
                    width.append(np.linalg.norm(p[3] - p[2]) / w)
                # 双肩各自相对真值的位置误差(px)
                sherr.append(np.linalg.norm(p[2] - tgt[2]) + np.linalg.norm(p[3] - tgt[3]))
    print(f'\n== {label} ==')
    stat('双肩倾斜角(°)', shang, '{:.1f}')
    stat('肩宽比(预/真)', width, '{:.3f}')
    stat('双肩位置误差和(px)', sherr, '{:.1f}')
    return np.array(shang), np.array(width), np.array(sherr)


fa, fw, fe = pose_metrics(
    lambda img: model.decode(model(img.to(device)))[0][0].cpu().numpy(), '浮点')

g8 = build_qgraph(None); ex8 = TorchExecutor(g8)
def qpred8(img):
    o = ex8.forward(img.to(device)); o = o[0] if isinstance(o, (list, tuple)) else o
    o = torch.as_tensor(o).detach().cpu().float()
    if o.dim() == 3: o = o.unsqueeze(0)
    return model.decode(o)[0][0].numpy()
ia, iw, ie = pose_metrics(qpred8, '纯int8')

gm = build_qgraph(disp_mix); exm = TorchExecutor(gm)
def qpredm(img):
    o = exm.forward(img.to(device)); o = o[0] if isinstance(o, (list, tuple)) else o
    o = torch.as_tensor(o).detach().cpu().float()
    if o.dim() == 3: o = o.unsqueeze(0)
    return model.decode(o)[0][0].numpy()
ma, mw, me = pose_metrics(qpredm, '混合精度(stem+head)')

print('\n===== 坐姿指标差异 (量化 vs 浮点) =====')
print(f'  双肩倾斜角差异 中位: int8={np.median(np.abs(ia - fa)):.2f}°  混合={np.median(np.abs(ma - fa)):.2f}°')
print(f'  肩宽比差异    中位: int8={np.median(np.abs(iw - fw)):.3f}  混合={np.median(np.abs(mw - fw)):.3f}')
print(f'  双肩位置误差和 中位: int8={np.median(ie):.1f}px  混合={np.median(me):.1f}px  (浮点={np.median(fe):.1f}px)')
