"""stem+head int16 量化可视化 cam_data2 相同100张，与纯int8(vis_cam2)对比。
画 浮点(蓝)/量化(红) 关键点+骨架+conf。"""
import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import TargetPlatform, TorchExecutor

import kp_config
from kp_config import IMG_MEAN, IMG_STD, IMG_WIDTH, IMG_HEIGHT, SKELETON
from model import PoseNet
from dataset import center_crop_resize
from quant_fix import build_calib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MIX = ['node_Conv_452', 'node_Conv_514', 'node_conv2d_32']
disp_mix = {L: TargetPlatform.ESPDL_S3_INT16.value for L in MIX}

cl = DataLoader(build_calib('cam'), batch_size=1, shuffle=False, num_workers=0)
print('量化(stem+head int16)...')
qg = espdl_quantize_onnx(
    onnx_import_file='output/current.onnx', espdl_export_file='output/_vcm.espdl',
    calib_dataloader=cl, calib_steps=64, input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3',
    num_of_bits=8, collate_fn=lambda b: b, dispatching_override=disp_mix, device=device,
    error_report=False, skip_export=True, export_test_values=False, verbose=0)
executor = TorchExecutor(qg)

m = PoseNet().to(device); m.eval()
sd = torch.load('checkpoints/best.pth', map_location=device)
m.load_state_dict(sd['model'] if 'model' in sd else sd)

files = sorted(glob.glob('data/cam_data2/*.jpg'))[:100]
os.makedirs('output/vis_cam2_mix', exist_ok=True)
print(f'可视化 {len(files)} 张 -> output/vis_cam2_mix/ (对比纯int8: output/vis_cam2/)')
low_sh = 0
for f in files:
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    inp, *_ = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
    x = inp.astype(np.float32) / 255.0
    x = (x - np.array(IMG_MEAN, np.float32)) / np.array(IMG_STD, np.float32)
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        pf, pfc = m.decode(m(xt))
        out = executor.forward(xt)
        out = out[0] if isinstance(out, (list, tuple)) else out
        out = torch.as_tensor(out).detach().cpu().float()
        if out.dim() == 3:
            out = out.unsqueeze(0)
        pq, pqc = m.decode(out)
    pf = pf[0].cpu().numpy(); pfc = pfc[0].cpu().numpy()
    pq = pq[0].cpu().numpy(); pqc = pqc[0].cpu().numpy()
    vis = cv2.cvtColor(inp, cv2.COLOR_RGB2BGR).copy()
    for a, b in SKELETON:
        cv2.line(vis, (int(pq[a, 0] * 320), int(pq[a, 1] * 240)),
                 (int(pq[b, 0] * 320), int(pq[b, 1] * 240)), (0, 0, 255), 1)
    for i in range(6):
        cv2.circle(vis, (int(pf[i, 0] * 320), int(pf[i, 1] * 240)), 5, (255, 0, 0), -1)
        cv2.circle(vis, (int(pq[i, 0] * 320), int(pq[i, 1] * 240)), 5, (0, 0, 255), 2)
        cv2.putText(vis, f'{pqc[i]:.2f}', (int(pq[i, 0] * 320) + 6, int(pq[i, 1] * 240)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(vis, 'stem+head int16  Blue=float Red=quant', (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(os.path.join('output/vis_cam2_mix', os.path.basename(f)), vis)
    if pqc[2] < 0.5 or pqc[3] < 0.5:
        low_sh += 1

print(f'\n完成: {len(files)} 张 -> output/vis_cam2_mix/')
print(f'stem+head 量化肩膀 conf<0.5: {low_sh}/{len(files)} ({low_sh/len(files)*100:.0f}%)  (纯int8 为 2%)')
