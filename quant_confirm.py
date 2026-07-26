"""确认可视化用的是量化模型：打印 cam_data2 几张图 浮点 vs 量化(int8) 的 conf 差异。
若两者完全相同则没用到量化；若有差异(量化略低)则量化在跑。"""
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import TorchExecutor
from kp_config import IMG_MEAN, IMG_STD, IMG_WIDTH, IMG_HEIGHT, TARGET_KP_NAMES
from model import PoseNet
from dataset import center_crop_resize
from quant_fix import build_calib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cl = DataLoader(build_calib('cam'), batch_size=1, shuffle=False, num_workers=0)
print('量化(纯int8)...')
qg = espdl_quantize_onnx(
    onnx_import_file='output/pose_model_6kp.onnx', espdl_export_file='output/_qc.espdl',
    calib_dataloader=cl, calib_steps=64, input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3',
    num_of_bits=8, collate_fn=lambda b: b, dispatching_override=None, device=device,
    error_report=False, skip_export=True, export_test_values=False, verbose=0)
executor = TorchExecutor(qg)
print(f'量化图: {len(qg.operations)} ops, executor=TorchExecutor(量化图) -> 用的确实是量化模型\n')

m = PoseNet().to(device); m.eval()
sd = torch.load('checkpoints/best.pth', map_location=device)
m.load_state_dict(sd['model'] if 'model' in sd else sd)

files = sorted(glob.glob('data/cam_data2/*.jpg'))[:8]
print(f'{"图":<32}{"浮点 conf [眼,眼,肩,肩]":<32}{"量化 conf":<32}{"差(浮-量)"}')
all_diff = []
for f in files:
    import cv2
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    inp, *_ = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
    x = inp.astype(np.float32) / 255.0
    x = (x - np.array(IMG_MEAN, np.float32)) / np.array(IMG_STD, np.float32)
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        _, pfc = m.decode(m(xt))
        out = executor.forward(xt)
        out = out[0] if isinstance(out, (list, tuple)) else out
        out = torch.as_tensor(out).detach().cpu().float()
        if out.dim() == 3:
            out = out.unsqueeze(0)
        _, pqc = m.decode(out)
    pfc = pfc[0].cpu().numpy(); pqc = pqc[0].cpu().numpy()
    d = pfc - pqc
    all_diff.append(d)
    bn = f.split('/')[-1][:28]
    print(f'{bn:<32}{str([round(v,3) for v in pfc]):<32}{str([round(v,3) for v in pqc]):<32}{[round(v,3) for v in d]}')

all_diff = np.array(all_diff)
print(f'\n平均 conf 差(浮点-量化): {[round(v,4) for v in all_diff.mean(0)]}')
print('-> 差>0(量化 conf 低于浮点) = 量化模型确实在跑，不是浮点')
