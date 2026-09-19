"""量化 conf 崩修复验证: 纯int8 vs head-int16 的 conf 对比 (浮点 0.9 → int8 崩到 0.3, 试 head16 救峰值)。
test_img = ESP32 实拍图, cam_data2 = 校准域实拍。"""
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import TargetPlatform, TorchExecutor
from kp_config import IMG_MEAN, IMG_STD, IMG_WIDTH, IMG_HEIGHT
from model import PoseNet
from dataset import center_crop_resize
from quant_fix import build_calib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 新模型 head 的 6 个 Conv(1×1 24→32 + 2×DWSep + 1×1 32→6)
HEAD = ['node_Conv_621', 'node_Conv_623', 'node_Conv_625',
        'node_Conv_627', 'node_Conv_629', 'node_conv2d_39']
cl = DataLoader(build_calib('cam'), batch_size=1, shuffle=False, num_workers=0)


def mk(disp):
    return TorchExecutor(espdl_quantize_onnx(
        onnx_import_file='output/current.onnx', espdl_export_file='output/_t.espdl',
        calib_dataloader=cl, calib_steps=32, input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3',
        num_of_bits=8, collate_fn=lambda b: b, dispatching_override=disp, device=device,
        error_report=False, skip_export=True, export_test_values=False, verbose=0))


print('量化中 (纯int8 + head16)...')
ex8 = mk(None)
ex16 = mk({h: TargetPlatform.ESPDL_S3_INT16.value for h in HEAD})
m = PoseNet().to(device); m.eval()
m.load_state_dict(torch.load('checkpoints/best.pth', map_location=device)['model'])


def conf(ex, p):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    inp, *_ = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
    x = inp.astype(np.float32) / 255.0
    x = (x - np.array(IMG_MEAN, np.float32)) / np.array(IMG_STD, np.float32)
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        if ex is None:
            _, c = m.decode(m(xt))
        else:
            o = ex.forward(xt); o = o[0] if isinstance(o, (list, tuple)) else o
            o = torch.as_tensor(o).detach().cpu().float()
            if o.dim() == 3:
                o = o.unsqueeze(0)
            _, c = m.decode(o)
    return [round(v, 3) for v in c[0].tolist()]


imgs = [sorted(glob.glob('test_img/*.jpg'))[0], sorted(glob.glob('data/cam_data2/*.jpg'))[0]]
print(f'\n{"图":<28}{"浮点conf":<22}{"纯int8":<22}{"head16"}')
for p in imgs:
    print(f'{p.split("/")[-1]:<28}{str(conf(None, p)):<22}{str(conf(ex8, p)):<22}{conf(ex16, p)}')
print('\n(6点: 左眼/右眼/左耳/右耳/左肩/右肩; ESP32 阈值0.6)')
