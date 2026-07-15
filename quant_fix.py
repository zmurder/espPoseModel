"""量化修复对比实验：混合校准集(cam+cam_data2) + 混合精度 int16(stem/heatmap head)
一次跑 4 个配置，输出分关键点对比，用数据选定最终方案。"""
import os, glob, json
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

import kp_config
from kp_config import (IMG_WIDTH, IMG_HEIGHT, IMG_MEAN, IMG_STD,
                       PCK_THRESHOLD_RATIO, TARGET_KP_NAMES, DATASET_SOURCES)
from model import PoseNet
from dataset import PoseDataset, center_crop_resize
from esp_ppq.api import espdl_quantize_onnx, DispatchingTable
from esp_ppq import TargetPlatform, TorchExecutor

TH = PCK_THRESHOLD_RATIO * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImgCalib(Dataset):
    """只读图做校准。cam_data 带 flipped 标注则翻正；无标注目录直接读图。"""
    def __init__(self, img_dir, ann_file=None):
        self.files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.flip = {}
        if ann_file and os.path.exists(ann_file):
            coco = json.load(open(ann_file))
            id2name = {im['id']: im['file_name'] for im in coco['images']}
            for a in coco['annotations']:
                self.flip[id2name[a['image_id']]] = a.get('flipped', False)
        print(f'  [calib] {img_dir}: {len(self.files)} 图')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.files[i]), cv2.COLOR_BGR2RGB)
        if self.flip.get(os.path.basename(self.files[i]), False):
            img = cv2.flip(img, 0)
        img, *_ = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(IMG_MEAN, np.float32)) / np.array(IMG_STD, np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)


class MixedCalib(Dataset):
    def __init__(self, subs):
        self.subs = subs

    def __len__(self):
        return sum(len(s) for s in self.subs)

    def __getitem__(self, i):
        for s in self.subs:
            if i < len(s):
                return s[i]
            i -= len(s)


def build_calib(name):
    cam = DATASET_SOURCES['cam']
    cam2_dir = os.path.join(os.path.dirname(cam['img_dir']), 'cam_data2')
    if name == 'cam':
        return ImgCalib(cam['img_dir'], cam['ann_file'])
    if name == 'cam_mix':
        return MixedCalib([ImgCalib(cam['img_dir'], cam['ann_file']), ImgCalib(cam2_dir)])


def float_pck():
    model = PoseNet(4).to(device); model.eval()
    sd = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    el = DataLoader(PoseDataset('cam', training=False), batch_size=1, shuffle=False, num_workers=0)
    fp = np.zeros(4); tot = np.zeros(4)
    with torch.no_grad():
        for img, hm, kps in el:
            pf, _ = model.decode(model(img.to(device))); pf = pf[0].cpu().numpy()
            vis = (kps[0, :, 2] >= 1).numpy(); tgt = kps[0, :, :2].numpy()
            for i in range(4):
                if vis[i]:
                    df = (((pf[i, 0] - tgt[i, 0]) * IMG_WIDTH) ** 2 +
                          ((pf[i, 1] - tgt[i, 1]) * IMG_HEIGHT) ** 2) ** 0.5
                    fp[i] += df < TH; tot[i] += 1
    return fp / np.clip(tot, 1, None)


def quant_eval(calib_name, mix16, calib_steps=64):
    calib_ds = build_calib(calib_name)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False, num_workers=0)
    # espdl_quantize_onnx 内部对 dispatching_override 调 .items()，要 dict 不能是 DispatchingTable
    disp = {L: TargetPlatform.ESPDL_S3_INT16.value for L in mix16} if mix16 else None
    print(f'\n>>> calib={calib_name}, mix16={mix16 or "无"}')
    qg = espdl_quantize_onnx(
        onnx_import_file='output/pose_model.onnx',
        espdl_export_file='output/_fix.espdl',
        calib_dataloader=calib_loader, calib_steps=calib_steps,
        input_shape=[1, 3, 240, 320], inputs=None, target='esp32s3', num_of_bits=8,
        collate_fn=lambda b: b, dispatching_override=disp, device=device,
        error_report=False, skip_export=True, export_test_values=False, verbose=0)
    executor = TorchExecutor(qg)

    model = PoseNet(4).to(device); model.eval()
    sd = torch.load('checkpoints/best.pth', map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    el = DataLoader(PoseDataset('cam', training=False), batch_size=1, shuffle=False, num_workers=0)
    qc = np.zeros(4); tot = np.zeros(4); qsh = [[] for _ in range(4)]
    with torch.no_grad():
        for img, hm, kps in el:
            tgt = kps[0, :, :2].numpy(); vis = (kps[0, :, 2] >= 1).numpy()
            pf, _ = model.decode(model(img.to(device))); pf = pf[0].cpu().numpy()
            out = executor.forward(img.to(device))
            out = out[0] if isinstance(out, (list, tuple)) else out
            out = torch.as_tensor(out).detach().cpu().float()
            if out.dim() == 3:
                out = out.unsqueeze(0)
            pq, _ = model.decode(out); pq = pq[0].numpy()
            for i in range(4):
                if vis[i]:
                    dq = (((pq[i, 0] - tgt[i, 0]) * IMG_WIDTH) ** 2 +
                          ((pq[i, 1] - tgt[i, 1]) * IMG_HEIGHT) ** 2) ** 0.5
                    qc[i] += dq < TH; tot[i] += 1
                    sx = (pf[i, 0] - pq[i, 0]) * IMG_WIDTH; sy = (pf[i, 1] - pq[i, 1]) * IMG_HEIGHT
                    qsh[i].append((sx * sx + sy * sy) ** 0.5)
    pck = qc / np.clip(tot, 1, None)
    sh90 = [np.percentile(qsh[i], 90) if qsh[i] else 0 for i in range(4)]
    shmed = [np.median(qsh[i]) if qsh[i] else 0 for i in range(4)]
    return pck, sh90, shmed


if __name__ == '__main__':
    # 已验证: baseline=左肩0.901, 混合校准无效, stem+head=左肩0.908(漂移90%=20px异常)
    # 换输出端通路: fusion1+head(离 argmax 近, 稳左肩弱峰值) vs 全保
    configs = [
        ('输出端(fusion1+head)',     'cam', ['node_Conv_512', 'node_Conv_514', 'node_conv2d_32']),
        ('全保(stem+fusion1+head)',  'cam', ['node_Conv_452', 'node_Conv_512', 'node_Conv_514', 'node_conv2d_32']),
    ]
    fpck = float_pck()
    print(f'\n浮点 PCK: ' + '  '.join(f'{n}={fpck[i]:.3f}' for i, n in enumerate(TARGET_KP_NAMES)))
    print(f'参照: baseline(cam/int8) 左肩=0.901 损失=+0.021')
    rows = []
    for name, cn, mx in configs:
        pq, sh90, shmed = quant_eval(cn, mx)
        rows.append((name, pq, sh90, shmed))
        print(f'  {name}: ' + '  '.join(f'{n}={pq[i]:.3f}' for i, n in enumerate(TARGET_KP_NAMES)))
    print('\n===== 汇总(左肩为重点) =====')
    print(f'浮点左肩={fpck[2]:.3f} | baseline int8 左肩=0.901(损失+0.021) | stem+head 左肩=0.908(损失+0.014,漂移90%=20px)')
    for name, pq, sh90, shmed in rows:
        print(f'  {name:<26} 左肩={pq[2]:.3f} 损失={fpck[2]-pq[2]:+.3f} '
              f'漂移中位={shmed[2]:.1f} 漂移90%={sh90[2]:.1f}px | 总={(pq.mean()):.3f}')
