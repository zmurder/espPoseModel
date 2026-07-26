"""
ESP-PPQ 量化 (esp32s3, int8) + 量化前后 PCK 对比
============================================
用 cam_data 校准（真实部署域），输出 .espdl/.info/.json，
并用 ppq TorchExecutor 评估量化后 PCK，与浮点模型对比。

用法:
    python3 quantize_espdl.py --onnx_path output/pose_model_6kp.onnx
"""
import os
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader

import kp_config
from kp_config import PCK_THRESHOLD_RATIO, IMG_WIDTH, IMG_HEIGHT, TARGET_KP_NAMES
from model import PoseNet
from dataset import PoseDataset


def compute_pck(pred_kps, target, vis):
    """pred_kps, target: (N,4,2) 归一化; vis: (N,4) bool"""
    threshold = PCK_THRESHOLD_RATIO * (IMG_WIDTH ** 2 + IMG_HEIGHT ** 2) ** 0.5
    dx = (pred_kps[..., 0] - target[..., 0]) * IMG_WIDTH
    dy = (pred_kps[..., 1] - target[..., 1]) * IMG_HEIGHT
    dist = (dx ** 2 + dy ** 2) ** 0.5
    return (dist < threshold) & vis


@torch.no_grad()
def eval_float_pck(model, loader, device):
    model.eval()
    correct = total = 0
    for img, hm, kps in loader:
        pred_hm = model(img.to(device))
        pred_kps, _ = model.decode(pred_hm)
        vis = kps[..., 2] >= 1
        mask = compute_pck(pred_kps.cpu().numpy(), kps[..., :2].numpy(), vis.numpy())
        correct += int(mask.sum())
        total += int(vis.sum())
    return correct / max(total, 1), total


def main():
    parser = argparse.ArgumentParser(description='ESP-PPQ 量化')
    parser.add_argument('--onnx_path', default='output/pose_model_6kp.onnx')
    parser.add_argument('--model_path', default='checkpoints/best.pth', help='浮点模型(算浮点PCK)')
    parser.add_argument('--output', default='output/pose_model.espdl')
    parser.add_argument('--target', default='esp32s3')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--calib_steps', type=int, default=32)
    parser.add_argument('--calib_source', default='cam')
    parser.add_argument('--eval_source', default='cam')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from esp_ppq.api import espdl_quantize_onnx

    # 校准数据：cam 源，shuffle=False（必须）
    calib_ds = PoseDataset(args.calib_source, training=False)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False, num_workers=0)

    def calib_collate(batch):
        # batch 是 default_collate 输出 (img, hm, kps)，只取 img
        return batch[0]

    print(f'量化: {args.onnx_path} -> {args.output} (target={args.target}, {args.bits}bit)')
    print(f'校准数据: {args.calib_source} ({len(calib_ds)} 样本), calib_steps={args.calib_steps}')
    quant_graph = espdl_quantize_onnx(
        onnx_import_file=args.onnx_path,
        espdl_export_file=args.output,
        calib_dataloader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=[1, 3, 240, 320],
        inputs=None,
        target=args.target,
        num_of_bits=args.bits,
        collate_fn=calib_collate,
        dispatching_override=None,
        device=device,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
    print(f'\n量化产物: {args.output} (+ .info / .json)')

    # 浮点 PCK
    model = PoseNet().to(device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    eval_ds = PoseDataset(args.eval_source, training=False)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)
    float_pck, total = eval_float_pck(model, eval_loader, device)
    print(f'浮点 PCK@0.1 ({args.eval_source}): {float_pck:.4f} ({total} 关键点)')

    # 量化 PCK（ppq TorchExecutor）
    try:
        from esp_ppq import TorchExecutor
        executor = TorchExecutor(quant_graph)
        model_cpu = PoseNet()
        correct = 0
        total_q = 0
        for img, hm, kps in eval_loader:
            out = executor.forward(img.to(device))
            pred_hm = out[0] if isinstance(out, (list, tuple)) else out
            pred_hm = torch.as_tensor(pred_hm).detach().cpu().float()
            if pred_hm.dim() == 3:
                pred_hm = pred_hm.unsqueeze(0)
            pred_kps, _ = model_cpu.decode(pred_hm)
            vis = kps[..., 2] >= 1
            mask = compute_pck(pred_kps.numpy(), kps[..., :2].numpy(), vis.numpy())
            correct += int(mask.sum())
            total_q += int(vis.sum())
        quant_pck = correct / max(total_q, 1)
        print(f'量化 PCK@0.1 ({args.eval_source}): {quant_pck:.4f}')
        print(f'精度损失: {(float_pck - quant_pck) * 100:.2f}%')
        if (float_pck - quant_pck) < 0.03:
            print('[OK] 量化损失 < 3%, 可部署')
        else:
            print('[警告] 量化损失较大, 考虑混合精度或 QAT')
    except Exception as e:
        print(f'量化后 PCK 评估跳过(ppq executor 异常: {e})')
        print('可在 ESP32 实机验证 .espdl 效果')


if __name__ == '__main__':
    main()
