"""
姿态检测模型训练
================
纯 Heatmap MSE Loss（带关键点权重），分阶段训练，GPU 自适应，checkpoint/早停/OOM 恢复。

分阶段:
    python3 train.py --phase quick --epochs 5        # 快速验证
    python3 train.py --phase full --epochs 100 --resume  # 从 checkpoint 继续完整训练
"""
import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import kp_config
from kp_config import (TARGET_KP_WEIGHTS, PCK_THRESHOLD_RATIO, IMG_WIDTH, IMG_HEIGHT,
                       CHECKPOINT_DIR)
from model import PoseNet, count_parameters
from dataset import get_dataloaders


def compute_pck(pred_kps, target_kps, vis, img_w, img_h, ratio):
    """pred_kps, target_kps: (B,4,2) 归一化; vis: (B,4) bool。返回 correct mask (B,4)"""
    threshold = ratio * (img_w ** 2 + img_h ** 2) ** 0.5
    dx = (pred_kps[..., 0] - target_kps[..., 0]) * img_w
    dy = (pred_kps[..., 1] - target_kps[..., 1]) * img_h
    dist = (dx ** 2 + dy ** 2) ** 0.5
    return (dist < threshold) & vis


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    kp_correct = torch.zeros(6, dtype=torch.long)
    kp_total = torch.zeros(6, dtype=torch.long)
    for i, (img, hm, kps) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        img = img.to(device)
        pred_hm = model(img)
        pred_kps, _ = model.decode(pred_hm)
        pred_kps = pred_kps.cpu()
        vis = kps[..., 2] >= 1
        mask = compute_pck(pred_kps, kps[..., :2], vis, IMG_WIDTH, IMG_HEIGHT, PCK_THRESHOLD_RATIO)
        correct += int(mask.sum())
        total += int(vis.sum())
        kp_correct += mask.long().sum(dim=0)
        kp_total += vis.long().sum(dim=0)
    pck = correct / max(total, 1)
    per_kp = (kp_correct.float() / kp_total.clamp(min=1)).tolist()
    return pck, per_kp


def save_ckpt(path, epoch, model, optimizer, scheduler, best_pck):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_pck': best_pck,
    }, path)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        if args.batch_size > 8:
            print(f'WARNING: CPU 模式, batch_size {args.batch_size}->8')
            args.batch_size = 8
    print(f'device={device}, batch_size={args.batch_size}')
    if device == 'cuda':
        torch.backends.cudnn.benchmark = False  # 5060(Blackwell) 上 benchmark 会卡住, 关闭

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(os.path.join(CHECKPOINT_DIR, 'runs'))

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_samples=args.max_samples, val_max_samples=args.val_max_samples)
    print(f'train batches/epoch={len(train_loader)}, val batches={len(val_loader)}')

    model = PoseNet().to(device)
    print(f'模型参数量: {count_parameters(model):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)
    kp_w = torch.tensor(TARGET_KP_WEIGHTS, device=device).view(1, -1, 1, 1)
    fg_alpha = args.fg_alpha
    print(f'loss: 前景加权 alpha={fg_alpha} (0=纯MSE), kp_w={TARGET_KP_WEIGHTS}')

    start_epoch = 0
    best_pck = 0.0
    if args.resume:
        ckpt = os.path.join(CHECKPOINT_DIR, 'latest.pth')
        if os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location=device)
            model.load_state_dict(sd['model'])
            optimizer.load_state_dict(sd['optimizer'])
            scheduler.load_state_dict(sd['scheduler'])
            start_epoch = sd['epoch'] + 1
            best_pck = sd['best_pck']
            print(f'恢复 checkpoint: epoch {start_epoch}, best_pck {best_pck:.4f}')
        else:
            print(f'未找到 {ckpt}, 从头训练')

    patience = 0
    EARLY_STOP = 20
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for batch_idx, (img, hm, kps) in enumerate(train_loader):
            img, hm = img.to(device), hm.to(device)
            try:
                pred_hm = model(img)
                diff2 = (pred_hm - hm) ** 2
                if fg_alpha > 0:
                    fg_w = 1.0 + fg_alpha * hm       # (B,4,120,160) 峰值区高权, 背景=1
                    loss = (diff2 * fg_w * kp_w).mean()
                else:
                    loss = (diff2 * kp_w).mean()     # 复现旧 loss, 便于 A/B
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'  OOM epoch {epoch} batch {batch_idx}, 跳过')
                optimizer.zero_grad()
                continue
            running += loss.item()
            if batch_idx % 50 == 0:
                print(f'  e{epoch} b{batch_idx}/{len(train_loader)} loss={loss.item():.4f} '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)
                if device == 'cuda' and batch_idx % 100 == 0:
                    mem = torch.cuda.memory_allocated() / 1024 ** 3
                    print(f'    GPU mem: {mem:.2f} GB')
        train_loss = running / max(len(train_loader), 1)

        pck, per_kp = evaluate(model, val_loader, device, max_batches=50 if args.phase == 'quick' else None)
        scheduler.step(pck)
        dt = time.time() - t0
        print(f'== epoch {epoch} loss={train_loss:.4f} val_pck={pck:.4f} '
              f'per_kp={[f"{x:.3f}" for x in per_kp]} {dt:.0f}s ==', flush=True)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('val/pck', pck, epoch)
        for i, v in enumerate(per_kp):
            writer.add_scalar(f'val/kp_{i}', v, epoch)

        save_ckpt(os.path.join(CHECKPOINT_DIR, 'latest.pth'), epoch, model, optimizer, scheduler, best_pck)
        if pck > best_pck:
            best_pck = pck
            save_ckpt(os.path.join(CHECKPOINT_DIR, 'best.pth'), epoch, model, optimizer, scheduler, best_pck)
            patience = 0
            print(f'  新 best_pck={best_pck:.4f}')
        else:
            patience += 1
            print(f'  patience {patience}/{EARLY_STOP}')
        if epoch % 10 == 0:
            save_ckpt(os.path.join(CHECKPOINT_DIR, f'epoch_{epoch}.pth'), epoch, model, optimizer, scheduler, best_pck)

        if patience >= EARLY_STOP:
            print(f'早停: {EARLY_STOP} epoch 无提升')
            break

    writer.close()
    print(f'\n训练结束. best_pck={best_pck:.4f}')
    if args.phase == 'quick':
        print('Phase 1 完成。请运行可视化对比确认效果:')
        print('  python3 compare_mediapipe.py --model_path checkpoints/latest.pth')
        print('确认效果后运行 Phase 2:')
        print('  python3 train.py --phase full --epochs 100 --resume')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['quick', 'full'], default='quick')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16, help='5060(8GB)用16, 大显存可32')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max_samples', type=int, default=None, help='每数据源最大样本数(降载)')
    p.add_argument('--val_max_samples', type=int, default=None)
    p.add_argument('--quick_val', action='store_true', help='(已弃用, 由 phase 控制)')
    p.add_argument('--fg_alpha', type=float, default=4.0,
                   help='前景(峰值)加权强度: 0=关闭(复现旧MSE), 推荐4.0')
    args = p.parse_args()
    if args.phase == 'quick' and args.epochs > 5:
        args.epochs = 5
        print('quick 模式, epochs 限制为 5')
    train(args)


if __name__ == '__main__':
    main()
