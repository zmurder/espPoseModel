"""
统一姿态数据集加载
================
支持 COCO / custom / cam 三数据源，统一加载为 4 关键点 heatmap。
- 单人图过滤 + 4 关键点至少 3 个可见
- center_crop_resize 到 320x240（避免 letterbox 黑边）
- 训练增强：旋转±15° / 缩放0.8-1.2 / 平移±10% / 水平翻转(交换左右点) / 色彩抖动
- cam 源按 annotation.flipped 字段垂直翻转图片（坐标已对应翻转图）
- 高分辨率 heatmap (120x160, sigma=6)
- WeightedRandomSampler 控制三源混合比例
"""
import os
import json
import random

import cv2
cv2.setNumThreads(0)  # 避免 DataLoader 多 worker fork 时 OpenCV 线程死锁
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset

import kp_config
from kp_config import (IMG_WIDTH, IMG_HEIGHT, HEATMAP_HEIGHT, HEATMAP_WIDTH, SIGMA,
                       IMG_MEAN, IMG_STD, DATASET_SOURCES, DATA_MIX_RATIO)


def center_crop_resize(image, target_w, target_h):
    """从中心裁剪最大 4:3 区域再 resize 到目标尺寸。返回 (图像, crop_x, crop_y, crop_w, crop_h)"""
    orig_h, orig_w = image.shape[:2]
    target_ratio = target_w / target_h  # 4/3
    if orig_w / orig_h > target_ratio:
        crop_h = orig_h
        crop_w = int(round(orig_h * target_ratio))
        crop_x = (orig_w - crop_w) // 2
        crop_y = 0
    else:
        crop_w = orig_w
        crop_h = int(round(orig_w / target_ratio))
        crop_x = 0
        crop_y = (orig_h - crop_h) // 2
    cropped = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized, crop_x, crop_y, crop_w, crop_h


def generate_heatmap(kps_norm, heatmap_h, heatmap_w, sigma):
    """生成高斯热图。kps_norm: (6,3) [x,y,v] 归一化。返回 (6,H,W)"""
    heatmap = np.zeros((kps_norm.shape[0], heatmap_h, heatmap_w), dtype=np.float32)
    ys = np.arange(heatmap_h).reshape(heatmap_h, 1).astype(np.float32)
    xs = np.arange(heatmap_w).reshape(1, heatmap_w).astype(np.float32)
    for i in range(6):
        if kps_norm[i, 2] < 1:
            continue
        hx = kps_norm[i, 0] * heatmap_w
        hy = kps_norm[i, 1] * heatmap_h
        heatmap[i] = np.exp(-((xs - hx) ** 2 + (ys - hy) ** 2) / (2 * sigma ** 2))
    return heatmap


def color_jitter(img):
    """亮度/对比度抖动（RGB uint8）"""
    if random.random() < 0.5:
        b = random.uniform(0.8, 1.2)
        img = np.clip(img.astype(np.float32) * b, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        c = random.uniform(0.8, 1.2)
        m = img.mean()
        img = np.clip((img.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)
    return img


def apply_augmentation(img, kps_norm):
    """img: (240,320,3) RGB, kps_norm: (4,3) 归一化。返回增强后的 (img, kps_norm)"""
    W, H = IMG_WIDTH, IMG_HEIGHT
    kps_px = kps_norm.copy().astype(np.float32)
    kps_px[:, 0] *= W
    kps_px[:, 1] *= H
    vis = kps_px[:, 2].copy()

    # 旋转 + 缩放（围绕中心）
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    # 平移
    M[0, 2] += random.uniform(-0.1, 0.1) * W
    M[1, 2] += random.uniform(-0.1, 0.1) * H
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
    # 关键点变换
    ones = np.ones((kps_px.shape[0], 1))
    pts = np.hstack([kps_px[:, :2], ones])  # (4,3)
    new_pts = (M @ pts.T).T  # (4,2)
    kps_px[:, 0] = new_pts[:, 0]
    kps_px[:, 1] = new_pts[:, 1]
    kps_px[:, 2] = vis

    # 水平翻转（交换左右：0<->1 眼, 2<->3 耳, 4<->5 肩）
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        kps_px[:, 0] = W - kps_px[:, 0]
        kps_px[[0, 1]] = kps_px[[1, 0]]   # 左右眼
        kps_px[[2, 3]] = kps_px[[3, 2]]   # 左右耳
        kps_px[[4, 5]] = kps_px[[5, 4]]   # 左右肩

    img = color_jitter(img)

    out = kps_px.copy()
    out[:, 0] /= W
    out[:, 1] /= H
    # 越界点 clip 到 [0,1]（heatmap 仍在边界生成）
    out[:, 0] = np.clip(out[:, 0], 0, 1)
    out[:, 1] = np.clip(out[:, 1], 0, 1)
    return img, out


class PoseDataset(Dataset):
    def __init__(self, source_name, training=True, max_samples=None):
        self.source = DATASET_SOURCES[source_name]
        self.training = training
        self.kp_indices = self.source['kp_indices']

        with open(self.source['ann_file']) as f:
            coco = json.load(f)
        self.img_info = {im['id']: im for im in coco['images']}

        person_count = {}
        for ann in coco['annotations']:
            if ann.get('category_id', 1) == 1:
                person_count[ann['image_id']] = person_count.get(ann['image_id'], 0) + 1
        single = {iid for iid, c in person_count.items() if c == 1}

        self.samples = []
        for ann in coco['annotations']:
            if ann['image_id'] not in single:
                continue
            kps = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            extracted = kps[self.kp_indices]  # (4,3)
            n_vis = int(sum(1 for i in range(6) if extracted[i, 2] >= 1))
            if n_vis < 3:
                continue
            self.samples.append({
                'image_id': ann['image_id'],
                'keypoints': extracted,
                'flipped': ann.get('flipped', False),
            })

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)
        print(f'[{source_name}] {len(self.samples)} 样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(6):
            s = self.samples[idx]
            info = self.img_info[s['image_id']]
            img_path = os.path.join(self.source['img_dir'], info['file_name'])
            img = cv2.imread(img_path)
            if img is not None:
                break
            idx = (idx + 1) % len(self.samples)
        if img is None:
            # 全部读取失败，返回零样本
            return (torch.zeros(3, IMG_HEIGHT, IMG_WIDTH),
                    torch.zeros(6, HEATMAP_HEIGHT, HEATMAP_WIDTH),
                    torch.zeros(6, 3))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if s['flipped']:
            img = cv2.flip(img, 0)

        kps_px = s['keypoints'].copy().astype(np.float32)  # (4,3) 原图像素
        img, crop_x, crop_y, crop_w, crop_h = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
        # 坐标转换：减 crop 偏移 -> 乘缩放比
        kps_px[:, 0] = (kps_px[:, 0] - crop_x) * (IMG_WIDTH / crop_w)
        kps_px[:, 1] = (kps_px[:, 1] - crop_y) * (IMG_HEIGHT / crop_h)

        kps_norm = kps_px.copy()
        kps_norm[:, 0] /= IMG_WIDTH
        kps_norm[:, 1] /= IMG_HEIGHT

        if self.training:
            img, kps_norm = apply_augmentation(img, kps_norm)

        heatmap = generate_heatmap(kps_norm, HEATMAP_HEIGHT, HEATMAP_WIDTH, SIGMA)

        img = img.astype(np.float32) / 255.0
        img = (img - np.array(IMG_MEAN, dtype=np.float32)) / np.array(IMG_STD, dtype=np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        heatmap = torch.from_numpy(heatmap)
        kps_norm = torch.from_numpy(kps_norm.astype(np.float32))
        return img, heatmap, kps_norm


def get_dataloaders(batch_size=32, num_workers=4, max_samples=None, val_max_samples=None):
    """构建训练/验证 DataLoader。训练用 WeightedRandomSampler 按比例混合三源。"""
    use_cuda = torch.cuda.is_available()
    nw = num_workers if use_cuda else 0

    train_datasets = []
    train_weights = []
    for name in DATA_MIX_RATIO:  # custom_train, coco_train, cam
        ds = PoseDataset(name, training=True, max_samples=max_samples)
        train_datasets.append(ds)
        w = DATA_MIX_RATIO[name] / len(ds)
        train_weights.extend([w] * len(ds))
    train_set = ConcatDataset(train_datasets)
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=use_cuda,
                              persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)

    val_set = PoseDataset('custom_val', training=False, max_samples=val_max_samples)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=use_cuda,
                            persistent_workers=nw > 0)
    return train_loader, val_loader


def geometric_sanity_check(dataset, n=100):
    """检查前 n 样本眼睛 y < 肩膀 y（归一化后），违反率 > 20% 则警告"""
    random.seed(0)
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    violate = 0
    for i in idxs:
        s = dataset.samples[i]
        kps = s['keypoints']
        eye_y = (kps[0, 1] + kps[1, 1]) / 2
        shoulder_y = (kps[4, 1] + kps[5, 1]) / 2
        if eye_y >= shoulder_y:
            violate += 1
    rate = violate / len(idxs)
    print(f'[sanity] eyes_y<shoulders_y 违反率: {rate*100:.1f}% ({violate}/{len(idxs)})')
    if rate > 0.2:
        print('  [警告] 违反率高, 可能关键点索引错误!')
    return rate


if __name__ == '__main__':
    # 测试三源加载
    for name in ['custom_train', 'coco_train', 'custom_val']:
        try:
            ds = PoseDataset(name, training=False, max_samples=200)
            geometric_sanity_check(ds)
            img, hm, kps = ds[0]
            print(f'  样本: img {img.shape}, heatmap {hm.shape}, kps {kps.shape}')
        except Exception as e:
            print(f'  [{name}] 加载失败: {e}')
    # cam 源单独测（可能标注未生成）
    try:
        ds = PoseDataset('cam', training=False)
        geometric_sanity_check(ds)
    except Exception as e:
        print(f'[cam] 加载失败(可能标注未生成): {e}')
