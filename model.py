"""
ESP32-S3 姿态检测模型 — 高分辨率 Heatmap（重构版, ~0.39G MACs, 6 关键点）
========================================================================
重构目标: 原 1.78G MACs / ESP32-S3 推理 10s → 0.39G MACs / 150-300ms。

主要改动（输入接口不变: (B,3,240,320) → Sigmoid (B,num_keypoints,120,160)）:
1. 上采样: ConvTranspose2d(InsertZeros+Conv, 中间张量溢出 PSRAM)
          → Resize nearest (int8 原生支持, 零中间张量)
2. 通道:  stem 32/s1 32 → stem 24/s1 24; expand_ratio 4 → 2
3. stage1: 2 块 → 1 块 (砍高分辨率大头)
4. 激活:   HardSwish → ReLU
5. Head:   单 3×3 → 1×1 expand + 2× DWSeparable(3×3) + 1×1
          (空间细化更强, 保肩膀定位; 输出通道由 num_keypoints 决定)

参数量: ~0.14M (原 0.9M); int8 .espdl ~150KB (原 1.2MB)
算子: Conv/BatchNorm/Relu/Add/Resize(nearest)/Sigmoid, 全 ESP-DL int8 支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块（ReLU 版, 默认 expand=2）。
    depthwise 的 groups=hidden 满足 ESP-DL 'groups only support 1 or input_channels'。"""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class DWSeparable(nn.Module):
    """深度可分离 3×3 块（DwConv3×3 + 1×1 Pointwise），给 head 增加空间细化。
    比 Conv3×3 便宜 ~9×, 级联感受野更大, 利于肩膀等边缘关键点定位。"""

    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
        self.bn_pw = nn.BatchNorm2d(ch)

    def forward(self, x):
        x = torch.relu(self.bn_dw(self.dw(x)))
        x = torch.relu(self.bn_pw(self.pw(x)))
        return x


class FPNUp(nn.Module):
    """FPN 单级: proj(deep)+BN+ReLU → Resize×2 → lateral(lat)+BN+ReLU → Add → fusion+BN+ReLU。
    无 ConvTranspose, 无 InsertZeros 大中间张量。"""

    def __init__(self, in_ch_up, in_ch_lat, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch_up, out_ch, 1, 1, 0, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_ch)
        self.lateral = nn.Conv2d(in_ch_lat, out_ch, 1, 1, 0, bias=False)
        self.bn_lat = nn.BatchNorm2d(out_ch)
        self.fusion = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn_fuse = nn.BatchNorm2d(out_ch)

    def forward(self, up_feat, lateral_feat):
        up = torch.relu(self.bn_proj(self.proj(up_feat)))
        up = F.interpolate(up, scale_factor=2.0, mode='nearest')   # 替代 ConvTranspose
        lat = torch.relu(self.bn_lat(self.lateral(lateral_feat)))
        return torch.relu(self.bn_fuse(self.fusion(up + lat)))


class PoseNet(nn.Module):
    """高分辨率 Heatmap 姿态模型（重构版）。
    输入 (B,3,240,320) → Sigmoid heatmap (B,num_keypoints,120,160)。"""

    def __init__(self, num_keypoints=6, in_channels=3):
        super().__init__()
        self.num_keypoints = num_keypoints

        c1, c2, c3, c4 = 24, 40, 72, 96

        # Stem: 240×320 → 120×160
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # ---- Backbone 4 级 ----
        self.stage1 = nn.Sequential(InvertedResidual(c1, c1, 1, 2))                  # @120×160
        self.stage2 = nn.Sequential(
            InvertedResidual(c1, c2, 2, 2), InvertedResidual(c2, c2, 1, 2),
            InvertedResidual(c2, c2, 1, 2),                                            # @60×80
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(c2, c3, 2, 2), InvertedResidual(c3, c3, 1, 2),
            InvertedResidual(c3, c3, 1, 2),                                            # @30×40
        )
        self.stage4 = nn.Sequential(InvertedResidual(c3, c4, 2, 2))                   # @15×20

        # ---- Top-down FPN（Resize nearest，无 ConvTranspose）----
        self.fpn3 = FPNUp(c4, c3, c3)   # 15×20 → 30×40
        self.fpn2 = FPNUp(c3, c2, c2)   # 30×40 → 60×80
        self.fpn1 = FPNUp(c2, c1, c1)   # 60×80 → 120×160

        # ---- Heatmap head @120×160: 1×1 expand + 普通 Conv3×3 + 1×1
        # 用普通 Conv3×3 替代 DWSeparable: depthwise int8 量化误差大(25-36%)会压垮峰值, 普通 Conv 量化友好
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c1, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),   # 普通 Conv3×3(参数量已 > 2×DWSeparable, 容量够)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_keypoints, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)         # (24,120,160)
        c1 = self.stage1(x)      # (24,120,160)
        c2 = self.stage2(c1)     # (40, 60, 80)
        c3 = self.stage3(c2)     # (72, 30, 40)
        c4 = self.stage4(c3)     # (96, 15, 20)
        p3 = self.fpn3(c4, c3)   # (72, 30, 40)
        p2 = self.fpn2(p3, c2)   # (40, 60, 80)
        p1 = self.fpn1(p2, c1)   # (24,120,160)
        return self.heatmap_head(p1)   # (num_keypoints,120,160)

    def decode(self, heatmap):
        """从 heatmap 提取关键点归一化坐标（仅推理/评测用，不导出）。
        接口逐行与旧版一致 — ESP32 C++ 后处理逻辑相同。"""
        B, K, H, W = heatmap.shape
        flat = heatmap.view(B, K, -1)
        max_vals, max_idx = flat.max(dim=2)
        row = max_idx // W
        col = max_idx % W
        x = col.float() / W
        y = row.float() / H
        return torch.stack([x, y], dim=2), max_vals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = PoseNet()                                  # 默认 6 关键点
    x = torch.randn(2, 3, 240, 320)
    with torch.no_grad():
        heatmap = model(x)
    print(f"模型参数量: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")
    print(f"Heatmap 形状: {heatmap.shape}  (期望 (2, 6, 120, 160))")
    kps, confs = model.decode(heatmap)
    print(f"关键点形状: {kps.shape}  conf 范围: [{confs.min():.3f}, {confs.max():.3f}]")
