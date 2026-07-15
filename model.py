"""
ESP32-S3 姿态检测模型 — 高分辨率纯 Heatmap 多尺度 FPN
从零设计，不参考 V2 BlazePose-lite。

设计要点：
- 输入 (B,3,240,320)，输出 heatmap (B,4,120,160) —— 1/2 输入分辨率，定位精度高
- 多尺度 top-down FPN 融合 120x160 / 60x80 / 30x40 / 15x20 四层特征
  （高分辨率特征对关键点定位至关重要，这是 V2 缺失的）
- 仅用 ESP-DL 支持算子：Conv / BatchNorm / HardSwish / ConvTranspose / Add / Sigmoid
- 完全无 Resize / F.interpolate，所有上采样用 ConvTranspose2d(kernel=2, stride=2)
- 纯 heatmap argmax 后处理（ESP32 C++ 实现简单），无 offset
- 参数量 ~0.9M
"""
import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块：1x1 升维 -> 3x3 depthwise -> 1x1 降维
    stride=1 且通道匹配时残差连接。depthwise 的 groups=hidden_dim（符合 ESP-DL groups 约束）。"""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.Hardswish(),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class PoseNet(nn.Module):
    """高分辨率纯 Heatmap 姿态模型
    输入: (B, 3, 240, 320)
    输出: heatmap (B, num_keypoints, 120, 160)
    """

    def __init__(self, num_keypoints=4, in_channels=3):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Stem: 下采样到 120x160（即输出分辨率）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
        )  # (32, 120, 160)

        # ---- Backbone：4 级逐步下采样，保留各尺度特征 ----
        # Stage1: 120x160
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 32, 1, 4),
        )  # c1 (32, 120, 160)

        # Stage2: 60x80
        self.stage2 = nn.Sequential(
            InvertedResidual(32, 64, 2, 4),
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 64, 1, 4),
        )  # c2 (64, 60, 80)

        # Stage3: 30x40
        self.stage3 = nn.Sequential(
            InvertedResidual(64, 128, 2, 4),
            InvertedResidual(128, 128, 1, 4),
            InvertedResidual(128, 128, 1, 4),
        )  # c3 (128, 30, 40)

        # Stage4: 15x20
        self.stage4 = nn.Sequential(
            InvertedResidual(128, 256, 2, 4),
        )  # c4 (256, 15, 20)

        # ---- Top-down FPN：ConvTranspose 逐级上采样，融合多尺度特征 ----
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2, bias=False)   # 15x20 -> 30x40
        self.fusion3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128), nn.Hardswish())

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2, bias=False)    # 30x40 -> 60x80
        self.fusion2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64), nn.Hardswish())

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2, bias=False)     # 60x80 -> 120x160
        self.fusion1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0, bias=False), nn.BatchNorm2d(32), nn.Hardswish())

        # ---- Heatmap head (120x160) ----
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, num_keypoints, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)          # (32, 120, 160)
        c1 = self.stage1(x)       # (32, 120, 160)
        c2 = self.stage2(c1)      # (64, 60, 80)
        c3 = self.stage3(c2)      # (128, 30, 40)
        c4 = self.stage4(c3)      # (256, 15, 20)

        # top-down FPN：深层语义上采样融合到高分辨率
        p3 = self.fusion3(self.up3(c4) + c3)   # (128, 30, 40)
        p2 = self.fusion2(self.up2(p3) + c2)   # (64, 60, 80)
        p1 = self.fusion1(self.up1(p2) + c1)   # (32, 120, 160)

        heatmap = self.heatmap_head(p1)        # (num_keypoints, 120, 160)
        return heatmap

    def decode(self, heatmap):
        """从 heatmap 提取关键点归一化坐标（仅推理/评测用，不导出）。
        Args: heatmap (B, K, H, W)
        Returns: keypoints (B, K, 2) 归一化 [x,y]∈[0,1], max_vals (B, K) 置信度
        ESP32 C++ 后处理只需 argmax + 归一化。
        """
        B, K, H, W = heatmap.shape
        flat = heatmap.view(B, K, -1)
        max_vals, max_idx = flat.max(dim=2)    # (B, K)
        row = max_idx // W                      # y
        col = max_idx % W                       # x
        x = col.float() / W
        y = row.float() / H
        return torch.stack([x, y], dim=2), max_vals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = PoseNet(num_keypoints=4)
    x = torch.randn(2, 3, 240, 320)
    with torch.no_grad():
        heatmap = model(x)
    params = count_parameters(model)
    print(f"模型参数量: {params:,} ({params / 1e6:.2f}M)")
    print(f"Heatmap 形状: {heatmap.shape}  (期望 (2, 4, 120, 160))")
    kps, confs = model.decode(heatmap)
    print(f"关键点形状: {kps.shape}  置信度范围: [{confs.min():.3f}, {confs.max():.3f}]")
