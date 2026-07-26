# 6 点重构模型量化精度分析（conf 崩问题）

> 6 点重构模型（0.13M、DWSeparable head、Resize 上采样、sigma=4）int8 量化后
> **heatmap 峰值（conf）全面崩溃**，ESP32 坐姿检测失败。本文记录现象、根因、尝试与解法。

## 1. 现象

| 指标 | 浮点模型 | int8 量化 | 说明 |
|---|---|---|---|
| cam PCK@0.1 | 0.9765 | 0.9650 | 损失 1.15%（<3%，看似可部署） |
| **conf（峰值高度）** | **0.9+** | **0.2-0.4** | **全面崩** |
| ESP32 实拍 | — | 所有 6 点 conf<0.6 | 坐姿检测全失败（n_valid=0） |

**关键反差**：PCK 损失小（位置 argmax 基本对，肩膀 PCK 0.90），但 conf 崩（峰值被 int8 压垮）。ESP32 后处理用 conf≥0.6 阈值过滤，conf 全 <0.6 → 全部过滤 → 检测失败。

ESP32 实拍日志（坐姿正常）：
```
kp[0] conf=0.375  kp[1] conf=0.320  (眼)
kp[2] conf=0.180  kp[3] conf=0.180  (耳)
kp[4] conf=0.266  kp[5] conf=0.266  (肩)
not reliable: n_valid=0 shoulders_ok=0
```

## 2. 分关键点量化损失（PC cam）

| 关键点 | 浮点 PCK | 量化 PCK | 损失 | 漂移中位 | 漂移 90% |
|---|---|---|---|---|---|
| 左/右眼 | 1.000 | 0.994/0.999 | 0.001-0.006 | 4-7px | 10-13px |
| 左/右耳 | 0.999/1.000 | 0.994/0.999 | 0.001-0.004 | 4-6px | 10-15px |
| 左肩 | 0.929 | 0.902 | +0.027 | 8px | 16px |
| 右肩 | 0.927 | 0.913 | +0.014 | 7px | 13px |

PCK 损失主要在肩膀，但**所有点 conf 都崩**（峰值问题，不是位置问题）。

## 3. 逐层量化误差（找元凶）

ESP-PPQ layerwise error（单层独立误差，越大越差）：

| 层 | 量化误差 | 说明 |
|---|---|---|
| node_Conv_629 | **35.7%** | head DWSeparable depthwise |
| node_Conv_627 | 34.9% | head DWSeparable depthwise |
| node_Conv_625 | 34.8% | head DWSeparable depthwise |
| node_Conv_623 | 25.5% | head DWSeparable depthwise |
| node_Conv_621 | 18.1% | head 1×1 |
| node_conv2d_39 | 9.2% | 输出层（32→6） |

**元凶：head 的 DWSeparable depthwise conv（误差 25-36%）**。depthwise conv 每通道独立量化，int8 下误差累积，把 heatmap 峰值压垮。

## 4. 根因

新模型为减 FLOPs 做的设计，叠加导致 int8 极不友好：

1. **head 用 DWSeparable（depthwise）**：depthwise int8 量化误差极大（25-36%），是 conf 崩主因。
2. **sigma=4 窄峰**：高斯峰尖锐（FWHM 19px），int8 量化压低峰值 logit → Sigmoid 后 conf 从 0.9 掉到 0.3。
3. **Resize 上采样（int8 nearest）**：可能有额外量化误差。
4. **模型小（0.13M）**：冗余少，量化容错低。

对比旧 4 点模型（普通 Conv + sigma=6 + ConvTranspose）量化 conf **没崩**（损失仅 0.23%）——结构差异是根因。

## 5. 尝试：head int16 混合精度（失败）

把 head 的 6 个 Conv（621/623/625/627/629/conv2d_39）保 int16，其余 int8：

| 图 | 浮点 conf | 纯 int8 | head int16 |
|---|---|---|---|
| test_img/1.jpg | [0.96,0.92,0.91,0.91,0.76,0.72] | [0.38,0.23,0.18,0.15,0.12,0.27] | [0.27,0.32,0.27,0.32,0.23,0.44] |

**head int16 没救回来**（仍 0.23-0.44，<0.6）。原因：depthwise 即使 int16 也难量化 + 上游 backbone/Resize 误差累积。

## 6. 解法（方案1，推荐）

**sigma 4→5 + head 去掉 DWSeparable 改普通 Conv3×3**：

- **sigma 5**（宽峰，量化友好）：峰不尖锐，int8 压不垮。
- **head 普通 Conv3×3**（替代 DWSeparable）：普通 Conv int8 量化误差远小于 depthwise（且 ESP32 AI 指令对普通 Conv 加速好）。
- 改动：`model.py` head 部分 + `kp_config.py` SIGMA=5，重训。

**速度影响**：
- head FLOPs：DWSeparable 68.7M → 普通 Conv3×3 ~195M（+126M）。
- 模型总：388M → ~514M（+33%）。
- ESP32 推理：当前 1.38s → 约 1.8s（普通 Conv 效率高部分抵消 FLOPs 增加）。
- **用户接受 <3s，方案1 后 ~1.8s 仍达标**。

## 7. 速度现状与目标

| 项 | 值 |
|---|---|
| 当前 ESP32 推理 | 1.38s（388M MACs）|
| ESP32 有效算力 | 0.28 GMAC/s（理论 ~4 的 7%，depthwise + Resize 拖累）|
| 用户可接受 | <3s ✓（当前达标）|
| 原目标 | 150-300ms（未达，需更彻底减 depthwise）|

当前 1.38s 已在用户可接受范围（<3s）。方案1 后约 1.8s，仍达标。若后续要冲 150-300ms，需 backbone 也减 depthwise（换普通 Conv 或更激进下采样减层）。

## 8. 待办

- [ ] 方案1：`model.py` head 换普通 Conv3×3 + `kp_config.py` SIGMA=5 → 重训 → 量化验证 conf 恢复（应 >0.6）。
- [ ] 重训后 ESP32 实测 conf + 速度确认。
- [ ] 若 conf 仍偏低：进一步 sigma=6 或 backbone depthwise 也调整。

## 附：关键命令

```bash
# 量化分关键点损失 + 漂移
python3 quant_compare_kp.py
# 逐层量化误差（找敏感层）
python3 -c "from esp_ppq.api import espdl_quantize_onnx; espdl_quantize_onnx(..., error_report=True, ...)"
# 浮点 vs int8 vs head16 conf 对比
python3 quant_conf_fix.py
# 量化前后可视化（cam_data2）
python3 vis_cam2_quant.py   # -> output/vis_cam2/
# 最新浮点模型可视化（cam_data 带真值）
python3 vis_cam.py          # -> output/vis_cam/
```
