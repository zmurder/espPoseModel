"""验证 output/current.onnx 的输出是否等于 best.pth（确认量化链路源头）"""
import numpy as np
import torch
import cv2
import onnx
import onnxruntime as ort

from kp_config import IMG_MEAN, IMG_STD, IMG_WIDTH, IMG_HEIGHT
from model import PoseNet
from dataset import center_crop_resize


def main():
    img = cv2.cvtColor(cv2.imread('test_img/1.jpg'), cv2.COLOR_BGR2RGB)
    inp, *_ = center_crop_resize(img, IMG_WIDTH, IMG_HEIGHT)
    x = inp.astype(np.float32) / 255.0
    x = (x - np.array(IMG_MEAN, dtype=np.float32)) / np.array(IMG_STD, dtype=np.float32)
    x_np = x.transpose(2, 0, 1)[None]

    # best.pth
    m = PoseNet().eval()
    sd = torch.load('checkpoints/best.pth', map_location='cpu')
    m.load_state_dict(sd['model'] if 'model' in sd else sd)
    with torch.no_grad():
        hm_torch = m(torch.from_numpy(x_np)).numpy()

    # onnx
    m_onnx = onnx.load('output/current.onnx')
    ext = [t for t in m_onnx.graph.initializer if t.data_location == 1]
    print(f'onnx external data tensors: {len(ext)} (0=自含权重)')

    sess = ort.InferenceSession('output/current.onnx', providers=['CPUExecutionProvider'])
    hm_onnx = sess.run(None, {'input': x_np})[0]

    diff = np.abs(hm_torch - hm_onnx)
    print(f'best.pth vs onnx  heatmap max diff: {diff.max():.6f}  mean: {diff.mean():.6f}')
    print(f'-> {"[OK] onnx == best.pth, 量化链路源头正确" if diff.max() < 1e-3 else "[警告] 不一致, 需重新 export"}')

    # decode 对比
    pk, conf = m(torch.from_numpy(x_np)).__class__  # noqa
    hm = torch.from_numpy(hm_onnx)
    kps, c = PoseNet().decode(PoseNet(), hm)
    print(f'onnx decode 关键点(归一化): {kps[0].tolist()}')
    print(f'onnx decode 置信度: {[round(v,3) for v in c[0].tolist()]}')


if __name__ == '__main__':
    main()
