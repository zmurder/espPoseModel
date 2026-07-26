"""
导出 ONNX + onnxsim 融合 BN + 算子检查
==================================
导出 opset18, onnxsim 融合 BatchNorm 进 Conv, 检查所有算子 ESP-DL 是否支持。

用法:
    python3 export_onnx.py --model_path checkpoints/best.pth --output output/pose_model_6kp.onnx
"""
import os
import argparse

import torch
import onnx

from model import PoseNet


def main():
    parser = argparse.ArgumentParser(description='导出 ONNX')
    parser.add_argument('--model_path', default='checkpoints/best.pth')
    parser.add_argument('--output', default='output/pose_model_6kp.onnx')
    parser.add_argument('--opset', type=int, default=18)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model = PoseNet()
    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval()

    dummy = torch.randn(1, 3, 240, 320)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=['input'], output_names=['heatmap'],
        opset_version=args.opset,
        dynamic_axes=None,  # 固定 batch=1（ESP-DL 要求）
    )
    print(f'导出 ONNX: {args.output}')

    # onnxsim 融合 BN
    try:
        import onnxsim
        sim_model, ok = onnxsim.simplify(args.output)
        if ok:
            onnx.save(sim_model, args.output)
            print('onnxsim 融合成功 (BN 已融入 Conv)')
        else:
            print('onnxsim 融合失败, 保留原模型')
    except ImportError:
        print('onnxsim 未安装, 跳过融合')

    # 算子检查
    model_onnx = onnx.load(args.output)
    ops = sorted({n.op_type for n in model_onnx.graph.node})
    print(f'算子: {ops}')

    # ESP-DL 支持的算子（docs/operator_support_state.md）
    espdl_ops = {
        'Conv', 'BatchNormalization', 'Relu', 'Sigmoid', 'Resize',
        'Add', 'Sub', 'Mul', 'Div', 'Concat', 'Reshape', 'Transpose',
        'Flatten', 'ReduceMean', 'ReduceSum', 'Pad', 'MaxPool', 'AveragePool',
    }
    unsupported = [o for o in ops if o not in espdl_ops]
    if unsupported:
        print(f'[警告] 可能不支持的算子: {unsupported}')
    else:
        print('[OK] 所有算子均在 ESP-DL 支持列表内')
    if 'BatchNormalization' in ops:
        print('[警告] 仍有 BatchNormalization 节点, 建议确认 onnxsim 融合')

    print(f'\n输入: input (1,3,240,320)  输出: heatmap (1,6,120,160)')


if __name__ == '__main__':
    main()
