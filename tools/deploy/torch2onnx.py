# coding=utf-8  
# @Time   : 2020/10/27 17:09
# @Auto   : zzf-jeff

import sys
import os

sys.path.append('./')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import onnx
import torch
import argparse
import cv2
from torchclas.utils.io_func import (config_load)
from torchclas.models import (build_backbone)


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--config', type=str, help='train config file path', default='config/resnet50.yaml')
    parser.add_argument('--weights', type=str, help='the model weights')
    parser.add_argument('--output', type=str, help='onnx save path')
    args = parser.parse_args()
    return args


def _torch2onnx(model, dummy_input, onnx_model_name, input_names, output_names, opset_version=9,
                do_constant_folding=False, verbose=False, is_dynamic=True):
    """convert PyTorch model to Onnx
    Args:
        model (torch.nn.Module): PyTorch model.
        dummy_input (torch.Tensor, tuple or list): dummy input.
        onnx_model_name (string or io object): saved Onnx model name.
        opset_version (int, default is 9): Onnx opset version.
        do_constant_folding (bool, default False): 是否执行常量折叠优化,If True, the
            constant-folding optimization is applied to the model during
            export. Constant-folding optimization will replace some of the ops
            that have all constant inputs, with pre-computed constant nodes.
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        is_dynamic : 批处理变量,指定则使用批处理或者固定批处理大小，默认不使用
    """
    # dynamic_axes = {name: [0] for name in input_names + output_names}
    # dynamic_axes = {"input": {0: "batch_size"},  # 批处理变量
    #                 "output": {0: "batch_size"}}
    if is_dynamic:
        # 批处理变量
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    else:
        dynamic_axes = None

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        dynamic_axes=dynamic_axes)


def main():
    args = parse_args()
    device = torch.device('cuda:0')
    cfg = config_load(args.config)
    model = build_backbone(cfg['BACKBONES'])
    model.load_state_dict(torch.load(args.weights, map_location=device)['state_dict'])
    model = model.to(device)
    input_data = torch.randn(4, 3, 224, 224).to(device)
    model.eval()
    output = args.output
    _torch2onnx(model=model, dummy_input=input_data, onnx_model_name=output, input_names=['input'],
                output_names=['output'], opset_version=9)
    onnx_model = onnx.load(output)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    print("Model was successfully converted to ONNX format.")
    print("It was saved to", output)


if __name__ == '__main__':
    main()
