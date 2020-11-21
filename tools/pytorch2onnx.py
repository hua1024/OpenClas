# coding=utf-8  
# @Time   : 2020/10/27 17:09
# @Auto   : zzf-jeff

import sys

sys.path.append('./')

import onnx
import torch
from torchclas.models.backbones.resnet_vd import resnet50_vd

device = torch.device('cuda:0')
net = resnet50_vd(3, 2, is_3x3=True).to(device)
input_data = torch.randn(1, 3, 224, 224).to(device)
net.eval()
print(net(input_data))

ONNX_FILE_PATH = "resnet50_vd.onnx"
torch.onnx.export(net, input_data, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
onnx_model = onnx.load(ONNX_FILE_PATH)
# check that the model converted fine
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)
print("Model was successfully converted to ONNX format.")
print("It was saved to", ONNX_FILE_PATH)
