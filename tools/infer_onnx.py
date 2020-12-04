# coding=utf-8  
# @Time   : 2020/12/3 17:31
# @Auto   : zzf-jeff


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('./')

import argparse
import copy
import os
import time
import torch
from tqdm import tqdm
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import onnxruntime

from torchclas.models import build_backbone
from torchclas.utils.io_func import config_load
from torchclas.utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--config', type=str, help='train config file path', default='config/resnet50.yaml')
    parser.add_argument('--source', type=str, help='inference img path')
    parser.add_argument('--weights', type=str, help='the model onnx weights')
    args = parser.parse_args()
    return args


def get_output_name(onnx_session):
    """
    output_name = onnx_session.get_outputs()[0].name
    :param onnx_session:
    :return:
    """
    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)
    return output_name


def get_input_name(onnx_session):
    """
    input_name = onnx_session.get_inputs()[0].name
    :param onnx_session:
    :return:
    """
    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)
    return input_name


def get_input_feed(input_name, image_numpy):
    """
    input_feed={self.input_name: image_numpy}
    :param input_name:
    :param image_numpy:
    :return:
    """
    input_feed = {}
    for name in input_name:
        input_feed[name] = image_numpy
    return input_feed


def main():
    args = parse_args()
    img = Image.open(args.source)
    img = img.convert('RGB')

    cfg = config_load(args.config)
    device = select_device(cfg['BASE']['gpu_id'], cfg['TRAIN']['batch_size'])
    onnx_session = onnxruntime.InferenceSession(args.weights)
    input_name = get_input_name(onnx_session)
    output_name = get_output_name(onnx_session)
    print("input_name:{}".format(input_name))
    print("output_name:{}".format(output_name))

    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    input_feed = get_input_feed(input_name, to_numpy(img))


    output = onnx_session.run(output_name, input_feed=input_feed)
    output = torch.from_numpy(output[0])

    prob = F.softmax(output, dim=1)
    value, predicted = torch.max(output.data, 1)
    pred_class = cfg['TEST']['dataset']['classes'][predicted.item()]
    pred_score = prob[0][predicted.item()].item()
    print(pred_class, pred_score)




if __name__ == '__main__':
    main()
