# coding=utf-8  
# @Time   : 2020/10/24 11:12
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

from torchclas.models import build_backbone
from torchclas.utils.io_func import config_load
from torchclas.utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--config', type=str, help='train config file path', default='config/resnet50.yaml')
    parser.add_argument('--source', type=str, help='inference img path')
    parser.add_argument('--weights', type=str, help='the model weights')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = config_load(args.config)
    img = Image.open(args.source)
    img = img.convert('RGB')
    device = select_device(cfg['BASE']['gpu_id'], cfg['TRAIN']['batch_size'])

    model = build_backbone(cfg['BACKBONES'])

    model.load_state_dict(torch.load(args.weights, map_location=device)['state_dict'])

    model = model.to(device)
    model.eval()
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

    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            output = model(img)
    end_time = time.time()
    print(end_time - start_time)



    prob = F.softmax(output, dim=1)
    value, predicted = torch.max(output.data, 1)
    pred_class = cfg['TEST']['dataset']['classes'][predicted.item()]
    pred_score = prob[0][predicted.item()].item()
    print(pred_class, pred_score)
    # print('输入图片为 ：{}'.format(args.source))
    # print('预测的结果为 : {}, 准确率为 : {}'.format(pred_class, str(pred_score)))


if __name__ == '__main__':
    main()
