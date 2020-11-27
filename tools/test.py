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

from torchclas.models import (build_backbone, build_loss)
from torchclas.datasets import (build_dataloader, build_dataset)
from torchclas.utils.io_func import config_load
from torchclas.utils.torch_utils import select_device
from tools.program import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--config', help='train config file path', default='config/resnet50.yaml')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = config_load(args.config)
    set_random_seed(cfg['BASE']['seed'])
    device = select_device(cfg['BASE']['gpu_id'], cfg['TRAIN']['batch_size'])

    model = build_backbone(cfg['BACKBONES'])
    model.load_state_dict(torch.load(args.weights, map_location=device)['state_dict'])
    model = model.to(device)

    test_dataset = build_dataset(cfg['TEST']['dataset'])
    test_loader = build_dataloader(test_dataset, batch_size=cfg['TEST']['batch_size'],
                                   num_workers=cfg['TEST']['num_workers'], shuffle=cfg['TEST']['shuffle'])

    num_class = len(cfg['TEST']['dataset']['classes'])
    top_k = 1 if num_class > 5 else num_class

    correct_1 = 0.0
    correct_n = 0.0
    total = 0

    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))

    # 常规计算top-1/top-n
    for idx, data in tqdm(enumerate(test_loader)):
        images = data['img']
        labels = data['label']
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, pred = outputs.topk(top_k, 1, largest=True, sorted=True)
        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()
        # compute top 5
        correct_n += correct[:, :top_k].sum()
        # compute top1
        correct_1 += correct[:, :1].sum()

    mean_acc = (correct_1 / len(test_loader.dataset)).item()
    print("Accuracy of mean all : %2d %%" % (100 * mean_acc))
    print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    if top_k > 1:
        print("Top {} err: ".format(top_k), 1 - correct_n / len(test_loader.dataset))



if __name__ == '__main__':
    main()
