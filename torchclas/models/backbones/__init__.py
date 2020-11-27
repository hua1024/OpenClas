# coding=utf-8  
# @Time   : 2020/11/9 15:25
# @Auto   : zzf-jeff

from .lenet import LeNet5
from .resnet import ResNet
from .resnet_vd import ResNetVd
from .vgg import VGG

__all__ = [
    'LeNet5',
    'ResNet',
    'ResNetVd',
    'VGG'
]
