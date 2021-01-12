# coding=utf-8  
# @Time   : 2020/11/9 15:25
# @Auto   : zzf-jeff

from .lenet import LeNet5
from .resnet import ResNet
from .resnet_vd import ResNetVd
from .vgg import VGG
from .densenet import DenseNet
from .mobilenetv1 import MobileNetV1
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3
from .resnext import ResNeXt
from .shufflenetv1 import ShuffleNetV1
from .shufflenetv2 import ShuffleNetV2
from .inceptionv3 import InceptionV3
from .inceptionv4 import InceptionV4
from .resnet_inception_v2 import InceptionResNetV2
from .senet import SeResNet

__all__ = [
    'LeNet5',
    'ResNet',
    'ResNetVd',
    'VGG',
    'DenseNet',
    'MobileNetV1',
    'MobileNetV2',
    'MobileNetV3',
    'ResNeXt',
    'ShuffleNetV1',
    'ShuffleNetV2',
    'InceptionV3',
    'InceptionV4',
    'InceptionResNetV2'
]
