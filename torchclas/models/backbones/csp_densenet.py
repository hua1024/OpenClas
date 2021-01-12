# coding=utf-8  
# @Time   : 2020/11/27 10:07
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import BACKBONES
import torch.nn.functional as F

from .base_backbone import BaseBackbone
from collections import OrderedDict

__all__ = [
    "CSPDenseNet"
]


class _DenseLayer(nn.Sequential):
    """Dense bottleneck
    DenseNet网络的块结构，bn+relu+conv,1x1+3x3

    """

    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        inner_channels = growth_rate * bn_size
        self.add_module('norm1', nn.BatchNorm2d(in_channels)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(inner_channels, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _CSPDenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate):
        super(_CSPDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)



@BACKBONES.register_module()
class CSPDenseNet(BaseBackbone):
    arch_settings = {
        121: (32, [6, 12, 24, 16]),
        161: (48, [6, 12, 36, 24]),
        169: (32, [6, 12, 32, 32]),
        201: (32, [6, 12, 48, 32]),
        264: (32, [6, 12, 64, 48])
    }

    def __init__(self, depth, in_channels, reduction=0.5, bn_size=4, drop_rate=0, part_ratio=0.5, num_classes=1000):
        super(CSPDenseNet, self).__init__()
        (self.growth_rate, self.num_block) = self.arch_settings[depth]

        inner_channels = 2 * self.growth_rate

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, inner_channels, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(inner_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))



        # dense block
        for idx, num_layers in enumerate(self.num_block):
            block = _DenseBlock(num_layers=num_layers, in_channels=inner_channels,
                                bn_size=bn_size, growth_rate=self.growth_rate, drop_rate=drop_rate)

            self.features.add_module('denseblock%d' % (idx + 1), block)
            inner_channels += self.growth_rate * num_layers

            if idx != len(self.num_block) - 1:
                out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
                self.features.add_module('transition{}%d'.format(idx + 1), _Transition(inner_channels, out_channels))
                inner_channels = out_channels

                # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(inner_channels))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(inner_channels, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
