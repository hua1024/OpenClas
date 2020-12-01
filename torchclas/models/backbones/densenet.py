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
    "DenseNet"
]


class Bottleneck(nn.Module):
    """Dense bottleneck
    DenseNet网络的块结构，bn+relu+conv,1x1+3x3

    """

    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inner_channels = growth_rate * 4
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)


@BACKBONES.register_module()
class DenseNet(BaseBackbone):
    arch_settings = {
        121: (Bottleneck, 32, [6, 12, 24, 16]),
        161: (Bottleneck, 48, [6, 12, 36, 24]),
        169: (Bottleneck, 32, [6, 12, 32, 32]),
        201: (Bottleneck, 32, [6, 12, 48, 32]),
        264: (Bottleneck, 32, [6, 12, 64, 48])
    }

    def __init__(self, depth, in_channels, reduction=0.5, num_classes=1000):
        super(DenseNet, self).__init__()
        (self.block, self.growth_rate, self.num_block) = DenseNet.arch_settings[depth]

        inner_channels = 2 * self.growth_rate

        # first conv
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, inner_channels, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn0', nn.BatchNorm2d(inner_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # dense block
        for idx in range(len(self.num_block)):
            self.features.add_module('dense_block_layer_{}'.format(idx),
                                     self._make_layer(self.block, inner_channels, self.num_block[idx]))
            inner_channels += self.growth_rate * self.num_block[idx]

            if idx != len(self.num_block) - 1:
                out_channels = int(reduction * inner_channels)  # int() will automatic floor the value
                self.features.add_module('transition_layer_{}'.format(idx), Transition(inner_channels, out_channels))
                inner_channels = out_channels

        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

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

    def _make_layer(self, block, in_channels, num_block):
        dense_block = nn.Sequential()
        for idx in range(num_block):
            dense_block.add_module('bottle_neck_layer_{}'.format(idx), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
