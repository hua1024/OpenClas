# coding=utf-8  
# @Time   : 2020/12/10 18:51
# @Auto   : zzf-jeff

from .base_backbone import BaseBackbone
from ..builder import BACKBONES

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super().__init__()
        self.inverted_residual = nn.Sequential(
            # 1x1
            nn.Conv2d(in_channels, in_channels * t, kernel_size=1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            # 3x3
            nn.Conv2d(in_channels * t, in_channels * t, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            # 1x1 + remove relu
            nn.Conv2d(in_channels * t, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        inverted_residual = self.inverted_residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            inverted_residual += x

        return inverted_residual


@BACKBONES.register_module()
class MobileNetV2(BaseBackbone):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.layer1 = LinearBottleNeck(32, 16, 1, 1)
        self.layer2 = self._make_layer(2, 16, 24, 2, 6)
        self.layer3 = self._make_layer(3, 24, 32, 2, 6)
        self.layer4 = self._make_layer(2, 32, 64, 2, 6)
        self.layer5 = self._make_layer(2, 64, 96, 1, 6)
        self.layer6 = self._make_layer(2, 96, 160, 2, 6)
        self.layer7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        self.avg = nn.AvgPool2d(7)
        self.fc = nn.Linear(1280, num_classes)

    def _make_layer(self, n, in_channels, out_channels, stride, t):
        layers = []
        strides = [stride] + [1] * (n - 1)
        for stride in strides:
            layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.conv2(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
