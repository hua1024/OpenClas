# coding=utf-8  
# @Time   : 2020/12/10 14:32
# @Auto   : zzf-jeff

from .base_backbone import BaseBackbone
from ..builder import BACKBONES

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthSeperabelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False, **kwargs),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


@BACKBONES.register_module()
class MobileNetV1(BaseBackbone):
    def __init__(self, in_channels, width_multiplier=1, num_classes=1000):
        super(MobileNetV1, self).__init__()
        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(in_channels, int(32 * alpha), kernel_size=3, stride=2, padding=1),
            DepthSeperabelConv2d(in_channels=int(32 * alpha), out_channels=int(64 * alpha), kernel_size=3, stride=1,
                                 padding=1)
        )

        self.conv1 = nn.Sequential(
            DepthSeperabelConv2d(in_channels=int(64 * alpha), out_channels=int(128 * alpha), kernel_size=3, stride=2,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(128 * alpha), out_channels=int(128 * alpha), kernel_size=3, stride=1,
                                 padding=1),
        )
        self.conv2 = nn.Sequential(
            DepthSeperabelConv2d(in_channels=int(128 * alpha), out_channels=int(256 * alpha), kernel_size=3, stride=2,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(256 * alpha), out_channels=int(256 * alpha), kernel_size=3, stride=1,
                                 padding=1),
        )
        self.conv3 = nn.Sequential(
            DepthSeperabelConv2d(in_channels=int(256 * alpha), out_channels=int(512 * alpha), kernel_size=3, stride=2,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha), kernel_size=3, stride=1,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha), kernel_size=3, stride=1,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha), kernel_size=3, stride=1,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha), kernel_size=3, stride=1,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha), kernel_size=3, stride=1,
                                 padding=1),
        )
        self.conv4 = nn.Sequential(
            DepthSeperabelConv2d(in_channels=int(512 * alpha), out_channels=int(1024 * alpha), kernel_size=3, stride=2,
                                 padding=1),
            DepthSeperabelConv2d(in_channels=int(1024 * alpha), out_channels=int(1024 * alpha), kernel_size=3, stride=1,
                                 padding=1),
        )
        # 按照原论文
        # self.avg = nn.AdaptiveAvgPool2d(1)
        self.avg = nn.AvgPool2d(7)
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
