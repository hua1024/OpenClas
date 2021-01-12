# coding=utf-8  
# @Time   : 2020/12/21 10:10
# @Auto   : zzf-jeff


from .base_backbone import BaseBackbone
from ..builder import BACKBONES

import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F


# conv+bn+relu
class ConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, is_relu=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if is_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


def channel_split(x, split):
    # c -->c1,c2
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleNetUnit, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        branch_features = int(out_channels / 2)
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, branch_features, kernel_size=1),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, padding=1,
                          groups=branch_features),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()
            self.residual = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, kernel_size=1),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, padding=1,
                          groups=branch_features),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut, residual = x, x
        short_cut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([short_cut, residual], dim=1)
        x = channel_shuffle(x, groups=2)
        return x


@BACKBONES.register_module()
class ShuffleNetV2(BaseBackbone):
    arch_settings = [[4, 8, 4], ShuffleNetUnit]

    def __init__(self, in_channels, ratio=1, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.block = ShuffleNetV2.arch_settings[1]
        self.num_block = ShuffleNetV2.arch_settings[0]
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            print("Ratio only support 0.5,1,1.5,2", )
            raise
        self.inner_channels = 24
        self.conv1 = ConvBnRelu(in_channels, 24, kernel_size=3, stride=2, padding=1, is_relu=True)
        self.pool1 = self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_layer(ShuffleNetUnit, self.num_block[0], out_channels[0], stride=2)
        self.stage3 = self._make_layer(ShuffleNetUnit, self.num_block[1], out_channels[1], stride=2)
        self.stage4 = self._make_layer(ShuffleNetUnit, self.num_block[2], out_channels[2], stride=2)

        self.conv2 = ConvBnRelu(out_channels[2], out_channels[3], kernel_size=1, is_relu=True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.stage2(x)
        # print(x.shape)
        x = self.stage3(x)
        # print(x.shape)
        x = self.stage4(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # def _make_layer(self, in_channels, out_channels, repeat):
    #     layers = []
    #     layers.append(ShuffleNetUnit(in_channels, out_channels, 2))
    #     while repeat:
    #         layers.append(ShuffleNetUnit(out_channels, out_channels, 1))
    #         repeat -= 1
    #     return nn.Sequential(*layers)

    def _make_layer(self, block, num_blocks, out_channels, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layer = []
        for _stride in strides:
            layer.append(
                block(self.inner_channels, out_channels, _stride)
            )
            self.inner_channels = out_channels
        return nn.Sequential(*layer)
