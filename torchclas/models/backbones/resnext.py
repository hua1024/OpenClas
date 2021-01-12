# coding=utf-8  
# @Time   : 2020/12/16 17:16
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


# 跨层连接
# 经过处理后的x要与x的维度相同(尺寸和深度)
# 如果不相同，需要添加卷积+BN来变换为同一维度
class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShortCut, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.conv = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=0, is_relu=False)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class BottleNeckC(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality=32, group_depth=4):
        super(BottleNeckC, self).__init__()
        # self.cardinality = cardinality
        # self.group_depth = group_depth
        self.inner_channels = int(out_channels * group_depth / 64.) * cardinality
        # self.inner_channels = in_channels * 2

        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, self.inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, stride=stride, groups=cardinality,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * self.expansion,
                                 stride=stride)

    def forward(self, x):
        # 注意细节问题，没留意x被覆盖,查了半天
        # x = self.split_transforms(x)
        # x = x + self.shortcut(x)
        y = self.split_transforms(x)
        y = y + self.shortcut(x)
        return F.relu(y)


@BACKBONES.register_module()
class ResNeXt(BaseBackbone):
    arch_settings = {
        50: (BottleNeckC, (3, 4, 6, 3)),
        101: (BottleNeckC, (3, 4, 23, 3)),
        152: (BottleNeckC, (3, 8, 36, 3))
    }

    def __init__(self, depth, in_channels, num_classes=1000, cardinality=32, group_depth=4, **kwargs):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.group_depth = group_depth
        self.block = self.arch_settings[depth][0]
        self.num_block = self.arch_settings[depth][1]
        self.channels = 64
        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2,
                                is_relu=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block=self.block, out_channels=64, num_blocks=self.num_block[0], stride=1)
        self.conv3_x = self._make_layer(block=self.block, out_channels=128, num_blocks=self.num_block[1], stride=2)
        self.conv4_x = self._make_layer(block=self.block, out_channels=256, num_blocks=self.num_block[2], stride=2)
        self.conv5_x = self._make_layer(block=self.block, out_channels=512, num_blocks=self.num_block[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        '''
        堆叠block网络结构
        '''
        # 除了第一层的block块结构,stride为1,其它都为2,构建strides按照block个数搭建网络
        # [1,2]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(block(self.channels, out_channels, _stride))
            self.channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out
