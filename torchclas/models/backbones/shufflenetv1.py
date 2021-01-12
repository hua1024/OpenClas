# coding=utf-8  
# @Time   : 2020/12/18 16:06
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


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.depthwise(x)


class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(PointwiseConv2d, self).__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.pointwise(x)


class ChannelShuffle(nn.Module):
    """ChannelShuffle
    通道打乱，假设将输入分为g组，总通道数为gxn,首先将通道拆分为(g,n)两个维度
    然后将这两个维度转置为(n,g),最后重新reshape为一个维度gxn


    """

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        # transpose交换位置，后面如果有view操作要接contiguous()
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stage, stride, groups):
        super(ShuffleNetUnit, self).__init__()
        self.pointwise1 = nn.Sequential(
            PointwiseConv2d(in_channels, int(out_channels / 4), kernel_size=1, stride=1, padding=0, groups=groups),
            nn.ReLU(inplace=True)
        )

        self.channel_shuffle = ChannelShuffle(groups)
        self.depthwise = DepthwiseConv2d(
            int(out_channels / 4), int(out_channels / 4), kernel_size=3, stride=stride, padding=1,
            groups=int(out_channels / 4))
        self.pointwise2 = PointwiseConv2d(
            int(out_channels / 4), out_channels, kernel_size=1, stride=1, padding=0, groups=groups
        )
        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.pointwise2 = PointwiseConv2d(
                int(out_channels / 4), (out_channels - in_channels), kernel_size=1, stride=1, padding=0, groups=groups
            )
            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        short_cut = self.shortcut(x)
        y = self.pointwise1(x)
        y = self.channel_shuffle(y)
        y = self.depthwise(y)
        y = self.pointwise2(y)
        y = self.fusion(short_cut, y)
        y = self.relu(y)
        return y


@BACKBONES.register_module()
class ShuffleNetV1(BaseBackbone):
    arch_settings = [[4, 8, 4], ShuffleNetUnit]

    def __init__(self, in_channels, num_classes=1000, groups=3):
        super(ShuffleNetV1, self).__init__()
        self.block = self.arch_settings[1]
        self.num_block = self.arch_settings[0]

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]
        else:
            print("Groups only support 1,2,3,4,8", )
            raise

        self.conv1 = ConvBnRelu(in_channels, out_channels[0], kernel_size=3, stride=2, padding=1, groups=1,
                                is_relu=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inner_channels = out_channels[0]
        self.stage2 = self._make_layer(ShuffleNetUnit, self.num_block[0], out_channels[1], stride=2, stage=2,
                                       groups=groups)
        self.stage3 = self._make_layer(ShuffleNetUnit, self.num_block[1], out_channels[2], stride=2, stage=3,
                                       groups=groups)
        self.stage4 = self._make_layer(ShuffleNetUnit, self.num_block[2], out_channels[3], stride=2, stage=4,
                                       groups=groups)
        self.avg = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, block, num_blocks, out_channels, stride, stage, groups):
        strides = [stride] + [1] * (num_blocks - 1)
        layer = []
        for _stride in strides:
            layer.append(
                block(self.inner_channels, out_channels, stage, _stride, groups)
            )
            self.inner_channels = out_channels
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
