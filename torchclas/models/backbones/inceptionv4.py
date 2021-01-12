# coding=utf-8  
# @Time   : 2020/12/24 9:21
# @Auto   : zzf-jeff


import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, padding=0, stride=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception_Stem(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3,padding=1),
            BasicConv2d(32, 64, kernel_size=3)
        )
        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=2)
        self.branch7x7a = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3)
        )
        self.branch7x7b = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3)
        )
        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = [
            self.branch3x3_conv(x),
            self.branch3x3_pool(x)
        ]
        x = torch.cat(x, 1)
        x = [
            self.branch7x7a(x),
            self.branch7x7b(x)
        ]
        x = torch.cat(x, 1)
        x = [
            self.branchpoola(x),
            self.branchpoolb(x)
        ]
        x = torch.cat(x, 1)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv2d(in_channels, 96, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 96, kernel_size=1)
        )

    def forward(self, x):
        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branch1x1(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.branch7x7stack = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch1x1 = BasicConv2d(in_channels, 384, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 128, kernel_size=1)
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7stack(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))

        self.branch1x1 = BasicConv2d(in_channels, 256, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 256, kernel_size=1)
        )

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output),
            self.branch3x3stackb(branch3x3stack_output)
        ]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)

        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output),
            self.branch3x3b(branch3x3_output)
        ]
        branch3x3_output = torch.cat(branch3x3_output, 1)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)


class ReductionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, k, kernel_size=1),
            BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2)
        )
        self.branch3x3 = BasicConv2d(in_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # 因为channels是传参进来的，所以记录下最终的output
        self.out_channels = in_channels + n + m

    def forward(self, x):
        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1)
        )
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = [
            self.branch3x3(x),
            self.branch7x7(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


def make_layers(in_channels, out_channels, block_num, block):
    layers = []
    for l in range(block_num):
        layers.append(block(in_channels))
        in_channels = out_channels
    return nn.Sequential(*layers)


@BACKBONES.register_module()
class InceptionV4(BaseBackbone):
    def __init__(self, in_channels, num_classes):
        # A = 3,B = 7,C = 3
        # k=192, l=224, m=256, n=384
        # 这里把一些配置写死了，之前有改过发现效果变差了
        super().__init__()
        self.stem = Inception_Stem(in_channels)
        self.inception_a = make_layers(384, 384, block_num=3, block=InceptionA)
        self.reduction_a = ReductionA(384, k=192, l=224, m=256, n=384)
        out_channels = self.reduction_a.out_channels
        self.inception_b = make_layers(out_channels, 1024, block_num=7, block=InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = make_layers(1536, 1536, block_num=3, block=InceptionC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.2)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
