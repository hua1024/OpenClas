# coding=utf-8  
# @Time   : 2020/10/24 12:13
# @Auto   : zzf-jeff


'''
1.论文resnet结构，参考torchvision
2.相对于原论文的修改
stride=2，放到第二个1x1卷积
# 替换第一层7*7-->3个3*3，这个同步到resnet_vd重，resnet取消使用，感觉对于28*28的输入图片，第一个卷积7*7会太大
resnet18 : [2, 2, 2, 2]
resnet34 : [3, 4, 6, 3]
resnet50 : [3, 4, 6, 3]
resnet101 : [3, 4, 23, 3]
resnet152 : [3, 8, 36, 3]
'''
import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

__all__ = [
    "ResNet"
]


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


# 用于18，34的block结果，使用2个3*3卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=1, is_relu=True)
        self.conv2 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels * BasicBlock.expansion,
                                kernel_size=3, padding=1, stride=1, is_relu=False)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * BasicBlock.expansion,
                                 stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
# 这里大部分复现都是将放在第一个卷积的下采样放到了第二个卷积上，实测确实效果好很多，这里默认修改

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                padding=0, is_relu=True)
        self.conv2 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=1, is_relu=True)
        self.conv3 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels * BottleneckBlock.expansion,
                                kernel_size=1, stride=1, padding=0, is_relu=False)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * BottleneckBlock.expansion,
                                 stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = y + self.shortcut(x)
        return self.relu(y)


@BACKBONES.register_module()
class ResNet(BaseBackbone):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (BottleneckBlock, (3, 4, 6, 3)),
        101: (BottleneckBlock, (3, 4, 23, 3)),
        152: (BottleneckBlock, (3, 8, 36, 3))
    }

    def __init__(self, depth, in_channels, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.block = ResNet.arch_settings[depth][0]
        self.num_block = ResNet.arch_settings[depth][1]
        # padding = 3 才能输出原论文的112
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
            layers.append(block(self.in_channels, out_channels, _stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x2 = self.conv2_x(x)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)
        out = self.avg_pool(x5)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
