# coding=utf-8  
# @Time   : 2020/12/14 9:34
# @Auto   : zzf-jeff

from .base_backbone import BaseBackbone
from ..builder import BACKBONES

import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out




class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        inner_channels = in_channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=inner_channels, out_channels=in_channels, kernel_size=1, bias=True)
        self.relu2 = HSigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn


class Block(nn.Module):
    def __init__(self, kernel_size, in_channels, inner_channels, out_channels, linear, se_module, stride):
        super().__init__()
        self.se = se_module
        self.stride = stride
        self.inverted_residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            linear,
            nn.Conv2d(inner_channels, inner_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                      groups=inner_channels, bias=False),
            nn.BatchNorm2d(inner_channels),
            linear,
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = None
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.inverted_residual(x)
        if self.se != None:
            out = self.se(out)
        if self.stride == 1 and self.shortcut is not None:
            out += self.shortcut(x)

        return out


@BACKBONES.register_module()
class MobileNetV3(BaseBackbone):
    def __init__(self, in_channels, num_classes, mode):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = HSwish()

        if mode == 'large':
            self.bneck = nn.Sequential(
                Block(3, 16, 16, 16, linear=nn.ReLU(inplace=True), se_module=None, stride=1),
                Block(3, 16, 64, 24, linear=nn.ReLU(inplace=True), se_module=None, stride=2),
                Block(3, 24, 72, 24, linear=nn.ReLU(inplace=True), se_module=None, stride=1),
                Block(5, 24, 72, 40, linear=nn.ReLU(inplace=True), se_module=SEModule(40), stride=2),
                Block(5, 40, 120, 40, linear=nn.ReLU(inplace=True), se_module=SEModule(40), stride=1),
                Block(5, 40, 120, 40, linear=nn.ReLU(inplace=True), se_module=SEModule(40), stride=1),
                Block(3, 40, 240, 80, linear=HSwish(), se_module=None, stride=2),
                Block(3, 80, 200, 80, linear=HSwish(), se_module=None, stride=1),
                Block(3, 80, 184, 80, linear=HSwish(), se_module=None, stride=1),
                Block(3, 80, 184, 80, linear=HSwish(), se_module=None, stride=1),
                Block(3, 80, 480, 112, linear=HSwish(), se_module=SEModule(112), stride=1),
                Block(3, 112, 672, 112, linear=HSwish(), se_module=SEModule(112), stride=1),
                Block(5, 112, 672, 160, linear=HSwish(), se_module=SEModule(160), stride=1),
                Block(5, 160, 672, 160, linear=HSwish(), se_module=SEModule(160), stride=2),
                Block(5, 160, 960, 160, linear=HSwish(), se_module=SEModule(160), stride=1),
            )
            self.conv2 = nn.Conv2d(160, 960, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(960)
            self.hs2 = HSwish()
            self.linear1 = nn.Linear(960, 1280)
            self.bn3 = nn.BatchNorm2d(1280)
            self.hs3 = HSwish()
            self.linear2 = nn.Linear(1280, num_classes)
        elif mode == 'small':
            self.bneck = nn.Sequential(
                Block(3, 16, 16, 16, nn.ReLU(inplace=True), SEModule(16), 2),
                Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
                Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
                Block(5, 24, 96, 40, HSwish(), SEModule(40), 2),
                Block(5, 40, 240, 40, HSwish(), SEModule(40), 1),
                Block(5, 40, 240, 40, HSwish(), SEModule(40), 1),
                Block(5, 40, 120, 48, HSwish(), SEModule(48), 1),
                Block(5, 48, 144, 48, HSwish(), SEModule(48), 1),
                Block(5, 48, 288, 96, HSwish(), SEModule(96), 2),
                Block(5, 96, 576, 96, HSwish(), SEModule(96), 1),
                Block(5, 96, 576, 96, HSwish(), SEModule(96), 1),
            )
            self.conv2 = nn.Conv2d(96, 576, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(576)
            self.hs2 = HSwish()
            self.linear1 = nn.Linear(576, 1280)
            self.bn3 = nn.BatchNorm2d(1280)
            self.hs3 = HSwish()
            self.linear2 = nn.Linear(1280, num_classes)
        else:
            raise NotImplementedError

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hs1(out)
        out = self.bneck(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.hs2(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        # out = self.bn3(out)
        out = self.hs3(out)
        out = self.linear2(out)
        return out
