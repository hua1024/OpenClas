# coding=utf-8  
# @Time   : 2020/11/9 15:45
# @Auto   : zzf-jeff

import torch.nn as nn
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class LeNet5(BaseBackbone):
    def __init__(self, in_channel, num_classes=1000):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            # nn.Dropout(p=0.4),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            # nn.Dropout(p=0.4),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
