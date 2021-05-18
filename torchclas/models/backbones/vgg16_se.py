# coding=utf-8  
# @Time   : 2020/11/26 18:46
# @Auto   : zzf-jeff

import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

# M: pooling 层
# O: endpoint ,设计用来输出指定层feature map
# C: 自定义的层

cfg = {
    # ori
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # se-vgg
    'S': [64, 64, 'C', 'M', 128, 128, 'C', 'M', 256, 256, 256, 'C',
          'M', 512, 512, 512, 'C', 'M', 512, 512, 512, 'C', 'M', ]
}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def make_layers(cfg_list, in_channel, custom_layer, is_bn=True):
    layers = []
    endpoint_index = []
    for v in cfg_list:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'O':
            endpoint_index.append(len(layers) - 1)
        elif v == 'C':
            # 用pop保证按照顺序使用custom layer
            layers += custom_layer.pop(0)
        else:
            conv = nn.Conv2d(in_channel, v, kernel_size=3, padding=1)
            if is_bn:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channel = v
    return nn.Sequential(*layers), endpoint_index
    # return layers, endpoint_index


@BACKBONES.register_module()
class VGGSeNet(BaseBackbone):
    def __init__(self, in_channels, num_classes=10000, init_weights=False):
        super(VGGSeNet, self).__init__()
        # senet
        custom_layers = [
            [SELayer(channel=64)],
            [SELayer(channel=128)],
            [SELayer(channel=256)],
            [SELayer(channel=512)],
            [SELayer(channel=512)]
        ]
        self.features, self.endpoint_index = make_layers(cfg['S'], in_channels, custom_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modulelist():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
