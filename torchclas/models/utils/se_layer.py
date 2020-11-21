# coding=utf-8  
# @Time   : 2020/11/5 12:21
# @Auto   : zzf-jeff

import torch.nn as nn


class SELayer(nn.Module):
    '''
    通道注意力机制 ： 通过在fea
    '''

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # inplace=True 不创建新内存，直接在原来上修改
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        # shape ==> b,c,1,1
        y = y.view(b, c)
        # shape ==> b,c
        # 为了相乘.reshape b,c,1,1
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
