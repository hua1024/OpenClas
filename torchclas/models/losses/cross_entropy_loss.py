# coding=utf-8  
# @Time   : 2020/11/10 9:33
# @Auto   : zzf-jeff

import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label)
    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss = LogSoftmax+NLLLoss
        交叉熵loss加入权重项，类别数据不均衡设置不同权重的一种方案
        TODO : 还未完成

    Args:
        reduction : F.cross_entropy-->reduction
        loss_weight : weights of loss

    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion = cross_entropy

    def forward(self, pred, label, weight=None, ):
        loss = self.loss_weight * self.criterion(pred, label, weight, reduction=self.reduction, )
        return loss
