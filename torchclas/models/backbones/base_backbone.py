# coding=utf-8
# @Time   : 2020/10/28 9:58
# @Auto   : zzf-jeff

import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone
    网络的基类，可以添加一些通用的保存权重，初始化权重等方法

    This class defines the basic functions of a backbone.
    Any backbone that inherits this class should at least
    define its own `forward` function.

    """

    def __init__(self):
        super(BaseBackbone, self).__init__()

    ##  用abstractmethod保证实例化的子类必须含有forward方法
    @abstractmethod
    def forward(self, x):
        """Forward computation

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass
