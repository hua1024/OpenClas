# coding=utf-8  
# @Time   : 2020/11/10 11:10
# @Auto   : zzf-jeff


'''
基于torchvision的transforms
熟悉一下相关API
# TODO ；多尝试自己复现功能，提高能力
'''

import torchvision.transforms as transforms
import inspect
import math
import random
from ..builder import PIPELINES


@PIPELINES.register_module()
class Normalize(object):
    """Image Normalize -->图片归一化操作
    Args:
        mean (float or list):  均值
        std (float or list): 方差
        image ：原始图片
    Returns:
        result :  Over Normalize
    """

    def __init__(self, params):
        self.mean = params['mean']
        self.std = params['std']

    def __call__(self, results):
        transform = transforms.Normalize(mean=self.mean, std=self.std)
        results['image'] = transform(results['image'])
        return results


@PIPELINES.register_module()
class ToTensor(object):
    """Image ToTensor -->numpy2Tensor
    Args:
        image ：原始图片
    Returns:
        result :  Over ToTensor
    """

    def __call__(self, results):
        transform = transforms.ToTensor()
        results['image'] = transform(results['image'])
        return results


@PIPELINES.register_module()
class Resize(object):
    """Image Resize --> 图片resize操作
    Args:
        size : resize后的尺寸
        image ：原始图片
    Returns:
        result :  Over Resize
    """

    def __init__(self, params):
        self.size = params['size']

    def __call__(self, results):
        transform = transforms.Resize(self.size)
        results['image'] = transform(results['image'])
        return results


@PIPELINES.register_module()
class CenterCrop(object):
    """Image center crop --> 图片中心crop
    Args:
        size : crop后的尺寸
        image ：原始图片
    Returns:
        result :  Over CenterCrop
    """

    def __init__(self, params):
        self.size = params['size']

    def __call__(self, results):
        transform = transforms.CenterCrop(self.size)
        results['image'] = transform(results['image'])
        return results


@PIPELINES.register_module()
class RandomRotation(object):
    """Image random rotation --> 图片随机旋转
    Args:
        degrees : 旋转的角度
        image ：原始图片
    Returns:
        result :  Over RandomRotation
    """

    def __init__(self, params):
        self.degrees = params['degrees']

    def __call__(self, results):
        transform = transforms.RandomRotation(self.degrees)
        results['image'] = transform(results['image'])
        return results
