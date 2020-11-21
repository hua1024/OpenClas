# coding=utf-8  
# @Time   : 2020/11/10 11:10
# @Auto   : zzf-jeff


'''
基于torchvision的transforms或者自己实现的
'''

import torchvision.transforms as transforms
import inspect
import math
import random
from ..builder import PIPELINES


@PIPELINES.register_module()
class Normalize(object):

    def __init__(self, params):
        self.mean = params['mean']
        self.std = params['std']

    def __call__(self, results):
        transform = transforms.Normalize(mean=self.mean, std=self.std)
        results['img'] = transform(results['img'])
        return results


@PIPELINES.register_module()
class ToTensor(object):

    def __call__(self, results):
        transform = transforms.ToTensor()
        results['img'] = transform(results['img'])
        return results


@PIPELINES.register_module()
class Resize(object):
    def __init__(self, params):
        self.size = params['size']

    def __call__(self, results):
        transform = transforms.Resize(self.size)
        results['img'] = transform(results['img'])
        return results
