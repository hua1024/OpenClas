# coding=utf-8  
# @Time   : 2020/10/27 14:52
# @Auto   : zzf-jeff
import os
import sys

sys.path.append('./')


# import torch.nn as nn
from torchclas.models import build_backbone
# from torchclas.models import build_loss
# from torchclas.models import build_optimizer
# from torchclas.models import build_lr_scheduler
# from torchclas.datasets import (build_dataloader, build_dataset)
#
# # model settings
# cfg_model = dict(
#     type='LeNet5',
#     in_channel=1,
#     num_classes=100,
# )
# cfg_loss = dict(
#     type='CrossEntropyLoss'
# )
# cfg_optimizer = dict(
#     type='SGDDecay',
#     params={
#         'weight_decay': 0.001,
#         'momentum': 0.99
#     }
#
# )
# cfg_lr = dict(
#     type='StepLR',
#     params={
#         'step_size': 10,
#         'gamma': 0.1
#     }
# )
# cfg_data = dict(
#     type='ReaderDogCat',
#     ann_file='/zzf/data/split_catdog/valid.txt',
#     classes=['cat', 'dog'],
#     pipeline=[dict(type='ToTensor'), dict(type='Normalize', params={'mean': [0.485, 0.456, 0.406],
#                                                                     'std': [0.229, 0.224, 0.225]})]
# )
#
import torch
from torchclas.utils.io_func import (config_load, create_log_folder, create_tb)
from torchsummary import summary

device = torch.device('cuda:0')

cfg = config_load('config/csp_densenet_121.yaml')
model = build_backbone(cfg['BACKBONES']).to(device)
# print(model)

input = torch.randn(1,3,224,224).to(device)
# summary(model, input_size=(3, 224, 224))

out = model(input)

# criterion = build_loss(cfg_loss)
# optimizer = build_optimizer(cfg_optimizer)(model,0.01)
# lr_scheduler = build_lr_scheduler(cfg_lr)(optimizer)
# dataset = build_dataset(cfg_data)
# dataloader = build_dataloader(dataset, batch_size=1, num_workers=0, shuffle=True)
#
# print(model)
# print(criterion)

# print(optimizer)
# print(lr_scheduler)
# print(dataset)
# print(dataloader)
#
# for idx, data in enumerate(dataloader):
#     print(data['img'].shape)
#     print(data['label'])
#     break

# class Person(object):
#     def __init__(self):
#         self.__age = 10
#
#     def get_age(self):
#         return self.__age
#
#     def set_age(self, x):
#         self.__age = x
#
#
# t = Person()
# # print(t.__age)
# print(t.set_age(22))
# print(t.get_age())
# class Person(object):
#     def __init__(self, x):
#         self.__age = x
#
#     @property
#     def age(self):
#         return self.__age
#
#     @age.setter
#     def age(self, new_age):
#         self.__age = new_age
#
#
# t = Person(22)
#
# print(t.age)
# t.age = 1
# print(t.age)
# def singleton(cls):
#     instances = {}
#
#     def wrapper(*args, **kwargs):
#         if cls not in instances:
#             instances[cls] = cls(*args, **kwargs)
#         return instances[cls]
#     return wrapper
#
# #
# @singleton
# class Foo(object):
#     pass
#
# class Singleton(object):
#     def __new__(cls, *args, **kwargs):
#         if not hasattr(cls, '_instance'):
#             cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
#         return cls._instance
#
# class Foo(Singleton):
#     pass
#
#
#
# foo1 = Foo()
# foo2 = Foo()
# print(id(foo1), id(foo2))
# print(foo1 is foo2)  # True

# class Solution(object):
#     def reverse(self, x):
#         if -10 < x < 10:
#             return x
#         str_x = str(x)
#         if str_x[0] != '-':
#             str_x = str_x[::-1]
#             x = int(str_x)
#         else:
#             str_x=str_x[1:][::-1]
#             x = int(str_x)
#             x = -x
#         return x if -2147483648<x<2147483647 else 0
#
# s = Solution()
# print(s.reverse(-132))
#
