# coding=utf-8  
# @Time   : 2020/11/9 15:27
# @Auto   : zzf-jeff

import torch.nn as nn
from torchclas.utils.registry import Registry, build_from_cfg

BACKBONES = Registry('backbone')
LOSSES = Registry('loss')
OPTIMIZERS = Registry('optimizer')
LR_SCHEDULERS = Registry('lr_scheduler')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_lr_scheduler(cfg):
    return build(cfg, LR_SCHEDULERS)

def build_optimizer(cfg):
    return build(cfg, OPTIMIZERS)