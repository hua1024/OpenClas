# coding=utf-8  
# @Time   : 2020/11/24 10:11
# @Auto   : zzf-jeff

import torch


def gen_ones(shape):
    if isinstance(shape[0], int):
        return torch.ones(*shape)

    data = []
    for sub_shape in shape:
        data.append(gen_ones(sub_shape))

    return data