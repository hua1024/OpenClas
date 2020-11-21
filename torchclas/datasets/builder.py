# coding=utf-8  
# @Time   : 2020/11/10 9:27
# @Auto   : zzf-jeff

import torch.nn as nn
import numpy as np
import random
from functools import partial
from torchclas.utils.registry import Registry, build_from_cfg
from torch.utils.data import DataLoader

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset, batch_size, num_workers, shuffle=True, pin_memory=False, seed=None, **kwargs):
    sampler = None
    collate_fn = None
    init_fn = partial(worker_init_fn, seed=seed) if seed is not None else None
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=shuffle, collate_fn=collate_fn, sampler=sampler, worker_init_fn=init_fn, **kwargs)
    return data_loader


def worker_init_fn(seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
