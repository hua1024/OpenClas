# coding=utf-8  
# @Time   : 2020/10/27 14:52
# @Auto   : zzf-jeff
import os
import sys

sys.path.append('./')

import torch.nn as nn
from torchclas.models import build_backbone
from torchclas.models import build_loss
from torchclas.models import build_optimizer
from torchclas.models import build_lr_scheduler
from torchclas.datasets import (build_dataloader, build_dataset)

# model settings
cfg_model = dict(
    type='LeNet5',
    in_channel=1,
    num_classes=100,
)
cfg_loss = dict(
    type='CrossEntropyLoss'
)
cfg_optimizer = dict(
    type='SGDDecay',
    params={
        'weight_decay': 0.001,
        'momentum': 0.99
    }

)
cfg_lr = dict(
    type='StepLR',
    params={
        'step_size': 10,
        'gamma': 0.1
    }
)
cfg_data = dict(
    type='ReaderDogCat',
    ann_file='/zzf/data/split_catdog/valid.txt',
    classes=['cat', 'dog'],
    pipeline=[dict(type='ToTensor'), dict(type='Normalize', params={'mean': [0.485, 0.456, 0.406],
                                                                    'std': [0.229, 0.224, 0.225]})]
)

model = build_backbone(cfg_model)
criterion = build_loss(cfg_loss)
optimizer = build_optimizer(cfg_optimizer)(model,0.01)
lr_scheduler = build_lr_scheduler(cfg_lr)(optimizer)
dataset = build_dataset(cfg_data)
dataloader = build_dataloader(dataset, batch_size=1, num_workers=0, shuffle=True)

print(model)
print(criterion)
print(optimizer)
print(lr_scheduler)
print(dataset)
print(dataloader)

for idx, data in enumerate(dataloader):
    print(data['img'].shape)
    print(data['label'])
    break
