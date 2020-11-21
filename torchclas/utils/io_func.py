# coding=utf-8  
# @Time   : 2020/10/28 9:34
# @Auto   : zzf-jeff

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import shutil
import torch
import yaml
import time

from pathlib import Path
from tensorboardX import SummaryWriter


## 常用Path 代替os.path ，Path(面向对象的文件系统路劲)

def config_load(cfg_path):
    yaml_file = open(cfg_path, 'r', encoding='utf-8')
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config


def create_log_folder(cfg):
    root_output_dir = Path(cfg['BASE']['checkpoints'])
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    dataset = cfg['BASE']['dataset_name']
    model = cfg['BASE']['algorithm']
    backones = cfg['BACKBONES']['type']
    time_str = time.strftime('%Y%m%d')
    checkpoints_output_dir = root_output_dir / dataset / model / backones / 'weights'
    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = root_output_dir / dataset / model / backones / time_str / 'tb_log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    txt_log_dir = root_output_dir / dataset / model / backones / 'txt_log'
    print('=> creating {}'.format(txt_log_dir))
    txt_log_dir.mkdir(parents=True, exist_ok=True)
    return {'chs_dir': str(checkpoints_output_dir),
            'tb_dir': str(tensorboard_log_dir),
            'txt_dir': str(txt_log_dir)}


def create_dir(path):
    if not (os.path.exists(path)):
        os.mkdir(path)


def create_tb(path):
    writer = SummaryWriter(log_dir=path)
    return writer
