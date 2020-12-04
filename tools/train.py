# coding=utf-8  
# @Time   : 2020/10/24 11:12
# @Auto   : zzf-jeff
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('./')

import argparse
import copy
import os
import time
import torch
from torchsummary import summary

from torchclas.models import (build_backbone, build_loss, build_optimizer, build_lr_scheduler)
from torchclas.datasets import (build_dataloader, build_dataset)
from torchclas.utils.io_func import (config_load, create_log_folder, create_tb)
from torchclas.utils.torch_utils import (select_device, save_checkpoints)
from tools.program import (trainer, evaler, set_random_seed)
from torchclas.utils.logger import (get_logger, print_log)


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--config', help='train config file path', default='config/resnet50.yaml')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = config_load(args.config)
    set_random_seed(cfg['BASE']['seed'])
    output_dict = create_log_folder(cfg)
    tb_writer = None
    if cfg['BASE']['use_tensorboard']:
        tb_writer = create_tb(output_dict['tb_dir'])

    logger = get_logger(cfg['BACKBONES']['type'], os.path.join(output_dict['txt_dir'], 'train.log'))

    logger.info(cfg)

    device = select_device(cfg['BASE']['gpu_id'], cfg['TRAIN']['batch_size'], logger)

    model = build_backbone(cfg['BACKBONES'])


    trainer_dataset = build_dataset(cfg['TRAIN']['dataset'])
    valid_dataset = build_dataset(cfg['VALID']['dataset'])

    train_loader = build_dataloader(trainer_dataset, batch_size=cfg['TRAIN']['batch_size'],
                                    num_workers=cfg['TRAIN']['num_workers'], shuffle=cfg['TRAIN']['shuffle'])

    valid_loader = build_dataloader(valid_dataset, batch_size=cfg['VALID']['batch_size'],
                                    num_workers=cfg['VALID']['num_workers'], shuffle=cfg['VALID']['shuffle'])

    criterion = build_loss(cfg['LOSS'])
    optimizer = build_optimizer(cfg['OPTIMIZER'])(model, cfg['BASE']['init_lr'])
    lr_scheduler = build_lr_scheduler(cfg['LR_SCHEDULER'])(optimizer)

    criterion = criterion.to(device)
    model = model.to(device)

    # 打印网络结构
    summary(model,input_size=(3,224,224))
    print(model)

    best_acc = 0.0
    start_epoch = 0

    # 断点恢复训练加载权重
    if cfg['BASE']['resume']:
        print("=> loading checkpoint '{}'".format(cfg['BASE']['resume_file']))
        assert os.path.isfile(cfg['BASE']['resume_file']), 'Error : no checkpoint directory found.'
        checkpoint = torch.load(cfg['BASE']['resume_file'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, cfg['BASE']['n_epoch']):

        t_loss, t_acc = trainer(train_loader, model, optimizer, criterion, device)

        # Update learning rate
        lr_scheduler.step()
        logger.info('*' * 20)
        logger.info('epoch : {} |train loss : {} |train acc : {} |lr : {}'.format(epoch, t_loss, t_acc,
                                                                                  optimizer.state_dict()[
                                                                                      'param_groups'][0]['lr']))
        if tb_writer is not None:
            tb_writer.add_scalar('Train/loss', t_loss, epoch)
            tb_writer.add_scalar('Train/acc', t_acc, epoch)
        # Do valid and saving best weight
        if (epoch >= cfg['BASE']['start_val']):
            v_loss, v_acc = evaler(valid_loader, model, criterion, device)
            logger.info('epoch : {} |valid loss : {} |valid acc : {}'.format(epoch, v_loss, v_acc))
            if tb_writer is not None:
                tb_writer.add_scalar('Valid/loss', v_loss, epoch)
                tb_writer.add_scalar('Valid/acc', v_acc, epoch)

            if (float(v_acc) > best_acc):
                best_acc = float(v_acc)
                save_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                }
                filename = 'bs_{}_best.pth'.format(str(cfg['TRAIN']['batch_size']))
                save_checkpoints(save_state, output_dict['chs_dir'], filename)

        # Every number epoch to save weight
        if (epoch + 1) % cfg['BASE']['save_epoch'] == 0:
            save_state = {
                'epoch': (epoch + 1),
                'state_dict': model.state_dict(),
                'lr': optimizer.param_groups[0]['lr'],
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            filename = 'bs_{}_eps_{}.pth'.format(str(cfg['TRAIN']['batch_size']), str(epoch))
            save_checkpoints(save_state, output_dict['chs_dir'], filename)

    # out of swap
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    tb_writer.close() if tb_writer is not None else None


if __name__ == '__main__':
    main()
