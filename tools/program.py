# coding=utf-8  
# @Time   : 2020/11/20 10:54
# @Auto   : zzf-jeff


from torchclas.utils.misc import (AverageMeter, accuracy)
import torch
import random
import numpy as np


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def trainer(train_loader, model, optimizer, criterion, device):
    model.train()

    t_loss = AverageMeter()
    t_acc = AverageMeter()

    for batch_index, data in enumerate(train_loader):
        images = data['image']
        labels = data['label']
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        pred_1 = accuracy(outputs, labels, topk=(1,))[0]
        t_loss.update(loss.item(), images.size(0))
        t_acc.update(pred_1.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return t_loss.avg, t_acc.avg


def evaler(valid_loader, model, criterion, device, topk=(1,)):
    model.eval()
    v_loss = AverageMeter()
    v_acc = AverageMeter()
    with torch.no_grad():
        for batch_index, data in enumerate(valid_loader):
            images = data['image']
            labels = data['label']
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            pred_1 = accuracy(outputs, labels, topk=topk)[0]
            v_loss.update(loss.item(), images.size(0))
            v_acc.update(pred_1.item(), images.size(0))
        return v_loss.avg, v_acc.avg
