# coding=utf-8  
# @Time   : 2020/10/24 11:13
# @Auto   : zzf-jeff

from .pipelines import Compose

from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
import copy
import numpy as np


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Dataset的基类
    Args:
        pipeline : 数据增强模块
        ann_file : 构建自己数据集时使用的文件索引
    """

    def __init__(self, pipeline, ann_file, classes):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.pipeline = Compose(pipeline)
        self.CLASSES = classes
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    def __len__(self):
        return len(self.data_infos)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        gt_labels = np.array([data['label'] for data in self.data_infos])
        return gt_labels

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __getitem__(self, idx):

        return self.prepare_data(idx)
