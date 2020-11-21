# coding=utf-8  
# @Time   : 2020/11/10 11:10
# @Auto   : zzf-jeff

from .base_dataset import BaseDataset
from .builder import DATASETS
from PIL import Image
import numpy as np
from tqdm import tqdm


@DATASETS.register_module()
class ReaderDogCat(BaseDataset):
    def __init__(self, pipeline, ann_file, classes):
        super(ReaderDogCat, self).__init__(pipeline=pipeline, ann_file=ann_file, classes=classes)

    def read_txt(self, txt_path):
        '''
        读取txt文件的标注信息，格式为
        xxx/a/1.png,a
        xxx/a/2.png,a
        Args:
            txt_path: train/valid/test data txt
        Returns:
            imgs：list, all data info
        '''
        with open(txt_path, 'r') as f:
            imgs = list(map(lambda line: line.strip().split(','), f))
        return imgs

    def load_annotations(self):
        imgs = self.read_txt(self.ann_file)
        data_infos = []
        for (filename, label) in tqdm(imgs):
            img = Image.open(filename).convert('RGB')
            info = {'img': img, 'label': self.class_to_idx[label]}
            data_infos.append(info)
        return data_infos
