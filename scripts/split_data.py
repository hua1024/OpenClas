# coding=utf-8  
# @Time   : 2020/11/20 9:57
# @Auto   : zzf-jeff

import os
import shutil
import random
import sys
import argparse
from tqdm import tqdm

# sys.path.append("..")

'''
针对kaggle的猫狗分类写的数据随机分割
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=r'D:\data\图像分类\cat_dog\train',
                        help='The split image origin floder')
    parser.add_argument('--output', type=str, default='dog_cat', help='The split image output floder')
    args = parser.parse_args()
    origin_img_path = args.source
    result_img_path = args.output

    if not os.path.exists(result_img_path):
        os.makedirs(result_img_path)

    train_path = os.path.join(result_img_path, 'train')
    valid_path = os.path.join(result_img_path, 'valid')

    train_scale = 0.9
    img_formats = ['.png', '.jpg', '.bmp', '.JPG']

    all_label_img = os.listdir(origin_img_path)
    for label_name in os.listdir(origin_img_path):

        label_name = label_name.split('.')[0]
        if not os.path.exists(os.path.join(train_path, label_name)):
            os.makedirs(os.path.join(train_path, label_name))
        if not os.path.exists(os.path.join(valid_path, label_name)):
            os.makedirs(os.path.join(valid_path, label_name))

    images = [os.path.join(origin_img_path, x) for x in all_label_img if
              os.path.splitext(x)[-1].lower() in img_formats]
    num_img = len(images)
    random.shuffle(images)

    train_num = int(num_img * train_scale)

    train_img_list = images[0:train_num]

    val_img_list = images[train_num:]

    for img in tqdm(train_img_list):
        with open(os.path.join(result_img_path, 'train.txt'), 'a+') as fw:
            base_name = os.path.basename(img)
            label_name = base_name.split('.')[0]
            fw.write('{},{}\n'.format(os.path.join(os.path.join(train_path, label_name, base_name)), label_name))
            shutil.copy(img, os.path.join(os.path.join(train_path, label_name, base_name)))

    for img in tqdm(val_img_list):
        with open(os.path.join(result_img_path, 'valid.txt'), 'a+') as fw:
            base_name = os.path.basename(img)
            label_name = base_name.split('.')[0]
            fw.write('{},{}\n'.format(os.path.join(os.path.join(valid_path, label_name, base_name)), label_name))
            shutil.copy(img, os.path.join(os.path.join(valid_path, label_name, base_name)))
