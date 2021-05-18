# coding=utf-8  
# @Time   : 2021/3/6 14:35
# @Auto   : zzf-jeff

class ConfigSetting():
    config = '../config/densenet_121.yaml'
    weight = '../checkpoints/DogCat/Clas/DenseNet/weights/bs_32_best.pth'
    engine = '../d50.engine'
    mode = 'trt'
    queue = 'img'
    client_time = 0.1
    batch_size = 8

opt = ConfigSetting()
