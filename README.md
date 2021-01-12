# OpenClas
## 简介
通过自己搭建图像分类的框架，总结学习到的图像分类算法、pytorch的操作、python操作、部署操作

性能上可能存在欠缺，更多是为了学习

## 更新日志

- 框架训练测试推理测试通过 2020.11.30
- TensorRT加速测试通过 2020.12.10
- 添加主干网络(ShuffleNetV1,ShuffleNetV2,InceptionV3,InceptionV4,Inception-ResNet-V2) 2020.12.26

## 目前已支持
- [x] ResNet、ResNet_vd、ResNext、SeResnet
- [x] DenseNet
- [x] VGG
- [x] MobileNetV1、MobileNetV2、MobileNetV3
- [x] InceptionV3,InceptionV4,Inception-ResNet-V2
- [x] ShuffleNetV1,ShuffleNetV2

## Usage
***
#### 1.requirements
```shell script
pip3 install requirements.txt
```
#### 2.dataset
#### 3.modify yaml config
#### 4.train the model
#### 5.test the model
#### 6.infer image
***

## cifar100效果

## 模型剪枝效果

## 模型蒸馏效果

## 模型TensorRT加速效果

## todo list
- [ ] cifar100训练结果
- [ ] 分布式训练支持
- [ ] 可视化特征输出
- [x] TensorRT加速
- [ ] 模型剪枝
- [ ] 模型量化
- [ ] 模型蒸馏
- [ ] 移动端移植








### reference
    1.https://github.com/open-mmlab/mmclassification
    2.https://github.com/weiaicunzai/pytorch-cifar100
    3.https://github.com/Media-Smart/volksdep.git
    4.https://github.com/pytorch/vision.git
> If this repository helps you，please star it. Thanks.
