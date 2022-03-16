# -*- coding: utf-8 -*-
# date: 2022/1/14
# Project: Pytorch深度学习实战
# File Name: config.py
# Description: 配置文件，超参数
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import os
import torch

# 配置训练和测试所用的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device != 'cpu':
    print('CUDA is available! Training on GPU!')
else:
    print('CUDA is not available! Training on CPU!')



# 训练数据，测试数据存放位置
data_dir = './data/flower_data'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

# 真正类别名文件路径
real_class_name = './data/flower_data/cat_to_name.json'

# 选择使用pytorch内置的模型
model_name = 'resnet'   # ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']

# 模型超参数
feature_extract = True  # 是否要冻结网络层数，True冻结部分层，False全不冻结
freeze_layer = 400  # 要冻结的网络层数
use_pretrained = True  # 是否使用预训练权重
is_inception = False   # 是否使用 inception网络，这是一种网络的结构，详见 https://www.cnblogs.com/dengshunge/p/10808191.html

num_classes = 102   # 分类类别数
epochs = 25
batch_size = 32
learn_rate = 1e-4   # 学习率
scheduler_step_size = 7  # 设置学习率每多少个epoch后进行衰减，默认每7个epoch后进行衰减
scheduler_gamma = 0.1   # 设置学习率衰减为原来的多少倍，默认衰减为原来的1/10

# 存放模型的主文件夹, 模型保存
model_dir = './model_data'
checkpoint = os.path.join(model_dir, 'checkpoint.pth')  # 设置checkpoint模型保存，进行断点续练，不使用改为None
