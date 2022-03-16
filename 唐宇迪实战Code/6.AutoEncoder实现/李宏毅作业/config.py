# -*- coding: utf-8 -*-
# date: 2022/1/26 21:53
# Project: 唐宇迪实战Code
# File Name: config.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 训练超参数
epochs = 50
batch_size = 32
learning_rate = 1e-3

# 模型选择
model_type = 'cnn'      # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}