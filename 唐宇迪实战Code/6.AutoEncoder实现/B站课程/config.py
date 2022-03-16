# -*- coding: utf-8 -*-
# date: 2022/1/23 19:12
# Project: 唐宇迪实战Code
# File Name: config.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 训练参数
epochs = 100
batch_size = 32
learn_rate = 1e-3
scheduler_step_size = 8
scheduler_gamma = 0.1