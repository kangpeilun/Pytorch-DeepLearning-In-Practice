# -*- coding: utf-8 -*-
# date: 2022/1/23 19:08
# Project: 唐宇迪实战Code
# File Name: ae.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import config

import torch
from torch import nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # [batch_size, 784] -> [batch_size, 20]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU(),
        )

        # [batch_size, 20] -> [batch_size, 784]
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()    # 将像素值压缩到0-1之间
        )

    def forward(self, x):
        '''
        :param x: [batch_size, 1, 28, 28]
        :return:
        '''
        batch_size = x.size(0)
        # flatten
        x = x.view(batch_size, -1)  # [batch_size, 784]
        # encoder
        x = self.encoder(x)     # [batch_size, 20]
        # decoder
        x = self.decoder(x)     # [batch_size, 784]
        # reshape
        output = x.view(batch_size, 1, 28, 28)

        return output