# -*- coding: utf-8 -*-
# date: 2022/1/23 19:08
# Project: 唐宇迪实战Code
# File Name: vae.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import numpy as np

import config

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # [batch_size, 784] -> [batch_size, 20]
        # mu: [batch_size, 10]
        # sigma: [batch_size, 10]
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
            nn.Linear(10, 64),
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
        h_ = self.encoder(x)     # [batch_size, 20], 包含mean和sigma
        # mu: [batch_size, 10]
        # sigma: [batch_size, 10]
        mu, sigma = h_.chunk(2, dim=1)   # 沿着第2个维度等分成两个块
        # reparametrize trick, epison~N(0,1) epison的值从一个正态分布中取
        h = mu + sigma * torch.randn_like(sigma)  # [batch_size, 20]

        # decoder
        x = self.decoder(h)     # [batch_size, 784]
        # reshape
        output = x.view(batch_size, 1, 28, 28)

        # KL divergence(分布) on two Gaussians
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) + torch.pow(sigma, 2) - torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / batch_size

        return output, kld