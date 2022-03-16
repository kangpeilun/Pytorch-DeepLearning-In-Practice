# -*- coding: utf-8 -*-
# date: 2022/1/26 21:53
# Project: 唐宇迪实战Code
# File Name: main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import config

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
from torch import optim

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans

from qqdm import qqdm, format_str
import pandas as pd
import numpy as np

import pdb  # use pdb.set_trace() to set breakpoints for debugging

from dataset import get_dataloader
from model.autoencoder import Resnet, fcn_autoencoder, conv_autoencoder, VAE, loss_vae


# Model
model_classes = {'resnet': Resnet(), 'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE(), }
model = model_classes[config.model_type].to(config.device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def train():
    best_loss = np.inf  # 使用这种方式设置无穷大，进行初始化
    model.train()

    qqdm_train = qqdm(range(config.epochs), desc=format_str('bold', 'Description'))
    train_dataloader, _ = get_dataloader()
    for epoch in qqdm_train:
        tot_loss = list()
        for data in train_dataloader:

            # ===================loading=====================
            if config.model_type in ['cnn', 'vae', 'resnet']:
                img = data.float().cuda()
            elif config.model_type in ['fcn']:
                img = data.float().cuda()
                img = img.view(img.shape[0], -1)

            # ===================forward=====================
            output = model(img)
            if config.model_type in ['vae']:
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)

            tot_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================save_best====================
        mean_loss = np.mean(tot_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, 'best_model_{}.pt'.format(config.model_type))
        # ===================log========================
        qqdm_train.set_infos({
            'epoch': f'{epoch + 1:.0f}/{config.epochs:.0f}',
            'loss': f'{mean_loss:.4f}',
        })
        # ===================save_last========================
        torch.save(model, 'last_model_{}.pt'.format(config.model_type))


if __name__ == '__main__':
    train()