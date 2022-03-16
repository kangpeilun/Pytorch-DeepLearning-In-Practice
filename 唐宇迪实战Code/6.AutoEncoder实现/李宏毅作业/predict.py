# -*- coding: utf-8 -*-
# date: 2022/1/26 22:57
# Project: 唐宇迪实战Code
# File Name: predict.py
# Description: 模型已加载并生成其异常分数预测
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import config

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from dataset import CustomTensorDataset, get_dataloader

import pandas as pd


eval_batch_size = 200


def test():
    # build testing dataloader
    _, test_dataloader = get_dataloader()
    eval_loss = nn.MSELoss(reduction='none')

    # load trained model
    checkpoint_path = 'last_model_cnn.pt'
    model = torch.load(checkpoint_path).to(config.device)
    model.eval()

    # prediction file
    out_file = 'PREDICTION_FILE.csv'

    anomality = list()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            if config.model_type in ['cnn', 'vae', 'resnet']:
                img = data.float().cuda()
            elif config.model_type in ['fcn']:
                img = data.float().cuda()
                img = img.view(img.shape[0], -1)
            else:
                img = data[0].cuda()
            output = model(img)
            if config.model_type in ['cnn', 'resnet', 'fcn']:
                output = output
            elif config.model_type in ['res_vae']:
                output = output[0]
            elif config.model_type in ['vae']:  # , 'vqvae'
                output = output[0]
            if config.model_type in ['fcn']:
                loss = eval_loss(output, img).sum(-1)
            else:
                loss = eval_loss(output, img).sum([1, 2, 3])
            anomality.append(loss)
    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['Predicted'])
    df.to_csv(out_file, index_label='Id')

