# -*- coding: utf-8 -*-
# date: 2022/1/23 19:08
# Project: 唐宇迪实战Code
# File Name: main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import config

import torch
from torch import optim,nn
from tensorboardX import SummaryWriter

from dataset import get_mnist_dataloader
from ae import AE
from vae import VAE


def train_AE():
    '''训练AE'''
    train_dataloader, test_dataloader = get_mnist_dataloader()

    model = AE().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    print(model)

    writer = SummaryWriter(comment='AE')
    for epoch in range(config.epochs):
        batch_loss = []
        for batch_idx, (inputs, _) in enumerate(train_dataloader):
            inputs = inputs.to(config.device)   # [batch_size, 1, 28, 28]
            output_hat = model(inputs)
            loss = criterion(output_hat, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('batch_loss: ', loss.item(), loss.item()*config.batch_size)
            # print('-'*10)
            batch_loss.append(loss.item() * config.batch_size)  # 存储每一个batch的总loss

        scheduler.step()
        loss = sum(batch_loss) / len(train_dataloader.dataset)  # 计算一个epoch的loss

        print('ep{}, loss: {}'.format(epoch, loss))

        visualization_img = iter(test_dataloader).next()   # 随机可视化一组测试数据
        with torch.no_grad():
            inputs = visualization_img[0].to(config.device)
            output_hat = model(inputs)

        writer.add_scalar('loss', scalar_value=loss, global_step=epoch)
        writer.add_images('ori_image', img_tensor=inputs, global_step=epoch)
        writer.add_images('pred_image', img_tensor=output_hat, global_step=epoch)
        writer.add_graph(model=model, input_to_model=inputs)

    writer.close()


def train_VAE():
    '''训练VAE'''
    train_dataloader, test_dataloader = get_mnist_dataloader()

    model = VAE().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    print(model)

    model.train()
    writer = SummaryWriter(comment='AE')
    for epoch in range(config.epochs):
        batch_loss = []
        kld_list = []
        for batch_idx, (inputs, _) in enumerate(train_dataloader):
            inputs = inputs.to(config.device)  # [batch_size, 1, 28, 28]
            output_hat, kld = model(inputs)
            loss = criterion(output_hat, inputs)

            if kld is not None:
                elbo = -loss - 1.08*kld
                loss = -elbo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('batch_loss: ', loss.item(), loss.item()*config.batch_size)
            # print('-'*10)
            batch_loss.append(loss.item() * config.batch_size)  # 存储每一个batch的总loss
            kld_list.append(kld.item() * config.batch_size)

        scheduler.step()

        model.eval()
        loss = sum(batch_loss) / len(train_dataloader.dataset)  # 计算一个epoch的loss
        kld = sum(kld_list) / len(train_dataloader.dataset)

        print('ep{}, loss: {}, kld: {}'.format(epoch, loss, kld))

        visualization_img = iter(test_dataloader).next()  # 随机可视化一组测试数据
        with torch.no_grad():
            inputs = visualization_img[0].to(config.device)
            output_hat, _ = model(inputs)

        writer.add_scalar('loss', scalar_value=loss, global_step=epoch)
        writer.add_images('ori_image', img_tensor=inputs, global_step=epoch)
        writer.add_images('pred_image', img_tensor=output_hat, global_step=epoch)
        writer.add_graph(model=model, input_to_model=inputs)

    writer.close()

if __name__ == '__main__':
    train_VAE()
