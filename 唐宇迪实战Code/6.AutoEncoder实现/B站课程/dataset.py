# -*- coding: utf-8 -*-
# date: 2022/1/23 19:10
# Project: 唐宇迪实战Code
# File Name: dataset.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import config

from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_mnist_dataloader():
    '''获取mnist数据集'''
    # 获取训练集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader = get_mnist_dataloader()
    data = iter(train_dataloader).next()
    print(data[0].size())