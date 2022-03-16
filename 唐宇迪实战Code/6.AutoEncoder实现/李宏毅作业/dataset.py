# -*- coding: utf-8 -*-
# date: 2022/1/26 21:53
# Project: 唐宇迪实战Code
# File Name: dataset.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms

import config
import numpy as np
import random

train = np.load('data/trainingset.npy', allow_pickle=True)
test = np.load('data/testingset.npy', allow_pickle=True)

def same_seeds(seed):
    '''
    # TODO: 学习这种写法
    将随机种子设置为某个值 以实现可重复性
    :param seed:
    :return:
    '''
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)     # 为CPU设置种子用于生成随机数，以使得结果是确定的
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子，以使得结果是确定的
        torch.cuda.manual_seed_all(seed)    # 为所有的 GPU 设置种子用于生成随机数，以使得结果是确定的
    '''
    torch.backends.cudnn.benchmark = True 
    将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    一般来讲，应该遵循以下准则:
        如果网络的输入数据维度或类型上变化不大，网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小尺寸，输入的通道）是不变的，
        设置 torch.backends.cudnn.benchmark = true 可以增加运行效率
    反之:
        如果网络的输入数据在每次 iteration 都变化的话，（例如，卷积层的设置一直变化、某些层仅在满足某些条件时才被“激活”，或者循环中的层可以重复不同的次数），
        会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会耗费更多的时间，降低运行效率
    '''
    torch.backends.cudnn.benchmark = False
    '''
    benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
    如果想要避免这种结果波动，设置：torch.backends.cudnn.deterministic = True保证实验的可重复性
    '''
    torch.backends.cudnn.deterministic = True



class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    用于获取和处理数据的模块。这里的变换函数将图像的像素从 [0, 255] 归一化为 [-1.0, 1.0]
    """

    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)   # pytorch默认的通道排布为[c, w, h]

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),  # 将数据类型由torch.uint8 -> torch.float
            transforms.Lambda(lambda x: 2. * x / 255. - 1.),  # 这里的变换函数将图像的像素从 [0, 255] 归一化为 [-1.0, 1.0]
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)  # 用上面定义的转换

        return x

    def __len__(self):
        return len(self.tensors)


def get_dataloader():
    x_train = torch.from_numpy(train)
    x_test = torch.from_numpy(test)

    train_dataset = CustomTensorDataset(x_train)
    test_dataset = CustomTensorDataset(x_test)

    # TODO: 了解RandomSampler https://www.cnblogs.com/marsggbo/p/11541054.html
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    get_dataloader()