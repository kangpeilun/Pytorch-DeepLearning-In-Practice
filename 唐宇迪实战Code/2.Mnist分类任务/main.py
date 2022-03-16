# -*- coding: utf-8 -*-
# date: 2022/1/14
# Project: Pytorch深度学习实战
# File Name: mnist_cnn_main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import pickle
import gzip
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 设置模型和数据在哪个设备上进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载mnist数据集
with gzip.open('./mnist/mnist.pkl.gz', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
plt.show()
print(x_train.shape, x_train[:10], type(x_train))

# 预处理数据集，将numpy格式数据转换为tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)   # 使用map方法，可以将数据全部映射为对应的数据类型
)


# 定义模型类
class mnist_nn(nn.Module):
    def __init__(self):
        super(mnist_nn, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # 激活函数不会改变形状
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)

        return x


# 构建dataloader
# TensorDataset可以理解为Dataset的简化版本，相当于zip，将data和label进行打包
train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
def get_data(train_dataset, valid_dataset):
    return (
        DataLoader(train_dataset, batch_size=64, shuffle=True),  # 训练数据集，打乱数据
        DataLoader(valid_dataset, batch_size=64*2)  # 验证集无需打乱
    )


# 定义计算每一个batch的损失函数
def loss_batch(model, loss_func, xb, yb, opt=None):
    '''
    :param model: 实例化后的模型
    :param loss_func: 实例化后的损失函数
    :param xb: 每个batch的训练数据
    :param yb: 每个batch的label
    :param opt: 优化器
    :return: loss值，当前batch的训练数据的个数
    '''
    loss = loss_func(model(xb), yb)

    # 若model.train() 计算梯度并更新
    if opt is not None:
        loss.backward()  # 反向传播
        opt.step()  # 更新参数
        opt.zero_grad() # 梯度清零

    return loss.item(), len(xb)


# 定义训练函数
def fit(epochs, model, optimizer, train_dataloader, valid_dataloader):
    '''
    :param epochs: 训练轮数
    :param model: 实例化后的模型
    :param optimizer: 优化器
    :param train_dataloader: 训练集dataloader
    :param valid_dataloader: 验证集dataloader
    :return:
    '''
    loss_func = F.cross_entropy
    # 一般在训练模型时加上model.train()，这样会正常使用Batch Normalization和 Dropout
    # 测试的时候一般选择model.eval()，这样就不会使用Batch Normalization和 Dropout
    for epoch in range(epochs):
        model.train()
        losses, nums = zip(*[loss_batch(model, loss_func, xb, yb, optimizer) for xb, yb in train_dataloader])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        model.eval()
        # 验证时，不需要进行梯度的传播
        with torch.no_grad():
            # zip(*) 将 [(loss1, num1), (loss2, num2), ...]这样的数据拆包为 [loss1, loss2, ...] [num1, num2, ...] 的格式
            # PS: loss_batch 计算出来的 loss 是每一个batch的平均loss，估计算全局loss时需要将loss与num相乘
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dataloader])

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f'epoch:{epoch}\ttrain_loss:{train_loss}\tvalid_loss:{val_loss}')


# 获取构建好的模型
def get_model():
    model = mnist_nn()
    return model, torch.optim.Adam(model.parameters(), lr=1e-4)


# 进行训练
train_dataloader, valid_dataloader = get_data(train_dataset, valid_dataset)
model, optimizer = get_model()
fit(25, model, optimizer, train_dataloader, valid_dataloader)