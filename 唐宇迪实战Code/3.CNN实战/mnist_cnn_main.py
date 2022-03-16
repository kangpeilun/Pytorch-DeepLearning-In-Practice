# -*- coding: utf-8 -*-
# date: 2022/1/14
# Project: Pytorch深度学习实战
# File Name: mnist_cnn_main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np

# 定义超参数
input_size = 28  # 图像尺寸 28*28
num_classes = 10  # 类别数
epochs = 3  # 训练轮数
batch_size = 64

# 训练集
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
# 测试集
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())
# 构建batch dataloader数据
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size*2,
                             shuffle=False)


# 构建CNN模型结构
class mnist_cnn(nn.Module):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        self.conv1 = nn.Sequential(         # 输入大小 (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # 通道数为1，灰度图
                out_channels=16,            # 要得到几多少个特征图
                kernel_size=5,              # 卷积核大小
                stride=1,                   # 步长
                padding=2,                  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
                # 卷积后特征图大小计算通用公式：
                #   长度：H2 = (H1 - Fh + 2P)/S + 1    宽度：W2 = (W1 - Fw + 2P)/S + 1
                #   其中W1、H1表示输入的宽度、长度；W2、H2表示输出特征图的宽度、长度；F表示卷积核长和宽的大小；S表示滑动窗口的步长;P表示边界填充(加几圈0)。
            ),
            nn.ReLU(),                      # relu激活函数
            nn.MaxPool2d(kernel_size=2),    # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)，池化后h,w变为原来的一半
        )
        self.conv2 = nn.Sequential(         # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # 输出 (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2)                 # 输出 (32, 7, 7)
        )
        self.out = nn.Linear(32*7*7, 10)    # 全连接层得到的结果

    def forward(self, x):
        # input: [1, 28, 28]
        x = self.conv1(x)  # [16, 14, 14]
        x = self.conv2(x)  # [32, 7, 7]
        x = x.view(x.size(0), -1)
        output = self.out(x)  # [32*7*7, 10]
        return output


# 评估指标——准确率
def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]  # 返回10个类别预测的概率中概率最大位置的索引
    # torch.max() 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
    # 因为手写数字的0-9正好和索引对应上，故要取索引
    rights = pred.eq(labels.data.view_as(pred)).sum()  # view_as 将labels形状变化为跟pred一样
    # pred.eq() 预测值pred中的元素和labels中对应元素 相等 则为True，最后通过sum()进行求和
    # 返回每个batch，总的预测正确的个数，batch的样本个数
    return rights, len(labels)


# 训练网络
mnist_model = mnist_cnn()  # 实例化
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.Adam(mnist_model.parameters(), lr=1e-4)  # 定义优化器

def fit(epochs):
    # 把当前epoch的结果保存下来
    for epoch in range(epochs):
        train_rights = []  # 记录当前epoch中每个batch上的准确率

        for batch_index, (data, label) in enumerate(train_dataloader):  # 针对容器中的每一批进行循环
            mnist_model.train()  # 告诉模型，从这行开始表示训练过程，需要进行梯度更新
            optimizer.zero_grad()
            output = mnist_model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            rights = accuracy(output, label)  # accuracy()函数的返回值有两个，如果用一个变量接收，会自动打包为 元组
            train_rights.append(rights)

            if batch_index % 100 == 0:
                mnist_model.eval()  # 告诉模型，从这行开始表示测试过程，不需要进行梯度更新
                test_rights = []  # 记录当前epoch下，在测试集上每个batch的准确率

                for (data, label) in test_dataloader:
                    output = mnist_model(data)
                    right = accuracy(output, label)
                    test_rights.append(right)

                # 准确率计算
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))  # 记录当前epoch中，训练集每个batch上正确个数并求和，计算所有的样本数
                test_r = (sum([tup[0] for tup in test_rights]), sum([tup[1] for tup in test_rights]))  # 记录当前epoch中，每100个batch进行一次测试，测试集每个batch上正确个数并求和，计算所有的样本数

                print('epoch:{} [{}/{} ({:.0f}%)]\t loss:{:.6f}\t train_acc:{:.2f}%\t test_acc:{:.2f}%'.format(
                    epoch, batch_index*batch_size, len(train_dataloader.dataset),  # len(train_dataloader.dataset) 整个训练集的数据量
                    100. * batch_index / len(train_dataloader),  # len(train_dataloader)每个batch中的数据量
                    loss.data,
                    100. * train_r[0].numpy() / train_r[1],  # 当前epoch中，训练集平均准确率
                    100. * test_r[0].numpy() / test_r[1]     # 当前epoch中，每100个batch进行一次测试，测试集平均准确率
                ))


if __name__ == '__main__':
    fit(3)