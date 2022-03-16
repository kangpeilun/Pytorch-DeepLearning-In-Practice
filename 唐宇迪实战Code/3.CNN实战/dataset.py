# -*- coding: utf-8 -*-
# date: 2022/1/14
# Project: Pytorch深度学习实战
# File Name: dataset.py
# Description: 专门用于生成和处理数据
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import os

import numpy as np

import config
import json
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader,Dataset

# 定义用于训练和测试的数据增强方法
'''
    一定要保证训练集和测试集 的数据处理方法是一致的。测试集可以不包含训练集中的 数据增强部分
'''
data_transformer = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),  # 随机旋转，-45到45度之间
        transforms.CenterCrop(224),     # 从中心开始裁剪, 保留一个224x224大小的图像，相当于直接原图像进行裁剪了
        transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率随机 水平 翻转
        transforms.RandomVerticalFlip(p=0.5),   # 以50%的概率随机 垂直 翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),     # 以25%概率对图像进行灰度化，转换后的图片还是 3 通道的
        transforms.ToTensor(),  # 必须把数据转换为tensor格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 三个通道对应的均值，标准差，这些值是由resnet在对应数据集上训练时所计算得到的均值和标准差，用他们已经计算好的值效果会好些
    ]),
    'valid': transforms.Compose([
        transforms.Resize((256, 256)), # 切记 Resize操作中的参数 必须是两个数(num1, num2)
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 必须把数据转换为tensor格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化，三个通道对应的均值，标准差
    ])
}


def get_data():
    '''
    获取训练和测试要用的数据集，以字典形式给出
    使用字典存储 训练集 和 测试集，不同的数据类型 对应不同的数据预处理操作
    :return: image_dataset, dataloader, dataset_size, class_names
    '''
    image_datasets = {x: datasets.ImageFolder(os.path.join(config.data_dir, x), data_transformer[x]) for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=config.batch_size, shuffle=True) for x in ['train', 'valid']}
    dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid']}  # 训练集和测试集 分别有多少样本
    class_name = image_datasets['train'].classes    # 一共有哪些类别，文件名即为类别名

    return image_datasets, dataloaders, dataset_size, class_name


def get_real_class_name(class_name: int) -> str:
    '''
    获取真正的类别名
    :param class_name: 预测出来的类别编号
    :return: 真正的类别名
    '''
    with open(config.real_class_name, 'r') as f:
        real_name_dict = json.load(f)

    return real_name_dict[str(class_name)]


def image_convert(tensor):
    '''
    将tensor格式的数据转换为numpy，方便做图
    :param tensor:
    :return:
    '''
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)  # 改变形状， Pytorch中将颜色通道放在了第一位即[c,h,w]，转换回去时需要改变位置为[h,w,c]
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def image_show():
    '''
    查看验证集部分图像
    :return:
    '''
    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2

    image_dataset, dataloader, dataset_size, class_name = get_data()
    dataiter = iter(dataloader['valid'])  # 创建迭代器
    inputs, classes = dataiter.next()   # 获取一个batch中所有的输入数据，以及对应的类别

    for index in range(columns*rows):
        ax = fig.add_subplot(rows, columns, index+1, xticks=[], yticks=[])
        ax.set_title(get_real_class_name(classes[index].item()))
        plt.imshow(image_convert(inputs[index]))
    plt.show()


if __name__ == '__main__':
    t = get_data()
    print(len(t[1]['train'].dataset))
    print(len(t[1]['train']))

    # name = get_real_class_name(1)
    # print(name)

