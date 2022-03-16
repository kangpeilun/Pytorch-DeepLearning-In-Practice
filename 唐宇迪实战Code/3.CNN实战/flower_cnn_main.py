# -*- coding: utf-8 -*-
# date: 2022/1/14
# Project: Pytorch深度学习实战
# File Name: flower_cnn_main.py
# Description: 程序主文件
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import numpy as np

import config
from model import initialize_model
from dataset import get_data, get_real_class_name, image_convert
import time
from tqdm import tqdm
import os
import copy
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, optim


def train(model, dataloaders, optimizer, scheduler, criterion, epochs, is_inception, checkpoint):
    '''
    用于模型训练
    :param model: 实例化后的模型
    :param dataloaders:
    :param optimizer: 优化器
    :param scheduler: 调整学习率，学习率衰减方法
    :param criterion: 损失函数
    :param epochs: 训练轮数
    :param is_inception: True为使用，是否使用inception网络结构 https://www.cnblogs.com/dengshunge/p/10808191.html
    :param checkpoint: 是否加载已训练的部分模型，进行断点续练，该值是一个路径
    :return:
    '''
    since = time.time()  # 记录训练开始时间
    best_acc = 0  # 纪律训练过程中最好的准确率

    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint)  # 如果checkpoint存在，就加载模型，进行断点续练
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # model.class_to_idx = checkpoint['mapping']

    model.to(config.device)  # 选择模型的训练设备

    valid_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    '''
        
    '''
    LRs = [optimizer.param_groups[0]['lr']]

    # 用于保存训练过程中在 验证集 上表现最好的模型，将其参数进行深层拷贝
    # 因为模型的参数每一个epoch都会进行更换，因此要用一个变量专门保存一下
    # 第一次 深拷贝的作用是，如果训练后的效果还没有原来的模型效果好，则仍然使用原来的模型的参数
    '''
        deepcopy会把嵌套的对象完全拷贝，并开辟一片空间，成为一个新的对象
    '''
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print('\n')
        print('-' * 10)
        # 训练和验证
        for mode in ['train', 'valid']:
            # 每个epoch都同时进行train和valid
            if mode == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0  # 当前epoch的 总 loss
            running_corrects = 0  # 当前epoch的 总 预测正确的个数

            # 把数据都遍历一遍
            # dataloaders 为字典形式
            bar = tqdm(enumerate(dataloaders[mode]), total=len(dataloaders[mode]), ascii=True, desc=f'{mode}')
            for index, (inputs, labels) in bar:
                # 将训练数据分配至指定的设别上训练
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()  # 梯度清零
                # 只有 训练 的时候计算和更新梯度
                with torch.set_grad_enabled(mode == 'train'):
                    if is_inception and mode == 'train':
                        '''
                            这里是使用inception网络结构 计算loss的过程
                        '''
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2

                    else:
                        print('inputs size', inputs.size())
                        '''
                            内置的resnet模型的输入尺寸要求，通道数在前面
                        '''
                        outputs = model(inputs)  # inputs: [batch_size, channel, h, w] [32, 3, 224, 224]
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(
                    0)  # 所有batch的总loss， loss.item()得到的是一个batch的平均loss，故需要乘上 batch中的数据量
                running_corrects += torch.sum(preds == labels.data)  # 得到所有batch中总的预测正确的个数
                batch_loss = loss.item() * inputs.size(0) / config.batch_size  # 当前batch的 平均 loss
                batch_acc = torch.sum(preds == labels.data).double() / config.batch_size  # 当前batch的 平均准确率

                bar.set_description('{} epoch({}) Loss:{:.4f} Acc:{:.4f}'.format(mode, epoch, batch_loss, batch_acc))

            epoch_loss = running_loss / len(dataloaders[mode].dataset)  # 当前epoch的 平均 loss
            epoch_acc = running_corrects.double() / len(dataloaders[mode].dataset)  # 当前epoch的 平均准确率

            time_elapsed = time.time() - since  # 得到一个epoch的训练时间
            print('{} epoch({}) finished, Time elapsed {:.0f}m {:.0f}s \t Loss:{} Acc:{}'.format(mode, epoch,
                                                                                                 time_elapsed // 60,
                                                                                                 time_elapsed % 60,
                                                                                                 epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if mode == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 用于保存训练过程中在 验证集 上表现最好的模型，将其参数进行深层拷贝
                # 第二次 深拷贝作用，是将训练过程中 效果最好的模型保存下来
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 保存模型参数
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(), # 保存优化器参数
                }
                torch.save(state, config.checkpoint)

            if mode == 'valid':
                valid_losses.append(epoch_loss)
                valid_acc_history.append(epoch_acc)
                # scheduler 用于学习率的更新
                scheduler.step(epoch_loss)

            if mode == 'train':
                train_losses.append(epoch_loss)
                train_acc_history.append(epoch_acc)

        print('Optimizer learning rate:{:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])  # 记录当前学习率

    time_elapsed = time.time() - since  # 计算所有epoch训练完成的时间
    print('Training complete in  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc:{:.4f}'.format(best_acc))

    # 训练完成后用最好一次当作模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, valid_acc_history, train_acc_history, valid_losses, train_losses, LRs


def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # 改变图片尺寸，thumbnail方法只能进行 成比例缩小，按照图片的h和w中的较小者成比例缩放
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 224))
    else:
        img.thumbnail((224, 10000))
    # crop操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    # 和valid数据相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std  # 标准化
    # 修改图片的通道，颜色通道应放在第一个位置
    # print('before img size:',img.size())
    img = img.transpose((2, 0, 1))  # 变为[c, h, w]
    # print('after img size:',img.size())

    return img


def revert_process_image(image, ax=None, title=None):
    '''还原预处理后的图片'''
    # 展示数据
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))  # 变为[h, w, c]
    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


def predict(model, image_path, dataloader=None):
    '''
    对图片进行预测
    :param model: resnet模型网络结构
    :param image_path: 要检测的图片
    :param dataloader: 从dataloader中获取检测的图片
    :return:
    '''
    # 加载模型
    checkpoint = torch.load(config.checkpoint)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    # 分配设备
    model.to(config.device)
    model.eval()

    if dataloader is None:
        image = process_image(image_path)
        image = torch.FloatTensor(image)  # pytorch中参与运算的浮点数必须是 FloatTensor 的格式
        image = image.unsqueeze(0)  # 在第一维上增加值为1的一个维度，即 [3,224,224] -> [1, 3, 224, 224]
        print('image size:',image.size())
        output = model(image.cuda())
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        pred_real_class_name = get_real_class_name(preds)
        print('pred_real_class_name:', pred_real_class_name)
    else:
        # 得到一个batch的测试数据
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        print('images size:',images.size())
        # 使用GPU进行预测
        output = model(images.cuda())
        _, preds_tensor = torch.max(output, 1)
        # 去掉 tensor中维度为1的维度，如原来的维度为(1,10,1) 使用squeeze后维度变为(10,)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        print('preds:', preds)
        # 绘制预测结果图像
        fig = plt.figure(figsize=(20, 20))
        columns = 4
        rows = 2

        for index in range(columns * rows):
            ax = fig.add_subplot(rows, columns, index + 1, xticks=[], yticks=[])
            plt.imshow(image_convert(images[index]))  # image_convert(images[index])将预测结果images中的第index个对象有tensor转换为numpy格式
            pred_real_class_name = get_real_class_name(preds[index].item())
            label_real_class_name = get_real_class_name(labels[index].item())
            ax.set_title("{} ({})".format(pred_real_class_name, label_real_class_name),
                         color=("green" if pred_real_class_name == label_real_class_name else "red"))

        plt.show()


if __name__ == '__main__':
    model, input_size = initialize_model(config.model_name, config.num_classes, config.feature_extract,
                                         config.freeze_layer, config.use_pretrained)
    image_datasets, dataloaders, dataset_size, class_name = get_data()

    is_train = False  # 控制训练和测试
    if is_train:
        # 优化器设置
        optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)  # 默认使用Adam优化器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size,
                                              gamma=config.scheduler_gamma)  # 学习率每7个epoch衰减为原来的1/10
        '''
            最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
        '''
        criterion = nn.NLLLoss()

        model, valid_acc_history, train_acc_history, valid_losses, train_losses, LRs = train(
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=config.epochs,
            is_inception=config.is_inception,
            checkpoint=config.checkpoint,
        )

    else:
        image_path = './data/test_data/img.png'
        predict(model, image_path)
        # predict(model, image_path, dataloaders['valid'])
