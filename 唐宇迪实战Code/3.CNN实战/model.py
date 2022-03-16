# -*- coding: utf-8 -*-
# date: 2022/1/15
# Project: Pytorch深度学习实战
# File Name: model_data.py
# Description: 构建模型
# Author: Anefuer_kpl
# Email: 374774222@qq.com


import config
import torchvision
from torchvision import models
from torch import nn


def set_parameter_requires_grad(model, feature_extracting, freeze_layer):
    '''
    冻结网络中不需要训练的层
    :param model: 实例化后的模型
    :param feature_extracting:  是否对模型权重进行梯度更新，True表示不更新部分层，False表示全都更新
    :param freeze_layer: 要冻结的层数，注意要小于传入的网络的总层数
    :return: 无返回值，直接对model对象本身进行了修改
    '''
    print('Params to learn:')
    if feature_extracting:
        total_layers = len([param for param in model.parameters()])  # 记录该网络共有多少层的参数可以学习
        assert (freeze_layer < total_layers), f'the total_layers is {total_layers}, but freeze_layer is {freeze_layer}'
        print('total layers:',total_layers)
        '''
              model_data.named_parameters() 会返回模型的对应层的 name 和 param
        '''
        for index, (name,param) in enumerate(model.named_parameters()):
            if index < freeze_layer:
                # 冻结前freeze_layer层网络
                param.requires_grad = False

            # if param.requires_grad == True:
            #     # 冻结前400层后, 剩下的层数需要进行训练
            #     params_to_update.append(param)
            # print(param.requires_grad, len(params_to_update))



def initialize_model(model_name, num_classes, feature_extract, freeze_layer, use_pretrained=True):
    '''
    初始化模型
    :param model_name: 要加载的模型名
    :param num_classes: 类别数
    :param feature_extract: 是否对模型权重进行梯度更新，True表示不更新，False表示更新
    :param freeze_layer: 要冻结的层数，注意要小于传入的网络的总层数
    :param use_pretrained: 是否加载预训练权重
    :return:
    '''

    if model_name == 'resnet':
        '''restnet152'''
        model = models.resnet152(pretrained=use_pretrained)   # 实例化模型
        # print(model_data)   # 查看模型的结构
        set_parameter_requires_grad(model, feature_extract, freeze_layer)
        # 通过上面对模型结构的观察，可以直接修改模型最后fc层的输出，只需要自己再重写一下fc层即可
        num_ftrs = model.fc.in_features   # 获取模型fc层中，in_features 的数量
        # print(num_ftrs)
        # model_data.fc 表示重写 resnet152模型的fc层
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1),    # 将输出变为一维向量
        )
        input_size = 224

    # 除了resnet152这个模型修改了输出层之外，剩下的模型在使用时仍需要进行修改输出层
    elif model_name == "alexnet":
        """ Alexnet"""
        model = models.alexnet(pretrained=use_pretrained)
        print(model)
        set_parameter_requires_grad(model, feature_extract, freeze_layer)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract, freeze_layer)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract, freeze_layer)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract, freeze_layer)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract, freeze_layer)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model_data name, exiting...")
        exit()

    print('model_data structure:\n',model)
    # PS: 注意 在加载 pytorch内置的模型时，也需要将模型分配一边device，否则会报错
    # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0
    return model.to(config.device), input_size



if __name__ == '__main__':
    initialize_model(config.model_name, config.num_classes, True, 300, False)
    # initialize_model('alexnet', config.num_classes, True, 300, False)