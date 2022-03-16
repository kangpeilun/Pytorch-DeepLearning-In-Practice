# -*- coding: utf-8 -*-
# date: 2022/1/21
# Project: 唐宇迪实战Code
# File Name: config.py
# Description:
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------
# Transformer 模型参数
d_model = 512   # Embedding Size(token embedding和position编码的维度)
d_ff = 2048     # FeedForward dimension(两次线性层中的隐藏层 512->2048->512, 线性层用来做特征提取), 最后再拼接一个projecting层
d_q = d_k = d_v = 64  # dimension of K(=Q), V(Q和K的维度需要相同，可以不和V相等，这里为了方便使得K=Q=V)
n_layers = 6    # number of Encoder of Decoder Layer (Encoder和Decoder中Block的个数)
n_heads = 8     # number of heads in Multi-Head Attention (多头注意力机制，有多少个头)

# ----------------------------------------------
# 模型超参数
epoch = 100
batch_size = 2
learn_rate = 1e-3
scheduler_step_size = 8
scheduler_gamma = 0.1