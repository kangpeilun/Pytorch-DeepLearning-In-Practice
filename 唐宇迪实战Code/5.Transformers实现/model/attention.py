# -*- coding: utf-8 -*-
# date: 2022/1/22 18:49
# Project: 唐宇迪实战Code
# File Name: attention.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import numpy as np
import torch
from torch import nn

from B站课程 import config


class ScaledDotProductAttention(nn.Module):
    '''实现注意力过程，公式计算'''

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        PS: 在encoder-decoder的Attention层中 len_q(q1, ...,qt)和len_k(k1, ...km)可能不同
        :param Q: [batch_size, n_heads, leq_q, d_q]
        :param K: [batch_size, n_heads, leq_k, d_k]
        :param V: [batch_size, n_heads, leq_v, d_v]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]
        :return:
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(config.d_k)  # [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores （用 -e9 填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask, -1e9)  # 这样就可以将 PAD 的注意力系数 mask掉

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v) 做softmax
        # scores: [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        # context: [[z1, z2, ...], [...]]向量
        # attn注意力稀疏矩阵(用于可视化)
        return context, attn


class MultiHeadAttention(nn.Module):
    '''该类可以同时实现
        Encoder的Self—Attention
        Decoder的Masked Self-Attention
        Encoder-Decoder的Attention
    '''

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        '''
            一个全连接层就像等于一个矩阵的惩罚，只是不需要 bias偏置
        '''
        self.W_Q = nn.Linear(config.d_model, config.d_q * config.n_heads, bias=False)  # q,k的维度必须相同，否则无法点积
        self.W_K = nn.Linear(config.d_model, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_model, config.d_v * config.n_heads, bias=False)
        self.fc = nn.Linear(config.n_heads * config.d_v, config.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        residual, batch_size = input_Q, input_Q.size(0)  # residual:[batch_size, len_q, d_model] 表示残差部分
        # 下面的多头的参数矩阵是放在一起做线性变化的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size,  S: seq_len,  D:dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换                拆成多头

        Q = self.W_Q(input_Q).view(batch_size, -1, config.n_heads, config.d_q).transpose(1,
                                                                                         2)  # [batch_size, n_heads, len_q, d_q]
        K = self.W_K(input_K).view(batch_size, -1, config.n_heads, config.d_k).transpose(1,
                                                                                         2)  # [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, config.n_heads, config.d_v).transpose(1,
                                                                                         2)  # [batch_size, n_heads, len_v, d_v]

        # 因为是多头，所以mask矩阵要扩展为4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        '''
            repeat(1, config.n_heads, 1, 1)  沿着指定的维度重复tensor，因为有 n_heads 个头，故需要新增的维度的值为 n_heads
            repeat中值为1的，表示 该维度 不进行重复
        '''
        attn_mask = attn_mask.unsqueeze(1).repeat(1, config.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 线面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, config.n_heads * config.d_v)
        # 在做一个projection
        output = self.fc(context)  # [batch_size, len_q, d_model]
        '''
            (output + residual) 这是一个残差连接
        '''
        return nn.LayerNorm(config.d_model).to(config.device)(output + residual), attn
