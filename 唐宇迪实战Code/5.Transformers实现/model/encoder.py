# -*- coding: utf-8 -*-
# date: 2022/1/22 16:29
# Project: 唐宇迪实战Code
# File Name: encoder.py
# Description: Transformer的Encoder和Decoder实现
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import math

import torch

from B站课程 import config
from B站课程.dataset import src_vocab_size
from model.attention import MultiHeadAttention

from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        '''
            nn.Dropout 随机将输入张量中部分元素设置为0，对于每次前向调用，被置0的元素都是随机的
        '''
        self.dropout = nn.Dropout(p=dropout)

        # TODO: 搞懂位置编码公式的含义  https://blog.csdn.net/Flying_sfeng/article/details/100996524?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&utm_relevant_index=1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [seq_len, batch_size, d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :]  # 将输入加上位置编码
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    pad mask作用: 在对value向量加权平均时，让pad对应的alpha_ij=0, 这样注意力就不会考虑到pad向量
    这里的q, k表示的是两个序列（跟注意力机制的q，k没有关系），例如 encoder_inputs:(x1, x2, ..., xm) 和 encoder_inputs:(x1, x2, ..., xm)
    :param seq_q: [batch_size, seq_len]  seq_len可以是src_len 或者 tgt_len
    :param seq_k: [batch_size, seq_len]  seq_q和seq_k可以不相等
    :return:
    '''
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # 0 是 PAD 填充字符对应的 token
    # 例如：seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k],  True代表被mask掉
    return pad_attn_mask.expand(batch_size, len_q,
                                len_k)  # [batch_size, len_q, len_k]  构成一个立方体(batch_size个(len_q, len_k)的矩阵)


class PoswiseFeedForwardNet(nn.Module):
    '''Attention block之后的全连接层'''

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param input: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs  # 残差结构
        output = self.fc(inputs)  # [batch_size, d_model]
        return nn.LayerNorm(config.d_model).to(config.device)(output + residual)


class EncoderLayer(nn.Module):
    '''Encoder block的结构'''

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 使用多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()  # Encoder block中全连接部分的网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵，值为True是，表示对应位置权重被mask掉
        :return:
        '''
        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # 得到Q，K，V(未线性变换前)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, src_len, d_model]

        return enc_outputs, attn


class Encoder(nn.Module):
    '''Transformer左边Encoder的整体结构'''

    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, config.d_model)  # 将输入的序列转换为d_model=512维的向量
        self.pos_emb = PositionalEncoding(
            config.d_model)  # Transformer中位置编码是固定的，不需要学习<不学习模型泛化能力更强>（但可以改为 进行学习的，和数据量有关，如果数据足够，可以学习）
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config.n_layers)])  # 可以快速的将多个相同的结构进行堆叠，便于代码编写

    def forward(self, enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :return: enc_outputs: [batch_size, src_len, d_model]
                 enc_self_attns: list
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵
        '''
            因为在构建数据集的时候，用字母P作为序列长度不够时的填充，其本身是无意义的
            但在transformer中，我们不希望把 P 跟别的单词关联上，因此需要添加mask把其对应的 注意力系数 去掉
        '''
        # TODO: mask为什么必须是[batch_size, len_q, len_k]的形状，mask的原理
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存接下来返回的attention的值(其作用是为了画热力图，用来查看各个词之间的关系)
        for layer in self.layers:  # for 循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # 传入的enc_outputs其实是input, 传入mask矩阵是因为要做self_attention
            enc_self_attns.append(enc_self_attn)  # 存储中间过程的attention便于可视化

        return enc_outputs, enc_self_attns
