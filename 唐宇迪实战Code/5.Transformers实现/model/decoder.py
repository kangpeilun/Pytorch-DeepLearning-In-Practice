# -*- coding: utf-8 -*-
# date: 2022/1/22 18:15
# Project: 唐宇迪实战Code
# File Name: decoder.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import numpy as np
import torch

from B站课程 import config
from B站课程.dataset import tgt_vocab_size
from model.encoder import PositionalEncoding, get_attn_pad_mask, PoswiseFeedForwardNet
from model.attention import MultiHeadAttention

from torch import nn


def get_attn_subsequence_mask(seq):
    '''
    可以打印出来看看输出结果
    生成一个上三角矩阵，对应位置值为1的，表示要被mask掉的
    这样就可以实现 不看 未来的词向量
    :param seq: [batch_size, tgt_len]
    :return:
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  # [batch_size, tgt_len, tgt_len]
    # print(attn_shape)
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成1个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class DecoderLayer(nn.Module):
    '''Decoder block的结构'''

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffc = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        :param dec_inputs: [batch_size, tgt_len, d_model]
        :param enc_outputs: [batch_size, src_len, d_model]
        :param dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        :param dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        :return:
        '''
        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)  # 这里的Q,K,V全是Decoder自己的输入
        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs = self.pos_ffc(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


class Decoder(nn.Module):
    '''Transformer模型右边Decoder的整体结构'''

    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, config.d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(config.d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config.n_layers)])  # Decoder的block

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        :param dec_inputs: [batch_size, tgt_len]
        :param enc_inputs: [batch_size, src_len]
        :param enc_outputs: [batch_size, src_len, d_model]  用在Encoder-Decoder Attention层
        :return:
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            config.device)  # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵(这个例子中decoder是没有加pad的，实际应用中都要加pad填充)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(
            config.device)  # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention: 当前时刻看不到未来的信息
        # 计算注意力的时候只能看到当前和之前的 词向量，而未来的词向量是看不到的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            config.device)  # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了PAD的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(
            config.device)  # [batch_size, tgt_len, src_len] torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad_mask矩阵
        # (因为enc是处理K、V的，求Attention时是用v1，v2，...vm去加权的，要把pad对应的v_i的相关系数设置为0，这样注意力就不会关注pad向量)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batch_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)

            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


if __name__ == '__main__':
    seq = [
        [1, 2, 3, 4, 5]
    ]
    seq = torch.LongTensor(seq)
    print(seq.size())
    mask = get_attn_subsequence_mask(seq)
    print(mask)
