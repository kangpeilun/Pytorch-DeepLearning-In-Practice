# -*- coding: utf-8 -*-
# date: 2022/1/21
# Project: 唐宇迪实战Code
# File Name: model.py
# Description: transformer模型及其组件
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from B站课程 import config
from B站课程.dataset import tgt_vocab_size
from model.encoder import Encoder
from model.decoder import Decoder

from torch import nn

class Transformer(nn.Module):
    '''Transformer模型整体架构'''
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(config.device)  # transformer模型左边Encoder block部分
        self.decoder = Decoder().to(config.device)  # 模型右边Decoder block部分
        self.projection = nn.Linear(config.d_model, tgt_vocab_size, bias=False).to(config.device) # 模型右边Decoder block后面紧接着的输出部分

    def forward(self, enc_inputs, dec_inputs):
        '''
        Transformers的两个输入: 两个序列
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是 [batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attns: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        '''
            enc_self_attns, dec_self_attns, dec_enc_attns 作用是为了可视化 attention系数
        '''
        outputs = dec_logits.view(-1, dec_logits.size(-1))  # [batch_size * tgt_len, tgt_vocab_size]
        return outputs, enc_self_attns, dec_self_attns, dec_enc_attns