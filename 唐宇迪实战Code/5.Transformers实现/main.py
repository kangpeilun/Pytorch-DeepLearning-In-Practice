# -*- coding: utf-8 -*-
# date: 2022/1/21
# Project: 唐宇迪实战Code
# File Name: main.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

'''
    更详细的Transformer原理讲解视频:

'''

from B站课程 import config

import torch
from torch import nn, optim

from model.transformer import Transformer
from B站课程.dataset import data_loader, tgt_vocab, src_idx2word, tgt_idx2word

model = Transformer().to(config.device)
# 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=config.learn_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)


def greedy_decoder(model, enc_input, start_symbol):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(config.device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(
            config.device)],
                              -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        next_word = prob.data[-1]  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_symbol = next_word
        if next_symbol == tgt_vocab["E"]:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


def train(epochs):
    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in data_loader:
            """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            """
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(config.device), dec_inputs.to(
                config.device), dec_outputs.to(config.device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs,
                             dec_outputs.view(-1))  # dec_outputs.view(-1): [batch_size * tgt_len * tgt_vocab_size]
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()


def predict(enc_inputs):
    for i in range(len(enc_inputs)):
        greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(config.device), start_symbol=tgt_vocab["S"])
        print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
        print([src_idx2word[t.item()] for t in enc_inputs[i]], '->',
              [tgt_idx2word[n.item()] for n in greedy_dec_predict.squeeze()])


if __name__ == '__main__':
    train(20)
    enc_inputs, _, _ = next(iter(data_loader))
    predict(enc_inputs)