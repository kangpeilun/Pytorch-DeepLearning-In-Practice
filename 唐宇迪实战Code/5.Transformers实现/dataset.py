# -*- coding: utf-8 -*-
# date: 2022/1/21
# Project: 唐宇迪实战Code
# File Name: dataset.py
# Description: 构建训练数据集
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch.utils.data import DataLoader, Dataset
from B站课程 import config

# 为了便于学习，手动构建一个简单的数据集，德语->英语
sentences = [
    # 德语和英语的单词个数不要求相同
    # enc_input                dec_input           dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# 德语和英语的单词要分开建立词库
# Padding Should be Zero
# 德语分词
src_vocab = {'P':0., 'ich':1, 'mochte':2, 'ein':3, 'bier':4, 'cola':5}   # 手动为每个单词构建索引
src_idx2word = {i: w for i,w in enumerate(src_vocab)}
# 英语分词
tgt_vocab = {'P':0, 'i':1, 'want':2, 'a':3, 'beer':4, 'coke':5, 'S':6, 'E':7, '.':8}
tgt_idx2word = {i: w for i,w in enumerate(tgt_vocab)}

# ----------------------------------------------
# dataset数据特征
src_len = 5     # (源句子长度)，enc_input max sequence length
tgt_len = 6     # (输出序列单词个数)dec_input(=dec_output) max sequence length
src_vocab_size = len(src_vocab)   # 输入的不同种类词的个数
tgt_vocab_size = len(tgt_vocab)   # 输出的不同种类词的个数


# 数据构建
def make_data(sentence: list) -> torch.LongTensor:
    '''
    把单词序列转换为数字序列，对enc_input dec_input dec_output都进行编码
    :param sentence: 数据格式如下
    # enc_input                dec_input           dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    :return: LongTensor格式
    '''
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentence)):
        enc_input = [[src_vocab[n] for n in sentence[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentence[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentence[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataset(Dataset):
    '''自定义DataSet'''
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        # 获取训练数据的个数
        return self.enc_inputs.size(0)

    def __getitem__(self, index):
        # 定义一组数据的返回值
        return self.enc_inputs[index], self.dec_inputs[index], self.dec_outputs[index]


data_loader = DataLoader(MyDataset(enc_inputs, dec_inputs, dec_outputs), batch_size=config.batch_size, shuffle=True)

if __name__ == '__main__':
    pass