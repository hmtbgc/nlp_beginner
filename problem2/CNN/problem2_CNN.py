# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:26:10 2019

@author: hmtbg
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


EPOCH2 = 200
num_class = 5
sentence_maxlength = 60
EMBEDDING_SIZE = 100
filter_size = [2, 3, 4, 5]
num_filter = 128
dropout_rate = 0.5

params = {'batch_size' : 64,
          'shuffle' : True,
          'num_workers' : 2}

def load_data(path):
    file = pd.read_csv(path, sep='\t', header=0, index_col='PhraseId')
    file = np.array(file)
    num = file.shape[0]
    for i in range(num):
        file[i][1] = file[i][1].lower()
    return file, num
    
def read_pretrain_vector(path):
    emb = []
    vocab = []
    dic = {}
    index = 0
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            row = line.strip().split()
            emb.append(row[1:])
            vocab.append(row[0])
            dic[row[0]] = index
            index += 1
    return vocab, emb, dic

class CustomDataset(data.Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels
        
    def __getitem__(self, index):
        sentence, label = self.datas[index], self.labels[index]
        return sentence, label
    
    def __len__(self):
        return len(self.datas)

    
    
def make_dicts(text):
    # text : list of sentences
    dicts = set()
    for sentence in text:
        sentence_list = sentence.split()
        for word in sentence_list:
            dicts.add(word)
    return dicts
        

def one_hot_vector(value, num):
    out = np.zeros(num)
    out[value] = 1
    return out


def make_data(dataset, dictts, train=True):
    num = dataset.shape[0]
    out = []
    label = []
    for i in range(num):
        sentence = dataset[i][1]
        embed = []
        sentence_list = sentence.split()
        for word in sentence_list:
            embed.append(dictts[word])
        if len(embed) <= sentence_maxlength:
            for j in range(sentence_maxlength - len(embed)):
                embed.append(0)
        else:
            embed = embed[:sentence_maxlength]
        out.append(embed)
        if train == True:
            label.append(dataset[i][2])
        
    return out, label


def softmax(y_hat):
    num = y_hat.shape[1]
    y_ave = np.sum(y_hat, axis=1) / num
    y_hat = (y_hat.T - y_ave).T
    exp_y = np.sum(np.exp(y_hat), axis=1)
    softmax_y = (np.exp(y_hat.T)) / exp_y
    softmax_y = softmax_y.T
    return softmax_y


    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.num_filters_total = num_filter * len(filter_size)
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, num_filter, (filter_size[0], EMBEDDING_SIZE), bias=True),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.MaxPool2d((sentence_maxlength - filter_size[0] + 1, 1)),
                                    )
        self.conv_block2 = nn.Sequential(nn.Conv2d(1, num_filter, (filter_size[1], EMBEDDING_SIZE), bias=True),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.MaxPool2d((sentence_maxlength - filter_size[1] + 1, 1)),
                                    )
        self.conv_block3 = nn.Sequential(nn.Conv2d(1, num_filter, (filter_size[2], EMBEDDING_SIZE), bias=True),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.MaxPool2d((sentence_maxlength - filter_size[2] + 1, 1)),
                                    )
        self.conv_block4 = nn.Sequential(nn.Conv2d(1, num_filter, (filter_size[3], EMBEDDING_SIZE), bias=True),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.MaxPool2d((sentence_maxlength - filter_size[3] + 1, 1)),
                                    )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.num_filters_total, num_class)
        
# input : [batch_size, 1, height=sentence_maxlength, width=EMBEDDING_SIZE]             
    def forward(self, x):
        pool_out = []
        conv_out1 = self.conv_block1(x) 
        pool1 = conv_out1.permute(0, 3, 2, 1)   #[batch_size, num_filters, height=1, width=1]
        pool_out.append(pool1)
        conv_out2 = self.conv_block2(x) 
        pool2 = conv_out2.permute(0, 3, 2, 1)  
        pool_out.append(pool2)
        conv_out3 = self.conv_block3(x) 
        pool3 = conv_out3.permute(0, 3, 2, 1)   
        pool_out.append(pool3)
        conv_out4 = self.conv_block4(x) 
        pool4 = conv_out4.permute(0, 3, 2, 1)   
        pool_out.append(pool4)
        h_pool = torch.cat(pool_out, 3)
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        h_pool_flat = self.dropout(h_pool_flat)
        out = self.fc(h_pool_flat)
        return out
        
    
# main -----------------------------------------
train_data, num_train = load_data('../input/sentiment-analysis-on-movie-reviews/train.tsv')
test_data, num_test = load_data('../input/sentiment-analysis-on-movie-reviews/test.tsv')
glove_vocab, glove_emb, dic = read_pretrain_vector('../input/glove-6b-100d/glove_6B_100d.txt')


train_sentence = []
test_sentence = []
for i in range(num_train):
    train_sentence.append(train_data[i][1])
for i in range(num_test):
    test_sentence.append(test_data[i][1])
train_dict = make_dicts(train_sentence)
test_dict = make_dicts(test_sentence)
tot_dict = train_dict | test_dict
tot_dict = sorted(list(tot_dict))
tot_num_word = len(tot_dict)
dicts = {w : i for i, w in enumerate(tot_dict)}

pretrain_weight = np.zeros([tot_num_word, EMBEDDING_SIZE])
for i in range(tot_num_word):
    word = tot_dict[i]
    if word in glove_vocab:
        num = dic[word]
        vector = glove_emb[num]
        vector = np.array(vector)
        pretrain_weight[i] = vector
    else:
        pretrain_weight[i] = np.random.rand(1, EMBEDDING_SIZE) 
        

train_emb, train_label = make_data(train_data, dicts, train=True)
test_emb, _ = make_data(test_data, dicts, train=False)
train_emb = np.array(train_emb)
train_label = np.array(train_label)
test_emb = np.array(test_emb)

   
train_set = CustomDataset(train_emb, train_label)
train_generator = data.DataLoader(train_set, **params)
 
net = CNN().to(device)
weight = torch.FloatTensor(pretrain_weight)
embedd = nn.Embedding.from_pretrained(weight).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.003, weight_decay=0.001)
#
net.train(mode=True)
for epoch in range(EPOCH2):
    for i, (batch_train, batch_label) in enumerate(train_generator):
        batch_train = batch_train.type(torch.long)
        batch_train = batch_train.to(device)
        batch_train = embedd(batch_train)
        batch_train = batch_train.unsqueeze(1)
        batch_label = batch_label.to(device)
        out = net(batch_train)
        batch_label = batch_label.long()
        loss = criterion(out, batch_label)
#        print('Epoch:{}, Step:{}, loss:{:.3f}'.format(epoch + 1, i + 1, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

#testing---------------------------------------
net.eval()
k = num_test // 10000 + 1
for i in range(k):
    begin = i * 10000
    end = (i + 1) * 10000
    if end > num_test:
        end = num_test
    mini_test = test_emb[begin : end]
    mini_test = torch.from_numpy(mini_test).long()
    mini_test = mini_test.to(device)
    mini_test = embedd(mini_test)
    mini_test = mini_test.unsqueeze(1)
    result = net(mini_test)
    result = result.cpu()
    result = result.detach().numpy()
    result = softmax(result)
    result_ = np.argmax(result, axis=1)
    result_ = list(result_)
    num_list = list(range(156061 + begin, 156061 + end))
    dataframe = pd.DataFrame({'PhraseId':num_list, 'Sentiment':result_})
    dataframe.to_csv('q2_L2_dropout_textcnn_mySubmission%d.csv' % (i + 1), index=False, sep=',')