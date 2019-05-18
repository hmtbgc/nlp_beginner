# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:48 2019

@author: hmtbg
"""

import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
import re
import os
import torch.optim as optim
from torch.utils import data
import _pickle as cPickle
import pickle
import sklearn.metrics as metrics
import iob2_iobes as ii
import find_all_char as fac

torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")



START = '<START>'
END = '<END>'
Epoch = 1
Embedding_size = 100
max_char_len = 20
Hidden_size = 200
dropout_rate = 0.5
Batch_size = 32
Lr = 0.003
Wd = 0.06


params = {'batch_size' : Batch_size,
          'shuffle' : True,
          'num_workers' : 4}


def num2zero(s):
    return re.sub('\d', '0', s)



def load_file(path):
    with open(path) as file:
        sentences = []
        labels = []
        label = []
        sentence = ''
        sen_len = []
        _len = 0
        for line in file.readlines():
            if line in ['\n', '\r\n']:
                sentence = num2zero(sentence)
                sentence = sentence.lower()
                sentences.append(sentence.strip())
                sentence = ''
                labels.append(label)
                label = []
                sen_len.append(_len)
                _len = 0
            else:
                sentence = sentence + ' ' + line.split()[0]
                label.append(line.split()[3])
                _len += 1
    return sentences, labels, sen_len




def get_word(raw):
    word_all = set()
    for sent in raw:
        for word in sent.strip().split():
            word_all.add(word.lower())
    word_all = sorted(list(word_all))
    return word_all

def load_char(path):
    char_all = set()
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if line not in ['\n', '\r\n']:
                word = line.strip().split()[0].lower()
                for char in list(word):
                    char_all.add(char)
    char_all = sorted(list(char_all))
    char_to_id = {char : i+1 for i, char in enumerate(char_all)}
    id_to_char = {i+1 : char for i, char in enumerate(char_all)}
    return char_all, char_to_id, id_to_char



def load_glove(path):
    word_to_id = {}
    id_to_word = {}
    weight = []
    with open(path, 'r', encoding='UTF-8') as f:
        for i, line in enumerate(f.readlines()):
            lines = line.strip().split()
            word = lines[0]
            weight.append(lines[1:])
            word_to_id[word] = i
            id_to_word[i] = word
    weight = np.array(weight).astype(float)
    word_to_id['<UNK>'] = len(word_to_id)
    unk_weight = list(np.mean(weight, axis=0))
    weight = np.insert(weight, weight.shape[0], values=unk_weight, axis=0)
    word_to_id['<PAD>'] = len(word_to_id)
    pad_weight = np.zeros([1, Embedding_size])
    weight = np.insert(weight, weight.shape[0], values=pad_weight, axis=0)
    id_to_word[len(id_to_word)] = '<UNK>'
    id_to_word[len(id_to_word)] = '<PAD>'
    return word_to_id, id_to_word, weight
            
        

def max_length(raw):
    maxlength = 0
    for sent in raw:
        maxlength = max(maxlength, len(sent.strip().split()))
    return maxlength

def padding(raw, seq_len):
    out = []
    for sent in raw:
        sent_list = sent.strip().split()
        len_sent_list = len(sent_list)
        for i in range(seq_len - len_sent_list):
            sent_list.append('<PAD>')
        out.append(sent_list)
    return np.array(out)

def sent2num(pad, word_to_id):
    sen2num = []
    for line in pad:
        out = [word_to_id[word] if word in word_to_id else word_to_id['<UNK>'] for word in line]
        sen2num.append(out)
    return np.array(sen2num)

def get_all_label(labels):
    label_to_id = {}
    id_to_label = {}
    all_label = set()
    for line in labels:
        for label in line:
            all_label.add(label)
    all_label = sorted(list(all_label))
    label_to_id = {label : i for i, label in enumerate(all_label)}
    id_to_label = {i : label for i, label in enumerate(all_label)}
    return label_to_id, id_to_label


def labelpadding(labels, seq_len):
    out = []
    for line in labels:
        temp = line
        for i in range(seq_len - len(line)):
            temp.append('<PAD>')
        out.append(temp)
    return np.array(out)

def label2num(labelpad, label_to_id):
    lab2num = []
    for line in labelpad:
        out = [label_to_id[word] if word in label_to_id else -1 for word in line]
        lab2num.append(out)
    return np.array(lab2num)

def char2num(dataset):
    out = np.zeros([dataset.shape[0], maxlength, max_char_len])
    for i in range(dataset.shape[0]):
        for j in range(dataset[i].shape[0]):
            word = id_to_word[dataset[i][j]]
            for k, char in enumerate(list(word)):
                out[i][j][k] = char_to_id[char]
    return out
                
                



  
train_raw, train_labels, train_len = load_file('eng.train')
val_raw, val_labels, val_len = load_file('eng.testa')
test_raw, test_labels, test_len = load_file('eng.testb')
train_labels = ii.update(train_labels)
val_labels = ii.update(val_labels)
test_labels = ii.update(test_labels)
word_to_id, id_to_word, weight = load_glove('glove_6B_100d.txt')
maxlength1 = max_length(train_raw)
maxlength2 = max_length(val_raw)
maxlength3 = max_length(test_raw)
maxlength = max(maxlength1, maxlength2, maxlength3)
train_pad = padding(train_raw, maxlength)
train_num = sent2num(train_pad, word_to_id)
val_pad = padding(val_raw, maxlength)
val_num = sent2num(val_pad, word_to_id)
test_pad = padding(test_raw, maxlength)
test_num = sent2num(test_pad, word_to_id)
label_to_id, id_to_label = get_all_label(train_labels)
train_label_pad = labelpadding(train_labels, maxlength)
train_label_num = label2num(train_label_pad, label_to_id)
val_label_pad = labelpadding(val_labels, maxlength)
val_label_num = label2num(val_label_pad, label_to_id)
test_label_pad = labelpadding(test_labels, maxlength)
test_label_num = label2num(test_label_pad, label_to_id)
label_to_id[START] = len(label_to_id)
id_to_label[len(label_to_id) - 1] = START
label_to_id[END] = len(label_to_id)
id_to_label[len(label_to_id) - 1] = END
vocab_size = len(word_to_id)
char_all, char_to_id, id_to_char = fac.load_char('engtrain.iobes')
train_char = char2num(train_num)
val_char = char2num(val_num)
test_char = char2num(test_num)


with open('params_for_ner_13.pkl', 'wb') as f:
    mappings = {'word_to_id' : word_to_id,
                'id_to_word' : id_to_word,
                'label_to_id' : label_to_id,
                'id_to_label' : id_to_label,
                'train_num' : train_num,
                'train_label_num' : train_label_num,
                'val_num' : val_num,
                'val_label_num' : val_label_num,
                'test_num' : test_num,
                'test_label_num' : test_label_num,
                'maxlength' : maxlength,
                'vocab_size' : vocab_size,
                'train_len' : train_len,
                'val_len' : val_len,
                'test_len' : test_len,
                'train_char' : train_char,
                'val_char' : val_char,
                'test_char' : test_char
                 }
    cPickle.dump(mappings, f)

        
        
        
        
    
        