# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:10:17 2019

@author: hmtbg
"""

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
import re
import sys
import os
import torch.optim as optim
from torch.utils import data
import _pickle as cPickle
import pickle
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import itertools
from collections import defaultdict, namedtuple
import conlleval
 
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")



START = '<START>'
END = '<END>'
Epoch = 40
Embedding_size = 100
Hidden_size = 400
dropout_rate = 0.5
Batch_size = 128
Lr = 0.001
Wd = 0.06

params = {'batch_size' : Batch_size,
          'shuffle' : True,
          'num_workers' : 0}
params_two = {'batch_size' : Batch_size,
                'shuffle' : False, 
                'num_workers' : 0} 
     
                

def load_file(path):
    word = []
    label = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if line in ['\n', '\r\n']:
                continue
            else:
                line_list = line.strip().split()
                word.append(line_list[0])
                label.append(line_list[-1])
    return word, label
            
def load_glove(path):
    weight = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            lines = line.strip().split()
            weight.append(lines[1:])
    weight = np.array(weight).astype(float)
    unk_weight = np.mean(weight, axis=0)
    weight = np.insert(weight, weight.shape[0], values=unk_weight, axis=0)
    pad_weight = np.zeros([1, Embedding_size])
    weight = np.insert(weight, weight.shape[0], values=pad_weight, axis=0)
    return weight
    
class CustomDataset(data.Dataset):       
    def __init__(self, datas, labels, length):
        self.datas = datas
        self.labels = labels
        self.length = length
        
    def __getitem__(self, index):
        sent, label, leng = self.datas[index], self.labels[index], self.length[index]
        return sent, label, leng
    
    def __len__(self):
        return self.datas.shape[0]
    
def log_sum_exp(x):
    # x label_size, label_size
    max_score = torch.max(x, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(x.size(1), x.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)



def cpu_gpu(a, b, c):
    a, b, c = a.to(device), b.to(device), c.to(device)
    return a, b, c

def gpu_cpu(a, b, c):
    a, b, c = a.cpu(), b.cpu(), c.cpu()
    return a, b, c
    
def numpy_torch(a, b, c):
    a, b, c = torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long), \
              torch.tensor(c, dtype=torch.long)
    return a, b, c

def print_to_file(model, generator, word, label, result_path):
    out = []
    all_paths = []
    for epoch, (batch_x, batch_y, batch_len) in enumerate(generator):
        batch_x = batch_x.long().to(device)
        batch_y = batch_y.long().to(device)
        batch_len = batch_len.long().to(device)
        _, paths = model(batch_x, batch_len)
        paths = list(itertools.chain.from_iterable(paths))
        all_paths.append(paths)
    all_paths = list(itertools.chain.from_iterable(all_paths))
    with open(result_path, 'a') as f:
        for i in range(len(word)):
            f.write(word[i] + ' ' + label[i] + ' ' + id_to_label[all_paths[i]] + '\n')
            out.append(word[i] + ' ' + label[i] + ' ' + id_to_label[all_paths[i]] + '\n')
    return out
        
            
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, label_to_id, embedding_dim, seq_len, hidden_dim, batch_size, dropout):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.label_to_id = label_to_id
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.label_size = len(self.label_to_id)
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, \
                            bias=True, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim, self.label_size)
        self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size))
        self.transitions.data[:, self.label_to_id[START]] = -1000
        self.transitions.data[self.label_to_id[END], :] = -1000
        self.hidden = None
        
    
    def get_feature(self, sent):
        
        self.hidden = (torch.randn(2, sent.shape[0], self.hidden_dim // 2).to(device),
                       torch.randn(2, sent.shape[0], self.hidden_dim // 2).to(device))
        
        embeds = self.embedding(sent).view(-1, self.seq_len, self.embedding_dim)
        out, self.hidden = self.lstm(embeds, self.hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.seq_len, self.hidden_dim) #batch, seq_len, hidden_dim
        feats = self.fc(out) # batch, seq_len, label_size
        return feats
    
    def gold_score(self, feats, labels):
        score = torch.zeros(1).to(device)
        labels = torch.cat([torch.tensor([self.label_to_id[START]], dtype=torch.long).to(device), labels])
        for i, feat in enumerate(feats):
            score += self.transitions[labels[i], labels[i + 1]] + feat[labels[i + 1]]
        score += self.transitions[labels[-1], label_to_id[END]]
        return score
    
    def forward_algorithm(self, feats):
        # feats seq_len, label_size 
        previous = torch.full((1, self.label_size), 0).to(device)
        for i in range(len(feats)): 
            previous = previous.expand(self.label_size, self.label_size).t()
            obs = feats[i].view(1, -1).expand(self.label_size, self.label_size).to(device)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.label_to_id[END]]
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores
     
    def neg_log_likelihood(self, sents, labels, length):
        feats = self.get_feature(sents).to(device)
        real_path_score = torch.zeros(1).to(device)
        total_score = torch.zeros(1).to(device)
        for sent, label, leng in zip(feats, labels, length):
            sent = sent[:leng].to(device)
            label = label[:leng].to(device)
            real_path_score += self.gold_score(sent, label)
            total_score += self.forward_algorithm(sent)
        return total_score - real_path_score
     

    
        
    def viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.label_size), -1000)
        init_vvars[0][self.label_to_id[START]] = 0
        
        forward_var = init_vvars.to(device)
        for feat in feats:
            index = []  
            score = []  

            for next_tag in range(self.label_size):
                next_tag_var = forward_var + self.transitions[:, next_tag]
                best_tag_id = torch.max(next_tag_var, 1)[1].item()
                index.append(best_tag_id)
                score.append(next_tag_var[0][best_tag_id].view(1))
                
            forward_var = (torch.cat(score) + feat).view(1, -1)
            backpointers.append(index)

        terminal_var = forward_var + self.transitions[:, self.label_to_id[END]]
        best_tag_id = torch.max(terminal_var, 1)[1].item()
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for index in reversed(backpointers):
            best_tag_id = index[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        best_path.reverse()
        return path_score, best_path
        
    
    def forward(self, sents, length):
        feats = self.get_feature(sents)
        scores = []
        paths = []
        for feat, leng in zip(feats, length):
            feat = feat[:leng]
            score, path = self.viterbi_decode(feat)
            scores.append(score)
            paths.append(path)
        return scores, paths
    
   
        
        

x = pickle.load(open('params_for_ner_iobes.pkl', 'rb'), encoding='UTF-8')
word_to_id = x['word_to_id']
id_to_word = x['id_to_word']
label_to_id = x['label_to_id']
id_to_label = x['id_to_label']
test_num = x['test_num']
test_label_num = x['test_label_num']
test_len = x['test_len']
vocab_size = x['vocab_size']
maxlength = x['maxlength']


test_word, test_tag = load_file('engtestb.iobes')
weight = load_glove('glove_6B_100d.txt')
weight = torch.FloatTensor(weight)
print('weight is ok!!')

up_path = 'E:/NLP/nlp_beginner/NER/conll2003/results/really_work/'
net = BiLSTM_CRF(vocab_size, label_to_id, Embedding_size, \
                 maxlength, Hidden_size, Batch_size, dropout_rate).to(device)

net.load_state_dict(torch.load(up_path + 'results_4_iobes_hidden_400/model_params.pkl', map_location='cpu'))
test_dataset = CustomDataset(test_num, test_label_num, test_len)
test_generator = data.DataLoader(test_dataset, **params_two)

test_num, test_label_num, test_len = numpy_torch(test_num, test_label_num, test_len)

with torch.no_grad():
    net.eval()
    test_num, test_label_num, test_len = cpu_gpu(test_num, test_label_num, test_len)
    out1 = print_to_file(net, test_generator, test_word, test_tag, result_path=up_path + 'results_4_iobes_hidden_400/test_result.txt')
    p1, r1, f1_1 = conlleval.evaluate(out1)
    with open(up_path + 'results_4_iobes_hidden_400/test_p_r_f1.txt', 'a') as fff:
        fff.write(str(p1) + ' ' + str(r1) + ' ' + str(f1_1) + '\n')
    del p1, r1, f1_1, out1
    
