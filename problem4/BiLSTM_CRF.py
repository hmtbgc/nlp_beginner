# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:11:00 2019

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
Wd = 1.0e-8
decay_rate = 0.010

params = {'batch_size' : Batch_size,
          'shuffle' : True,
          'num_workers' : 4}
          
# some codes from conlleval.py----------------------------------------------


ANY_SPACE = '<SPACE>'

class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)

def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def evaluate(iterable, options=None):
    if options is None:
        options = parse_args([])    # use defaults

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for line in iterable:
        line = line.rstrip('\r\n')

        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    P, R, F = 1, 1, 1

    overall, by_type = metrics(counts)
    
    if counts.token_counter > 0:
        P = 100 * overall.prec
        R = 100 * overall.rec
        F = 100 * overall.fscore
    
    return P, R, F

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start
    
    
# end of codes from conlleval.py---------------------------------------------------------


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

def print_to_file(model, datasets, length, word, label, result_path):
    _, paths = model(datasets, length)
    paths = list(itertools.chain.from_iterable(paths))
    out = []
    with open(result_path, 'a') as f:
        for i in range(len(word)):
            f.write(word[i] + ' ' + label[i] + ' ' + id_to_label[paths[i]] + '\n')
            out.append(word[i] + ' ' + label[i] + ' ' + id_to_label[paths[i]] + '\n')
    return out
        
def adjust_learning_rate(optimizers, lr):
    for param in optimizers.param_groups:
        param['lr'] = lr


    
def init_linear(input_linear):
    x = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -x, x)
    input_linear.bias.data.zero_()
    
def init_lstm(input_lstm):
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1        
                
                
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, label_to_id, embedding_dim, seq_len, hidden_dim, batch_size, weights, dropout):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.label_to_id = label_to_id
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.label_size = len(self.label_to_id)
        self.batch_size = batch_size
        self.weights = weights
        self.dropout = dropout
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(self.weights)
        self.embed_dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, \
                            bias=True, batch_first=True, bidirectional=True)
        init_lstm(self.lstm)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim, self.label_size)
        init_linear(self.fc)
        self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size))
        self.transitions.data[:, self.label_to_id[START]] = -1000
        self.transitions.data[self.label_to_id[END], :] = -1000
        self.hidden = None
        
    
    def get_feature(self, sent):
        # sent : batch, seq_len
        self.hidden = (torch.randn(2, sent.shape[0], self.hidden_dim // 2).to(device),
                       torch.randn(2, sent.shape[0], self.hidden_dim // 2).to(device))
        
        embeds = self.embedding(sent).view(-1, self.seq_len, self.embedding_dim)
        embeds = self.embed_dropout(embeds)
        # embeds : batch, seq_len, embedding_dim
        out, self.hidden = self.lstm(embeds, self.hidden)
        # out : batch, seq_len, hidden_dim
        out = self.dropout_layer(out)
        out = torch.tanh(out)
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
    
   
        
        

x = pickle.load(open('../input/params-for-ner-10/params_for_ner_10.pkl', 'rb'), encoding='UTF-8')
word_to_id = x['word_to_id']
id_to_word = x['id_to_word']
label_to_id = x['label_to_id']
id_to_label = x['id_to_label']
train_num = x['train_num']
train_label_num = x['train_label_num']
train_len = x['train_len']
val_num = x['val_num']
val_label_num = x['val_label_num']
val_len = x['val_len']
test_num = x['test_num']
test_label_num = x['test_label_num']
test_len = x['test_len']
vocab_size = x['vocab_size']
maxlength = x['maxlength']

train_word, train_tag = load_file('../input/ner2003-iobes/engtrain.iobes')
val_word, val_tag = load_file('../input/ner2003-iobes/engtesta.iobes')
weight = load_glove('../input/glove-6b-100d/glove_6B_100d.txt')
weight = torch.FloatTensor(weight)
print('weight is ok!!')

train_dataset = CustomDataset(train_num, train_label_num, train_len)
train_generator = data.DataLoader(train_dataset, **params)

net = BiLSTM_CRF(vocab_size, label_to_id, Embedding_size, \
                 maxlength, Hidden_size, Batch_size, weight, dropout_rate).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=Lr, weight_decay=Wd)
current_lr = Lr

best_f1 = 0.
index = 0
train_num, train_label_num, train_len = numpy_torch(train_num, train_label_num, train_len)
val_num, val_label_num, val_len = numpy_torch(val_num, val_label_num, val_len)
global_step = 0

for epoch in range(Epoch):
    for i, (batch_train, batch_label, batch_len) in enumerate(train_generator):
        net.train(mode=True)
        global_step += 1
        batch_train = batch_train.long().to(device)
        batch_len = batch_len.long().to(device)
        batch_label = batch_label.long().to(device)
        loss = net.neg_log_likelihood(batch_train, batch_label, batch_len)
        with open('loss.txt', 'a') as ff:
            ff.write('Epoch:{}, Step:{}, Loss:{:.3f}\n'.format(epoch + 1, i, loss.item()))
        if i % 20 == 0:
            print('Epoch:{}, Step:{}, Loss:{:.3f}'.format(epoch + 1, i, loss.item()))
            print('learning_rate:{:.3f}'.format(current_lr))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        batch_train, batch_label, batch_len = gpu_cpu(batch_train, batch_label, batch_len)
        del batch_train, batch_label, batch_len, loss
        if global_step % 600 == 0:
            index += 1
            if global_step > Epoch * len(train_generator) // 3:
                current_lr = current_lr / (1 + decay_rate * ((global_step - Epoch * len(train_generator) // 3) // 600 + 1)) 
                adjust_learning_rate(optimizer, current_lr)
            with torch.no_grad(): 
                net.eval()
                train_num, train_label_num, train_len = cpu_gpu(train_num, train_label_num, train_len)
                out1 = print_to_file(net, train_num, train_len, train_word, train_tag, result_path='train_result{}.txt'.format(index))
                p1, r1, f1_1 = evaluate(out1)
                with open('train_p_r_f1.txt', 'a') as fff:
                    fff.write(str(p1) + ' ' + str(r1) + ' ' + str(f1_1) + '\n')
                del p1, r1, f1_1, out1
                train_num, train_label_num, train_len = gpu_cpu(train_num, train_label_num, train_len)
                val_num, val_label_num, val_len = cpu_gpu(val_num, val_label_num, val_len)
                out2 = print_to_file(net, val_num, val_len, val_word, val_tag, result_path='val_result{}.txt'.format(index))
                p2, r2, f1_2 = evaluate(out2)
                if f1_2 > best_f1:
                    best_f1 = f1_2
                    torch.save(net.state_dict(), 'model_params.pkl')
                with open('val_p_r_f1.txt', 'a') as fff:
                    fff.write(str(p2) + ' ' + str(r2) + ' ' + str(f1_2) + ' ' + str(best_f1) + '\n')
                del p2, r2, f1_2, out2
                val_num, val_label_num, val_len = gpu_cpu(val_num, val_label_num, val_len)