#!/usr/bin/env python
# coding: utf-8




import numpy as np
import torch
from torch.utils import data
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

Epoch = 100
Dropout = 0.5
Hidden_size = 512
Embedding_size = 512
Batch_size = 64
len_seq = 20
LR = 0.006

params = {'batch_size' : Batch_size,
          'shuffle' : True,
          'num_workers' : 2}




def load_data(path):
    tot = ''
    with open(path, 'r', encoding='UTF-8') as file:
        data = file.readlines()
        for i in range(len(data)):
            tot = tot + data[i]
    return tot




def dicts(raw):
    char_list = set()
    for char in raw:
        char_list.add(char)
    char_list = sorted(list(char_list))
    char2num = {char : i for i, char in enumerate(char_list)}
    tot_num = len(char_list)
    word_num_freq = np.zeros([tot_num], dtype=np.int16)
    for word in raw:
        word_num_freq[char2num[word]] += 1
    return char_list, char2num, tot_num, word_num_freq





def one_hot(num, tot_len):
    out = np.zeros(tot_len)
    out[num] = 1
    return out




def sentence_char2num(x, char2num):
    return [char2num[char] for char in x]




def sentence_num2char(x, char_list):
    return [char_list[num] for num in x]




def softmax(y):
    ave = np.mean(y)
    y = y - ave
    tot = np.sum(np.exp(y))
    out = np.exp(y) / tot
    return out



def random_choice(x, k=5):
    index = np.argsort(-x)
    character = index[:k]
    prob = x[character]
    prob = prob / prob.sum()
    out = np.random.choice(character, size=1, p=prob)
    return out




class CustomDataset(data.Dataset):
    def __init__(self, datas):
        self.datas = datas
        
    def __getitem__(self, index):
        train = self.datas[index, :]
        label = np.zeros(train.shape, dtype=np.int16)
        label[:-1], label[-1] = train[1:], train[0]
        return train, label
    
    def __len__(self):
        return self.datas.shape[0]
   





class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, dropout, num_class):
        super(RNN, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_class = num_class
        self.embed = nn.Embedding(self.num_class, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers, bias=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias=True)
        
    def forward(self, x, init=None):
        # x : batch, len_seq, num_class
        if init == None:
            h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        else:
            (h0, c0) = init
            
        word_emb = self.embed(x)
        change = word_emb.permute(1, 0, 2)
        out, (hn, cn) = self.lstm(change, (h0, c0))
        # out : Time_step, batch, hidden_size
        ts, ba, hd = out.shape
        out = out.view(ts * ba, hd)
        out = self.fc(out)
        out = out.view(ts, ba, -1)
        out = out.permute(1, 0, 2).contiguous() 
        # out : batch, Time_step, num_class
        out = out.view(-1, out.shape[2])
        return out, (hn, cn)





# main ----------------------------------------------       
raw_data = load_data('../input/poetryfromtang/poetryFromTang.txt')
raw_data = raw_data.replace('\n', '')
raw_data = raw_data.replace('\ufeff', '')
raw_data = raw_data.replace('；', ' ')
raw_data = raw_data.replace('，', ' ')
raw_data = raw_data.replace('。', ' ')
raw_data = ' '.join(raw_data.split())
char_list, char2num, tot_num, word_num_freq = dicts(raw_data)
word_freq = []
for i in range(word_num_freq.shape[0]):
    word_freq.append((char_list[i], word_num_freq[i]))
word_freq.sort(key=lambda x:x[1], reverse=True)
high_fre_word = [word_freq[i][0] for i in range(len(word_freq)) if word_freq[i][1] > 1]
high_freq = len(high_fre_word)

corpus = raw_data
num_seq = len(corpus) // len_seq
corpus = corpus[:num_seq * len_seq]




dataset = np.zeros([num_seq, len_seq], dtype=np.int16)
for i in range(num_seq):
    dataset[i] = np.array([char2num[char] for char in raw_data[i * len_seq : (i + 1) * len_seq]])




train_set = CustomDataset(dataset)
train_generator = data.DataLoader(train_set, **params)

net = RNN(emb_size=Embedding_size, hidden_size=Hidden_size, num_layers=3, dropout=Dropout, num_class=tot_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)





net.train(mode=True)
for epoch in range(Epoch):
    train_loss = 0
    for i, (batch_train, batch_label) in enumerate(train_generator):
        batch_train = batch_train.type(torch.long)
        batch_label = batch_label.type(torch.long)
        batch_train = batch_train.to(device)
        batch_label = batch_label.to(device)
        out, _ = net(batch_train)
        loss = criterion(out, batch_label.view(-1))
        train_loss +=  loss.item()
        #print('Epoch:{}, Step:{}, loss:{:.3f}'.format(epoch + 1, i + 1, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    print('perplexity:{:.3f}'.format(np.exp(train_loss / len(train_generator))))




# generate text -----------------------------------------
net.eval()
first = ['巴山上峡重复重', '君不见黄河之水天上来', '细蕊慢逐风', '月落辕门鼓角鸣', '吐谷浑盛强']
for begin in first: 
    begin2num = [char2num[x] for x in begin]
    begin2num = torch.from_numpy(np.expand_dims(np.array(begin2num), axis=0)).type(torch.long).to(device)
    _, init = net(begin2num)
    text_len = 100
    input_vector = begin2num[:, -1]
    for i in range(text_len):
        input_vector = torch.unsqueeze(input_vector, dim=0).type(torch.long).to(device)
        out, init = net(input_vector, init)
        out = torch.squeeze(out, dim=0)
        out = out.cpu()
        out = out.detach().numpy()
        next_char_index = random_choice(out, k=1)
        begin = begin + char_list[next_char_index.item()]
        input_vector = torch.Tensor([next_char_index.item()]).type(torch.long)
    with open('result.txt', 'a') as f:       
        f.write(begin)
        f.write('\n')
    
