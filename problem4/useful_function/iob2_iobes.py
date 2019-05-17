# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:29:04 2019

@author: hmtbg
"""

def iob2(labels):
    for i, label in enumerate(labels):
        if label == 'O':
            continue
        label_split = label.split('-')
        if label_split[0] == 'B':
            continue
        elif i == 0 or labels[i - 1] == 'O':
            labels[i] = 'B' + label[1:]
        elif labels[i - 1][1:] == labels[i][1:]:
            continue
        else:
            labels[i] = 'B' + label[1:]
    return labels
        
        

def iobes(labels):
    new_labels = []
    for i, label in enumerate(labels):
        if label == 'O':
            new_labels.append(label)
        elif label.split('-')[0] == 'B':
            if i + 1 != len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_labels.append(label)
            else:
                new_labels.append(label.replace('B-', 'S-'))
        else:
            if i + 1 < len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_labels.append(label)
            else:
                new_labels.append(label.replace('I-', 'E-'))
    return new_labels

def update(label_list):
    new_label_list = []
    for i in range(len(label_list)):
        labels = label_list[i]
        labels = iob2(labels)
        new_label_list.append(iobes(labels))
    return new_label_list
        
        
                
                