# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:22:03 2019

@author: hmtbg
"""

import numpy as np

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

char_all, char_to_id, id_to_char = load_char('engtrain.iobes')
