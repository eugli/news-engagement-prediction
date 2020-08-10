import os
import json
import html
import re
from collections import OrderedDict
from collections import Counter
from string import punctuation

import numpy as np
import pandas as pd
import random
import math

import matplotlib.pyplot as plt
import matplotlib.dates as dates

import matplotlib.ticker as ticker
import matplotlib.dates as md

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from hparams import hps_data

def read_data(max_count=hps_data.max_count):
    # https://webhose.io/free-datasets/popular-news-articles/
    data_path = hps_data.data_path_wh_popular
    files = os.listdir(data_path)
    data_all = []
 
    if hps_data.use_all_data:
        max_count = len(files)
        
    for file in os.listdir(data_path)[:max_count]:
        with open(f"{data_path}/{file}", encoding='utf-8') as json_file:
            data = json.load(json_file)
            data_all.append(data)
    return data_all

def remove_keys(data_all):
    for file in data_all:
        for thread_key in file['thread'].keys():
            file[thread_key] = file['thread'][thread_key]
            
        keys = list(file.keys())
        for key in keys:
            if not key in hps_data.keep_keys:
                file.pop(key, None)
    return data_all

def pprint(file):
    print(json.dumps(file, indent=hps_data.indent))

def cal_all_engagements(data_all):
    for file in data_all:
        engagements = {}
        engagements['log_weigh'] = cal_engagement(file)
        engagements['log_no_weigh'] = cal_engagement(file, comment_weight=1)
        engagements['no_log_weigh'] = cal_engagement(file, take_log=False)
        engagements['no_log_no_weight'] = cal_engagement(file, comment_weight=1, take_log=False)
        file['engagement_scores'] = engagements
        file['engagement_scores']['original'] = file['performance_score']
        file.pop('performance_score', None)
        file.pop('social', None)
    return data_all
    
def cal_engagement(file, comment_weight=hps_data.comment_weight, take_log=hps_data.take_log):
    engagement = 0
    for key in file['social'].keys():
        for metric in file['social'][key]:
            if metric != 'likes':
                if metric == 'comments':
                    engagement += comment_weight * file['social'][key][metric]
                else:
                    engagement += file['social'][key][metric]
    try:
        domain_rank = math.log(file['domain_rank']) if take_log else file['domain_rank']
    except:
        domain_rank = 1
    engagement *= domain_rank
    return engagement
            
def order_keys(data_all):
    data_all_ordered = []
    for file in data_all:
        if file['sanitized_title'] != 'fail':
            new_file = {key : file[key] for key in hps_data.key_order if key in file}
            data_all_ordered.append(new_file)
    return data_all_ordered

def sanitize_text(text):
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(re.compile('<.*?>'), '', text)    
    try: 
        right = text[:text.index("|")] 
        left = text[text.index("|"):]
        text = right if len(right) > len(left) else left
    except: 
        pass
    try: 
        remaining = text[text.rindex("-"):]
        if any(x in remaining for x in hps_data.banned):
            text = text[:text.rindex("-")]  
    except:
        pass
    text = ''.join([c for c in text if c not in hps_data.punct])
    text = ' '.join(text.split())    
    test_text = ''.join([c for c in text if c not in hps_data.allowed])
    if test_text.isascii():
        return text
    else:
        return 'fail'

def get_titles(data_all):
    titles = []
    for file in data_all:
        title = sanitize_text(file['title'])
        file['sanitized_title'] = title
        if title != 'fail':
            titles.append(title)
    return titles
        
def get_all_text(titles):
    all_text = ''
    for title in titles:
        all_text += title + ' '
    return all_text

def get_words(all_text):
    return all_text.split()

def get_scores(data_all_ordered, score=hps_data.score):
    scores = []
    for file in data_all_ordered:
        scores.append(file['engagement_scores'][score])
    return scores
    
def tokenize_words(words):
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    tokens = {word:ii for ii, word in enumerate(vocab, 1)}
    return tokens

def tokenize_titles(titles, tokens):
    title_tokens = []
    for title in titles:
        title_tokens.append([tokens[word] for word in title.split()])
    return title_tokens

def get_title_lengths(title_tokens):
    title_lengths = Counter([len(title) for title in title_tokens])
    return title_lengths

def remove_shorts(title_tokens, titles, scores, min_len=hps_data.min_len):
    title_tokens_temp = title_tokens.copy()
    for title in title_tokens:
        if len(title) < min_len:
            scores.pop(title_tokens.index(title))
            titles.pop(title_tokens.index(title))
            title_tokens.remove(title)
    return title_tokens, titles, scores

def pad_titles(title_tokens, seq_length=hps_data.seq_length):
    padded_titles = []
    for title in title_tokens:
        if len(title) > seq_length:
            title = title[:seq_length]
        else:
            zeros = np.zeros(seq_length - len(title), dtype=int)
            title = np.concatenate((zeros, title))
        padded_titles.append(title)        
    return np.array(padded_titles)

def update_hps(hps, tokens):
    hps.embed_in = len(tokens)
    hps.linear_in = len(hps.kernel_sizes)*hps.conv_out+hps.hidden_dim*hps.num_layers
    return hps

def get_mean_std(data):
    return np.mean(data), np.std(data)

def scale_data(data, mean, std):
    return list((data-mean)/std)
    
def split_data(data, split_frac=hps_data.split_frac):
    data = np.array(data)
    split_idx = int(len(data)*split_frac)
    train, remaining = data[:split_idx], data[split_idx:]
    
    remaining_idx = int(len(remaining)*.5)
    val, test = remaining[:remaining_idx], remaining[remaining_idx:]
    return train, val, test

def create_tensor_dataset(data_x, data_y):
    return TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))

def create_loader(data, batch_size=hps_data.batch_size):
    return DataLoader(data, shuffle=True, batch_size=batch_size)