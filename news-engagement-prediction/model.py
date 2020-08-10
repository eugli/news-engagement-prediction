import numpy as np
import pandas as pd
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {
    'cnn-bilstm': ('embedding', 'cnn', 'bilstm')
}

class CNN_BiLSTM(nn.Module):
    def __init__(self, hps):
        super(CNN_BiLSTM, self).__init__()
        self.hps = hps
        self.hidden_dim = hps.hidden_dim
        self.num_layers = hps.num_layers
        
        self.embed_in = hps.embed_in
        self.embed_dim = hps.embed_dim
        self.padding_id = hps.padding_id

        self.conv_in = 1
        self.conv_out = hps.conv_out
        self.kernel_sizes = hps.kernel_sizes
        self.padding_size = 
        self.stride = hps.stride
        
        self.embed = nn.Embedding(self.embed_in, self.embed_dim, padding_idx=self.padding_id)
        # to be implemented using https://fasttext.cc/
        if hps.pretrained_embed:
            pass

        self.convs = [nn.Conv2d(self.conv_in, self.conv_out, (kernel_size, self.embed_dim), padding=(kernel_size//2, 0), stride=1) for kernel_size in self.kernel_sizes]
        
        if hps.cuda is True:
            for conv in self.convs:
                conv = conv.cuda()

        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=hps.dropout, bidirectional=True, bias=True)

        L = len(Ks) * Co + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)
        self.output_size = hps.output_size
        
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)

        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(bilstm_x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)

        # CNN and BiLSTM CAT
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)

        # linear
        cnn_bilstm_out = self.hidden2label1(F.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(F.tanh(cnn_bilstm_out))

        # output
        logit = cnn_bilstm_out
        return logit