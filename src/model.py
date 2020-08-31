import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {
    'cnn-bilstm': ('embedding', 'cnn', 'bilstm', 'linear')
}

class CNN_BiLSTM(nn.Module):
    def __init__(self, hps):
        super(CNN_BiLSTM, self).__init__()
        self.hps = hps
        self.hidden_dim = hps.hidden_dim
        self.num_layers = hps.num_layers
        self.lstm_dropout = hps.lstm_dropout
        self.bidrectional = hps.bidrectional
        self.batch_first = hps.batch_first
        
        self.embed_in = hps.embed_in
        self.embed_dim = hps.embed_dim
        self.padding_id = hps.padding_id

        self.conv_in = hps.conv_in
        self.conv_out = hps.conv_out
        self.kernel_sizes = hps.kernel_sizes
        self.paddings = hps.paddings
        self.stride = hps.stride
        
        self.linear_in = hps.linear_in
        self.linear_in2 = hps.linear_in2
        self.linear_out = hps.linear_out      
        self.linear_dropout = hps.linear_dropout
        
        self.embed = nn.Embedding(self.embed_in, self.embed_dim, padding_idx=self.padding_id)
        # to be implemented using https://fasttext.cc/
        if hps.pretrained_embed:
            pass

        self.convs = [nn.Conv2d(self.conv_in, self.conv_out, (kernel_size, self.embed_dim), padding=(self.paddings[kernel_size], 0), stride=1)           for kernel_size in self.kernel_sizes]

        self.bilstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.lstm_dropout,                                   bidirectional=self.bidrectional, batch_first=self.batch_first)

        self.fc1 = nn.Linear(self.linear_in, self.linear_in2)
        self.fc2 = nn.Linear(self.linear_in2, self.linear_out)
        
        self.dropout = nn.Dropout(self.linear_dropout)

    def forward(self, x):
        embed = self.embed(x)
        
        cnn_x = embed     
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.relu(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)      
        cnn_x = torch.cat(cnn_x, 1)

        bilstm_x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(bilstm_x)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.relu(bilstm_out)
        
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)

        cnn_bilstm_out = self.fc1(F.relu(cnn_bilstm_out))
        cnn_bilstm_out = self.fc2(F.relu(cnn_bilstm_out)).squeeze(1)

        return cnn_bilstm_out