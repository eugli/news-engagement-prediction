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

from hparams import hps_opt

def get_optimizer(model, lr=hps_opt.lr, momentum=hps_opt.momentum):
    if hps_opt.adam:
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif hps_opt.sgd:
        return torch.optim.SGD(model.paramteres(), lr=lr, momentum=momentum)

def get_criterion():
    if hps_opt.mse:
        return nn.MSELoss()
    
# rework to follow OpenAi's metric dictionary
def train(model, train_loader, opt, criterion, hps):
    train_losses = []
    for e in range(hps.epochs):
        batch_loss = 0
        for inputs, labels in train_loader:
            model.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels.float())
            loss.backward()
            if hps.clip:
                nn.utils.clip_grad_norm_(model.parameters(), hps.clip)
            opt.step()
            batch_loss += loss.item()
        train_losses.append(batch_loss/hps.batch_size)
        
        if e==0 or (e+1)%10 == 0:
            print("Epoch: {}/{}...".format(e+1, hps.epochs))
            print("Train Loss: {:.6f}".format(train_losses[-1]))
    return train_losses

def run(model, train_loader, opt, criterion, hps):
    train_losses = train(model, train_loader, opt, criterion, hps)
    return train_losses