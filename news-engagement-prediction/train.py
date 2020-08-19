import os
import pickle
import pytz
from termcolor import colored

import numpy as np
import math

import torch
import torch.nn as nn

from hparams import hps_data, hps_save, hps_opt, setup_hparams
from preprocess import update_hps
from model import MODELS, LSTM

def get_optimizer(model, lr=hps_opt.lr, momentum=hps_opt.momentum):
    if hps_opt.adam:
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif hps_opt.sgd:
        return torch.optim.SGD(model.paramteres(), lr=lr, momentum=momentum)

def get_criterion():
    if hps_opt.mse:
        return nn.MSELoss()
    
def set_seed():
    torch.manual_seed(hps_opt.seed)
    return hps_opt.seed

def create_folder():
    local = pytz.timezone(hps_data.timezone)
    dt = datetime.now().replace(tzinfo=pytz.utc).astimezone(local)
    folder = dt.strftime('%m-%d-%I-%M-%S-%p')
    os.mkdir(f'{hps_save.ckpts_s}/{folder}')
    return folder

def get_hps(update_dict):
    architecture = MODELS[update_dict['ml_choice']]
    hps = setup_hparams(architecture, **update_dict)
    hps = update_hps(hps)
    return hps

def save_model(hps, model, min_loss):
    print('Saving model...')
    if hps.use_min:
        min_loss = round(min_loss, 5)              
        ml_file = f'{hps_save.ckpts_s}/{hps.folder_s}/{hps.dataset_s}_{hps.count_s}_{min_loss}_model.pt'
    else:   
        ml_file = f'{hps_save.ckpts_s}/{hps.folder_s}/{hps.dataset_s}_{hps.count_s}_model.pt'
    torch.save(model, ml_file)
    return ml_file   

def get_model(hps):
    model = LSTM(hps)
    return model

def save(folder, item_name, item):
    phile = open(f'{hps_save.ckpts_s}/{folder}/{item_name}.pkl', 'wb')
    pickle.dump(item, phile)
    phile.close()    
    
def load(folder, item_name):
    phile = open(f'{hps_save.ckpts_s}/{folder}/{item_name}.pkl', 'rb')
    item = pickle.load(phile)    
    return item          

def train(hps, model, train_loader, test_loader, opt, criterion, epochs=hps_opt.epochs):
    losses = {}
    losses['train'] = []
    losses['val'] = []
    losses['min'] = math.inf
    running_out_of = 0
    
    print(colored(f'Running...', 'magenta'))        
    for e in range(epochs):
        if running_out_of >= hps.patience:
            print(f'No progress detected in {hps.patience} epochs...\n')
            break
        model.train()            
        train_loss = 0
        for inputs, labels in train_loader:
            model.zero_grad()     
            output = model(inputs)
            loss = criterion(output, labels.float())
            loss.backward()
            if hps.clip:            
                nn.utils.clip_grad_norm_(model.parameters(), hps.clip)
            opt.step()
            train_loss += loss.item()
        train_loss = train_loss/hps.batch_size
        val_loss = evaluate(hps, model, test_loader, opt, criterion)        
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        if e == 0 or (e+1)%10 == 0:
            print('Epoch: {}/{}'.format(e+1, hps.epochs))
            print(colored('Train Loss: {:.5f}'.format(losses['train'][-1]), 'red'))
            print(colored('Val Loss: {:.5f}'.format(losses['val'][-1]), 'blue'))
        if losses['val'][-1] < losses['min']:
            losses['min'] = losses['val'][-1]
            print(colored('Min Loss: {:.5f}'.format(losses['min']), 'cyan'))
            running_out_of = 0
        running_out_of+=1
    return model, ml_file, losses

def evaluate(hps, model, test_loader, opt, criterion):           
    with torch.no_grad():
        model.eval()
    val_loss = 0                 
    for inputs, labels in test_loader:
        output = model(inputs)
        loss = criterion(output, labels.float())
        val_loss += loss.item()
    val_loss = val_loss/hps.batch_size
    return val_loss

def run(train_loader, test_loader, update_dict, folder, **kwargs):
    update_dict.update(**kwargs)
    hps = get_hps(update_dict)
    model = get_model(chps)
    opt = get_optimizer(cmodel)
    criterion = get_criterion()  
    model, ml_file, losses = train(hps, model, train_loader, test_loader, opt, criterion, epochs=hps_opt.epochs)
    save_model(hps, model, losses['min'])
    save(folder, 'hps', hps)    
    save(folder, 'ml_file', ml_file)
    save(folder, 'losses', losses)
    return model, hps, ml_file, losses