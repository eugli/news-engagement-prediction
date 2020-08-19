import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from hparams import hps_data

def test(hps, model, test_loader):  
    inputses = []
    labelses = []
    outputs = []
    with torch.no_grad():
        model.eval()            
    for inputs, labels in test_loader:
        output = model(inputs).detach()
        inputses.append(np.array(inputs))
        labelses.append(np.array(labels))
        outputs.append(np.array(output))
    inputses = np.hstack(inputses)
    labelses = np.hstack(labelses)
    outputs = np.hstack(outputs)
    return inputses, labelses, outputs

def rescale_outputs(outputs, mean, std):
    outputs = outputs*std+mean
    return outputs

def get_detokens(tokens):
    detokens = {token:word for word, token in tokens.items()}
    return detokens

def detokenize(inputses, detokens):
    inputses_detokens = []
    for i in inputses:
        inputses_detokens.append([detokens[token] for token in i if token != 0])
    return inputses_detokens

def join_results(inputses, outputs):
    results = []
    for i, o in zip(inputses, outputs):
        results.append([i, o])
    return results