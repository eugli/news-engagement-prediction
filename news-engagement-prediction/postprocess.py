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
    return inputses, labelses, outputs

def rescale_outputs(outputs, mean, std):
    outputs = [(o*std+mean) for o in outputs]
    return outputs

def get_detokens(tokens):
    detokens = {token:word for word, token in tokens.items()}
    return detokens

def detokenize(inputses, detokens):
    inputses_detokens = []
    for inputs in inputses:
        inputs_detokens = []
        for seq in inputs:
            inputs_detokens.append([detokens[token] for token in seq if token != 0])
        inputses_detokens.append(inputs_detokens)
    return inputses_detokens

def join_results(inputses, outputs):
    results = []
    for inn, out in zip(inputses, outputs):
        for i, o in zip(inn, out):
            results.append([i, o])
    return results