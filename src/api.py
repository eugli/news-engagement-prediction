import json
import pickle

import torch

from flask import Flask, request, jsonify, render_template

import preprocess as pp
import postprocess as pop
from train import load

app = Flask(__name__)
folder = '08-29-04-45-06-PM'
hps = load(folder, 'hps')
ml_file = load(folder, 'ml_file')
model = torch.load(ml_file)
tokens = load(folder, 'tokens')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        params = request.get_json(force=True)
        title = params['title']
        prediction = get_prediction(title).item()
    return jsonify({'engagement': prediction})

def load_all(folder):
    hps = load(folder, 'hps')
    ml_file = load(folder, 'ml_file')
    model = torch.load(ml_file)
    tokens = load(folder, 'tokens')
    return hps, model, tokens

def set_all(chps, cmodel, ctokens):
    global hps
    global model
    global tokens
    hps = chps
    model = cmodel
    tokens = ctokens

def preprocess(title, tokens):
    title = pp.sanitize_text(title)
    title = pp.tokenize_titles([title], tokens)
    title = pp.pad_titles(title)
    title = torch.tensor(title).to(torch.int64)
    return title

def postprocess(prediction, hps):
    prediction = pop.detach_numpy(prediction)
    prediction = pop.rescale(prediction, hps)
    return prediction

def get_prediction(title):
    title = preprocess(title, tokens)
    prediction = model.forward(title)
    prediction = postprocess(prediction, hps)
    return prediction

if __name__ == '__main__':
    app.run(port=5000, debug=True)