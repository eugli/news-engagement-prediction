import os
import json
import pickle

import torch

from flask import Flask, request, jsonify, render_template, send_from_directory

import preprocess as pp
import postprocess as pop
from train import load

template_folder = os.path.abspath('../app/templates')
app = Flask(__name__, template_folder=template_folder)

folder = '08-30-07-07-19-PM'
hps = load(folder, 'hps')
ml_file = load(folder, 'ml_file')
model = torch.load(ml_file)
tokens = load(folder, 'tokens')
dictionary = list(tokens.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        params = request.get_json(force=True)
        title = params['title']
        prediction = get_prediction(title).item()
    return jsonify({'engagement': prediction})

@app.route('/')
def load():
    return render_template('index.html')

@app.route('/favicon.png')
def favicon():
    return send_from_directory(os.path.join(app.root_path, '../app/assets/images'), 'favicon.png')

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

def preprocess(title):
    title = pp.sanitize_text(title)
    title = pp.remove_words(title, dictionary)
    title = pp.tokenize_titles([title], tokens)
    title = pp.pad_titles(title)
    title = torch.tensor(title).to(torch.int64)
    return title

def postprocess(prediction, hps):
    prediction = pop.detach_numpy(prediction)
    prediction = pop.rescale(prediction, hps)
    return prediction

def get_prediction(title):
    title = preprocess(title)
    prediction = model.forward(title)
    prediction = postprocess(prediction, hps)
    return prediction

if __name__ == '__main__':
    app.run(port=5000, debug=True)