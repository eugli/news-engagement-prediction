import pickle

import numpy as np

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from train import load

def setup_api():
    app = Flask(__name__)
    api = Api(app)

def setup_parser():
    parser = reqparse.RequestParser()
    parser.add_argument('query')

class PredictVirality(Resource):
    def get(self):
        args = parser.parse_args()
        user_query = args['query']

        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)

        output = {'prediction': pred_text}
        return output

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True)