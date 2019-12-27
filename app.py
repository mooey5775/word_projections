from flask import Flask, jsonify

import gensim
import argparse

import numpy as np

app = Flask(__name__, static_url_path='', static_folder='static')

AXES = {
    'test1': np.ones(300),
    'test2': np.random.rand(300)
}

@app.route("/get_vec/<string:word>/")
def get_word(word):
    try:
        return jsonify(model.wv[word].tolist())
    except KeyError:
        return jsonify(-1)
        
@app.route("/get_axes/")
def get_axes():
    return jsonify(list(AXES.keys()))

@app.route("/get_projection/<string:axis>/<string:word>/")
def get_projection(axis, word):
    if axis not in AXES:
        return jsonify(-1)
        
    axis_vec = AXES[axis]
    
    try:
        word_vec = model.wv[word].tolist()
    except KeyError:
        return jsonify(-1)
    
    return jsonify(np.dot(word_vec, axis_vec) / np.sqrt(np.dot(axis_vec, axis_vec)))

@app.route("/get_2d_projection/<string:xaxis>/<string:yaxis>/<string:word>/")
def get_2d_projection(xaxis, yaxis, word):
    if yaxis not in AXES or xaxis not in AXES:
        return jsonify(-1)
    
    yaxis_vec = AXES[yaxis]
    xaxis_vec = AXES[xaxis]
    
    try:
        word_vec = model.wv[word].tolist()
    except KeyError:
        return jsonify(-1)
    
    basis = np.c_[xaxis, yaxis]
    basis_inv = np.linalg.pinv(basis)
    
    return jsonify(basis_inv.dot(word_vec))

class RandomDictionary():
    def __init__(self, vector_size):
        self.vec_size = vector_size
    
    def __getitem__(self, key):
        np.random.seed([ord(i) for i in key])
        return np.random.rand(self.vec_size)
        
class DebugModel():
    def __init__(self, vector_size):
        self.rand_dict = RandomDictionary(vector_size)
    
    @property
    def wv(self):
        return self.rand_dict

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug', action='store_true',
                    help="use random data instead of loading model")
    args = vars(ap.parse_args())
    
    if args['debug']:
        model = DebugModel(300)
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    
    app.run(host='0.0.0.0', port=5001)
