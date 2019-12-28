from flask import Flask, jsonify

import gensim
import argparse

import numpy as np

app = Flask(__name__, static_url_path='', static_folder='static')

AXES = {}

AXIS_DEFS = {
    'gender': [('man', 'woman'), ('men', 'women'), ('he', 'she'), ('him', 'her'), ('his', 'her'), ('his', 'hers'), ('boy', 'girl'), ('boys', 'girls'), ('male', 'female'), ('masculine', 'feminine')],
    'class': [('rich', 'poor'), ('richer', 'poorer'), ('richest', 'poorest'), ('affluence', 'poverty'), ('affluent', 'impoverished'), ('expensive', 'inexpensive'), ('luxury', 'cheap'), ('opulent', 'needy')],
    'race': [('black', 'white'), ('blacks', 'whites'), ('Blacks', 'Whites'), ('Black', 'White'), ('African', 'European'), ('African', 'Caucasian')]
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

    yaxis_vec /= np.linalg.norm(yaxis_vec)
    xaxis_vec /= np.linalg.norm(xaxis_vec)

    try:
        word_vec = model.wv[word].tolist()
    except KeyError:
        return jsonify(-1)

    basis = np.c_[xaxis_vec, yaxis_vec]
    basis_inv = np.linalg.pinv(basis)

    return jsonify(basis_inv.dot(word_vec).tolist())

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
    # Check for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug', action='store_true',
                    help="use random data instead of loading model")
    args = vars(ap.parse_args())

    # Selectively load model based on debug mode
    print("[INFO] Loading model...")
    if args['debug']:
        print("DEBUG MODE")
        model = DebugModel(300)
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    # Generate axes
    print("[INFO] Generating axes")
    for axis in AXIS_DEFS:
        word_pairs = AXIS_DEFS[axis]
        all_axis_vecs = [model.wv[wp[0]] - model.wv[wp[1]] for wp in word_pairs]
        AXES[axis] = np.average(all_axis_vecs, axis=0)

    print("[INFO] Available axes:")
    print(list(AXES.keys()))

    app.run(host='0.0.0.0', port=5001)
