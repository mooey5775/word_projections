from flask import Flask, jsonify
import numpy as np

import gensim
import argparse
import logging

app = Flask(__name__, static_url_path='', static_folder='static')
logging.basicConfig(level=logging.INFO)

# Axis definitions

AXES = {}

AXIS_DEFS = {
    'gender': [('man', 'woman'), ('men', 'women'), ('he', 'she'), ('him', 'her'), ('his', 'her'), ('his', 'hers'), ('boy', 'girl'), ('boys', 'girls'), ('male', 'female'), ('masculine', 'feminine')],
    'class': [('rich', 'poor'), ('richer', 'poorer'), ('richest', 'poorest'), ('affluence', 'poverty'), ('affluent', 'impoverished'), ('expensive', 'inexpensive'), ('luxury', 'cheap'), ('opulent', 'needy')],
    'race': [('black', 'white'), ('blacks', 'whites'), ('Blacks', 'Whites'), ('Black', 'White'), ('African', 'European'), ('African', 'Caucasian')]
}

# Error definitions

class WordNotFound400(Exception):
    pass

@app.errorhandler(WordNotFound400)
def word_not_found_error(error):
    return "Requested word not found in word embedding model", 400

class AxisNotFound400(Exception):
    pass

@app.errorhandler(AxisNotFound400)
def axis_not_found_error(error):
    return "Requested axis not found in list of axes", 400

# Helper functions

def _get_word_vec(word):
    try:
        return model.wv[word].tolist()
    except:
        raise WordNotFound400

def _get_axis_vec(axis):
    if axis not in AXES:
        raise AxisNotFound400

    return AXES[axis]

# Debug mode random dictionary

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

# Flask routes

@app.route("/get_vec/<string:word>/")
def get_word(word):
    return jsonify(_get_word_vec(word))

@app.route("/get_axes/")
def get_axes():
    return jsonify(list(AXES.keys()))

@app.route("/get_projection/<string:axis>/<string:word>/")
def get_projection(axis, word):
    axis_vec = _get_axis_vec(axis)

    word_vec = _get_word_vec(word)

    return jsonify(np.dot(word_vec, axis_vec) / np.sqrt(np.dot(axis_vec, axis_vec)))

@app.route("/get_2d_projection/<string:xaxis>/<string:yaxis>/<string:word>/")
def get_2d_projection(xaxis, yaxis, word):
    yaxis_vec = _get_axis_vec(yaxis)
    xaxis_vec = _get_axis_vec(xaxis)

    yaxis_vec /= np.linalg.norm(yaxis_vec)
    xaxis_vec /= np.linalg.norm(xaxis_vec)

    word_vec = _get_word_vec(word)

    basis = np.c_[xaxis_vec, yaxis_vec]
    basis_inv = np.linalg.pinv(basis)

    return jsonify(basis_inv.dot(word_vec).tolist())

if __name__ == '__main__':
    # Check for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug', action='store_true',
                    help="use random data instead of loading model")
    args = vars(ap.parse_args())

    # Selectively load model based on debug mode
    logging.info("Loading model...")
    if args['debug']:
        logging.warning("DEBUG MODE ENABLED")
        logging.warning("DO NOT RUN IN PRODUCTION ENVIRONMENT")
        model = DebugModel(300)
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    # Generate axes
    logging.info("Generating axes")
    for axis in AXIS_DEFS:
        word_pairs = AXIS_DEFS[axis]
        all_axis_vecs = [model.wv[wp[0]] - model.wv[wp[1]] for wp in word_pairs]
        AXES[axis] = np.average(all_axis_vecs, axis=0)

    logging.info(f"Available axes: {list(AXES.keys())}")

    app.run(host='0.0.0.0', port=5001)
