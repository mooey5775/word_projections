from flask import Flask, jsonify
import numpy as np

import gensim
import argparse
import logging

from .debug import DebugModel
from .word2vec import Word2VecModel
from .errors import WordNotFound400, AxisNotFound400
from .axis_handler import AxisHandler

app = Flask(__name__, static_url_path='', static_folder='static')
logging.basicConfig(level=logging.INFO)

MODEL_FILE_PATH = './GoogleNews-vectors-negative300.bin'

AXIS_DEFS = {
    'gender': [('man', 'woman'), ('men', 'women'), ('he', 'she'), ('him', 'her'), ('his', 'her'), ('his', 'hers'), ('boy', 'girl'), ('boys', 'girls'), ('male', 'female'), ('masculine', 'feminine')],
    'class': [('rich', 'poor'), ('richer', 'poorer'), ('richest', 'poorest'), ('affluence', 'poverty'), ('affluent', 'impoverished'), ('expensive', 'inexpensive'), ('luxury', 'cheap'), ('opulent', 'needy')],
    'race': [('black', 'white'), ('blacks', 'whites'), ('Blacks', 'Whites'), ('Black', 'White'), ('African', 'European'), ('African', 'Caucasian')]
}

# Error handlers

@app.errorhandler(WordNotFound400)
def word_not_found_error(error):
    return "Requested word not found in word embedding model", 400

@app.errorhandler(AxisNotFound400)
def axis_not_found_error(error):
    return "Requested axis not found in list of axes", 400

# Flask routes

@app.route("/get_vec/<string:word>/")
def get_word(word):
    return jsonify(model.get_word_vec_list(word))

@app.route("/get_axes/")
def get_axes():
    return jsonify(axes.list_axes())

@app.route("/get_projection/<string:axis>/<string:word>/")
def get_projection(axis, word):
    axis_vec = axes.get_axis_vec(axis)

    word_vec = model.get_word_vec_list(word)

    return jsonify(np.dot(word_vec, axis_vec) / np.sqrt(np.dot(axis_vec, axis_vec)))

@app.route("/get_2d_projection/<string:xaxis>/<string:yaxis>/<string:word>/")
def get_2d_projection(xaxis, yaxis, word):
    yaxis_vec = axes.get_axis_vec(yaxis)
    xaxis_vec = axes.get_axis_vec(xaxis)

    yaxis_vec /= np.linalg.norm(yaxis_vec)
    xaxis_vec /= np.linalg.norm(xaxis_vec)

    word_vec = model.get_word_vec_list(word)

    basis = np.c_[xaxis_vec, yaxis_vec]
    basis_inv = np.linalg.pinv(basis)

    return jsonify(basis_inv.dot(word_vec).tolist())

# Check for debug mode

ap = argparse.ArgumentParser()
ap.add_argument('--debug', action='store_true',
                help="use random data instead of loading model")
ap.add_argument('--debug-dims', type=int, default=300,
                help="dimension of debug model data")
args = vars(ap.parse_args())

# Selectively load model based on debug mode

logging.info("Loading model...")
if args['debug']:
    logging.warning("DEBUG MODE ENABLED")
    logging.warning("DO NOT RUN IN PRODUCTION ENVIRONMENT")
    model = DebugModel(args['debug_dims'])
else:
    model = Word2VecModel(MODEL_FILE_PATH)

# Generate axes

logging.info("Generating axes")
axes = AxisHandler(model, AXIS_DEFS)

logging.info(f"Available axes: {axes.list_axes()}")
