from flask import Flask, jsonify, request
import numpy as np

import argparse
import logging

from .models.debug import DebugModel
from .models.word2vec import Word2VecModel
from .errors import WordNotFound400, AxisNotFound400, InvalidList400
from .axis_handler import AxisHandler

app = Flask(__name__, static_url_path='', static_folder='static')
logging.basicConfig(level=logging.INFO)

AXIS_DEFS = {
    'gender': [('man', 'woman'), ('men', 'women'), ('he', 'she'), ('him', 'her'), ('his', 'her'), ('his', 'hers'), ('boy', 'girl'), ('boys', 'girls'), ('male', 'female'), ('masculine', 'feminine')],
    'class': [('rich', 'poor'), ('richer', 'poorer'), ('richest', 'poorest'), ('affluence', 'poverty'), ('affluent', 'impoverished'), ('expensive', 'inexpensive'), ('luxury', 'cheap'), ('opulent', 'needy')],
    'race': [('black', 'white'), ('blacks', 'whites'), ('Blacks', 'Whites'), ('Black', 'White'), ('African', 'European'), ('African', 'Caucasian')]
}

# Empty container to be filled on setup

class ModelContainer:
    def __init__(self):
        self.model = None
        self.axes = None

mc = ModelContainer()

# Error handlers

@app.errorhandler(WordNotFound400)
def word_not_found_error(error):
    return "Requested word not found in word embedding model", 400

@app.errorhandler(AxisNotFound400)
def axis_not_found_error(error):
    return "Requested axis not found in list of axes", 400

@app.errorhandler(InvalidList400)
def invalid_list_error(error):
    return "Invalid list format", 400

# Flask routes

@app.route("/get_vec/<string:word>/")
def get_word(word):
    return jsonify(mc.model.get_word_vec_list(word))

@app.route("/get_axes/")
def get_axes():
    return jsonify(mc.axes.list_axes())

@app.route("/get_projection/<string:axis>/<string:word>/")
def get_projection(axis, word):
    axis_vec = mc.axes.get_axis_vec(axis)

    word_vec = mc.model.get_word_vec_list(word)

    return jsonify(np.dot(word_vec, axis_vec) / np.sqrt(np.dot(axis_vec, axis_vec)))

@app.route("/get_2d_projection/<string:xaxis>/<string:yaxis>/<string:word>/")
def get_2d_projection(xaxis, yaxis, word):
    yaxis_vec = mc.axes.get_axis_vec(yaxis)
    xaxis_vec = mc.axes.get_axis_vec(xaxis)

    yaxis_vec /= np.linalg.norm(yaxis_vec)
    xaxis_vec /= np.linalg.norm(xaxis_vec)

    word_vec = mc.model.get_word_vec_list(word)

    basis = np.c_[xaxis_vec, yaxis_vec]
    basis_inv = np.linalg.pinv(basis)

    return jsonify(basis_inv.dot(word_vec).tolist())

@app.route("/get_batch_projections/<string:axis>/", methods=['POST'])
def get_batch_projections(axis):
    axis_vec = mc.axes.get_axis_vec(axis)

    if not isinstance(request.json, list):
        raise InvalidList400

    word_vecs = [mc.model.get_word_vec_list(word) for word in request.json]
    projections = [np.dot(word_vec, axis_vec) / np.sqrt(np.dot(axis_vec, axis_vec)) for word_vec in word_vecs]

    return jsonify(projections)

@app.route("/get_batch_2d_projections/<string:xaxis>/<string:yaxis>", methods=['POST'])
def get_batch_2d_projections(xaxis, yaxis):
    yaxis_vec = mc.axes.get_axis_vec(yaxis)
    xaxis_vec = mc.axes.get_axis_vec(xaxis)

    yaxis_vec /= np.linalg.norm(yaxis_vec)
    xaxis_vec /= np.linalg.norm(xaxis_vec)

    if not isinstance(request.json, list):
        raise InvalidList400

    projections = []

    for word in request.json:
        word_vec = mc.model.get_word_vec_list(word)
        basis = np.c_[xaxis_vec, yaxis_vec]
        basis_inv = np.linalg.pinv(basis)

        projections.append(basis_inv.dot(word_vec).tolist())

    return jsonify(projections)

def setup_app():
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
        mc.model = DebugModel(args['debug_dims'])
    else:
        mc.model = Word2VecModel()

    # Generate axes

    logging.info("Generating axes")
    mc.axes = AxisHandler(mc.model, AXIS_DEFS)

    logging.info(f"Available axes: {mc.axes.list_axes()}")
