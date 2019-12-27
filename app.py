from flask import Flask, jsonify
import gensim
app = Flask(__name__, host='0.0.0.0', port=5001, static_url_path='', static_folder='static')

@app.route("/get_vec/<string:word>/")
def getWord(word):
    try:
        return jsonify(model.wv[word].tolist())
    except KeyError:
        return jsonify(-1)

if __name__ == '__main__':
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    app.run()
