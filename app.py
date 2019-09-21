from flask import Flask, jsonify
import gensim
app = Flask(__name__)

@app.route("/get_vec/<string:word>/")
def getWord(word):
    return jsonify(model.wv[word].tolist())

if __name__ == '__main__':
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    app.run()
