import gensim
import numpy as np

from .errors import WordNotFound400

class Word2VecModel:
    def __init__(self, file_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(file_path,
            binary=True)

    def get_word_vec(self, word):
        try:
            return self.model.wv[word]
        except:
            raise WordNotFound400

    def get_word_vec_list(self, word):
        return self.get_word_vec(word).tolist()
