import gensim.downloader as api
import numpy as np

from .errors import WordNotFound400

class Word2VecModel:
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')
        self.dims = 300

    def get_word_vec(self, word):
        try:
            return self.model[word]
        except:
            raise WordNotFound400

    def get_word_vec_list(self, word):
        return self.get_word_vec(word).tolist()
