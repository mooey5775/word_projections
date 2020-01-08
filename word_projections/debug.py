import numpy as np
from .errors import WordNotFound400

class DebugModel():
    def __init__(self, vector_size):
        self.vec_size = vector_size

    def get_word_vec(self, word):
        if word == "":
            raise WordNotFound400

        np.random.seed([ord(i) for i in word])
        return np.random.rand(self.vec_size)

    def get_word_vec_list(self, word):
        return self.get_word_vec(word).tolist()
