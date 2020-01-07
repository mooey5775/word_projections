import numpy as np

class DebugModel():
    def __init__(self, vector_size):
        self.vec_size = vector_size

    def get_word_vec(self, word):
        np.random.seed([ord(i) for i in word])
        return np.random.rand(self.vec_size)

    def get_word_vec_list(self, word):
        return self.get_word_vec(word).tolist()
