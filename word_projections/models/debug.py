import numpy as np

from word_projections.errors import WordNotFound400
from .basemodel import BaseModel

class DebugModel(BaseModel):
    def __init__(self, vector_size):
        self.dims = vector_size
        if self.dims == 0:
            raise ValueError('Cannot generate vectors of dimension 0')

    def get_word_vec(self, word):
        if word == "":
            raise WordNotFound400

        np.random.seed([ord(i) for i in word])
        return np.random.rand(self.dims)

    def get_word_vec_list(self, word):
        return self.get_word_vec(word).tolist()
