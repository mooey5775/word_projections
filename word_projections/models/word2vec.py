import gensim.downloader as api
import numpy as np

from word_projections.errors import WordNotFound400
from .basemodel import BaseModel

class Word2VecModel(BaseModel):
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')
        self.dims = 300

    def get_word_vec(self, word):
        try:
            return self.model[word]
        except:
            raise WordNotFound400
