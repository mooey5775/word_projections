import pytest

from word_projections.word2vec import Word2VecModel

@pytest.fixture(scope='module')
def word2vec_model():
    return Word2VecModel('./GoogleNews-vectors-negative300.bin')
