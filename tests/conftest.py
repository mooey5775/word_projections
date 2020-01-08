import pytest

from word_projections.word2vec import Word2VecModel
from word_projections.debug import DebugModel

@pytest.fixture(scope='module')
def word2vec_model():
    return Word2VecModel()

# @pytest.fixture(scope='module')
