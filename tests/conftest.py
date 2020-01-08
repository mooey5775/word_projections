import pytest
import pickle

from word_projections.word2vec import Word2VecModel
from word_projections.debug import DebugModel

@pytest.fixture(scope='module')
def word2vec_model():
    return Word2VecModel()

@pytest.fixture(scope='module', params=[1, 2, 3, 10, 40, 100, 300])
def debug_model(request):
    return DebugModel(request.param)

@pytest.fixture(scope='session')
def debug_model_data():
    with open('tests/unit/debug_test_data.pickle', 'rb') as f:
        return pickle.load(f)
