import pytest
import pickle

from word_projections.models.word2vec import Word2VecModel
from word_projections.models.debug import DebugModel
from word_projections.models.basemodel import BaseModel
from word_projections.errors import WordNotFound400

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

@pytest.fixture(scope='module')
def parrot_model():
    class ParrotModel(BaseModel):
        def get_word_vec(self, word):
            return word

    return ParrotModel()

@pytest.fixture(scope='module')
def error_model():
    class ErrorModel(BaseModel):
        def get_word_vec(self, word):
            raise WordNotFound400

    return ErrorModel()
