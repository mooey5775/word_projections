import pytest
import numpy as np
import pickle

from word_projections.errors import WordNotFound400
from word_projections.debug import DebugModel

ENGLISH_WORDS = ['a', 'b', 'c', 'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
SPECIAL_CHARACTERS = ['test!', '!@#%&$%^)^(&*())', '{}|"']
UNICODE_WORDS = ['Быстрая', 'коричневая', 'лиса', 'прыгает', 'через', 'ленивую', 'собаку', '敏捷的棕色狐狸跳过了懒狗', '😀😁😂', '🤣😃😄😅😆']

def test_zero_dims():
    with pytest.raises(ValueError):
        model = DebugModel(0)

def test_empty_word(debug_model):
    with pytest.raises(WordNotFound400):
        debug_model.get_word_vec('')

    with pytest.raises(WordNotFound400):
        debug_model.get_word_vec_list('')

def _test_word(model, labels, word):
    assert np.all(model.get_word_vec(word) == labels[word])
    assert model.get_word_vec_list(word) == labels[word].tolist()

@pytest.mark.parametrize('word', ENGLISH_WORDS)
def test_english(debug_model, debug_model_data, word):
    size = debug_model.dims
    _test_word(debug_model, debug_model_data[size], word)

@pytest.mark.parametrize('word', SPECIAL_CHARACTERS)
def test_special(debug_model, debug_model_data, word):
    size = debug_model.dims
    _test_word(debug_model, debug_model_data[size], word)

@pytest.mark.parametrize('word', UNICODE_WORDS)
def test_unicode(debug_model, debug_model_data, word):
    size = debug_model.dims
    _test_word(debug_model, debug_model_data[size], word)
