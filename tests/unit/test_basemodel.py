import pytest
import numpy as np

from word_projections.errors import WordNotFound400

def test_error_passthrough(error_model):
    with pytest.raises(WordNotFound400):
        error_model.get_word_vec('')

    with pytest.raises(WordNotFound400):
        error_model.get_word_vec_list('')

def test_empty_list(parrot_model):
    assert parrot_model.get_word_vec_list(np.array([])) == []

def test_random_lists(parrot_model):
    np.random.seed(42)

    for i in range(100):
        dims = np.random.randint(low=1, high=1000)
        test_array = np.random.rand(dims)
        assert (parrot_model.get_word_vec_list(test_array)
                == test_array.tolist())
