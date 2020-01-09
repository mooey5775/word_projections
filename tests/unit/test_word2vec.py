# This file contains unit tests for the word2vec.py file

import pytest

from word_projections.errors import WordNotFound400

NONEXISTENT_WORDS = ['a', 'shalin', 'foisjljljhkgsdn', '']

@pytest.mark.parametrize('word', NONEXISTENT_WORDS)
def test_nonexistent_words(word2vec_model, word):
    """
    GIVEN a Word2Vec model
    WHEN a non-existent word is requested
    THEN check the model throws a WordNotFound400 error
    """
    with pytest.raises(WordNotFound400):
        word2vec_model.get_word_vec(word)
