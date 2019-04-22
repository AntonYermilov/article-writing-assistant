import numpy as np
from nltk.tokenize import word_tokenize


class Sentence:
    def __init__(self, sentence):
        if isinstance(sentence, str):
            self.tokens = np.array(word_tokenize(sentence), np.str)
        else:
            self.tokens = sentence

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def get_tokens_by_indices(self, indices: np.array) -> np.array:
        """
        Returns list of tokens by their indices. All indices should exist!
        :param indices: indices of tokens
        :return: list of tokens
        """
        return self.tokens[indices]

    def get_alphabetic_tokens(self) -> np.array:
        """
        Returns list of alphabetic tokens indices
        :return: list of alphabetic tokens indices
        """
        isalpha = np.array([str.isalpha(token) for token in self.tokens])
        return np.arange(len(self))[isalpha]