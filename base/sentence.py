import numpy as np
from nltk.tokenize import word_tokenize


class Sentence:
    def __init__(self, sentence: str):
        self.tokens = np.array(word_tokenize(sentence), np.str)

    def __len__(self):
        return len(self.tokens)

    def get_tokens_by_indices(self, indices: np.array) -> np.array:
        """
        Returns list of tokens by their indices. All indices should exist!
        :param indices: indices of tokens
        :return: list of tokens
        """
        return self.tokens[indices]
