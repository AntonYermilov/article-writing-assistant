from abc import ABC, abstractmethod
from typing import Union
import numpy as np

from .sentence import Sentence


class SentenceSplitter(ABC):
    @abstractmethod
    def split(self, sentence: Sentence) -> Union[np.array, None]:
        pass


class KGramSplitter(SentenceSplitter):
    def __init__(self, k):
        """
        Creates k-gram splitter for splitting sentences into k-grams
        :param k: k-gram size
        """
        self.k = k

    def split(self, sentence: Sentence) -> Union[np.array, None]:
        """
        Returns matrix, each row corresponds to some k-gram.
        K-grams are described as lists of token indices.
        Only alphabetic tokens are left.
        :param sentence: sentence to be splitted
        :return: matrix, each row is a list of indices of size k
        """
        alphabetic_tokens = sentence.get_alphabetic_tokens()
        k_grams = len(alphabetic_tokens) - self.k + 1
        result = np.array([alphabetic_tokens[i:i+self.k] for i in range(k_grams)])
        return result if result.shape[0] != 0 else None
