from abc import ABC, abstractmethod
import numpy as np

from .sentence import Sentence


class SentenceSplitter(ABC):
    @staticmethod
    @abstractmethod
    def split(sentence: Sentence) -> np.array:
        pass


class KGramSplitter(SentenceSplitter):
    @staticmethod
    def split(sentence: Sentence) -> np.array:
        """
        Returns matrix, each row corresponds to some k-gram.
        K-grams are described as lists of token indices.
        Only alphabetic tokens are left.
        :param k: number of tokens
        :return: matrix, each row is a list of indices of size k
        """
        pass
