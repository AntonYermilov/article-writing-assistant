from abc import ABC, abstractmethod
from typing import List, Dict
from nltk.tokenize import word_tokenize
import numpy as np

from tools.textutils import remove_string_special_characters
from .dataset import Dataset


class WordWeight(ABC):
    @abstractmethod
    def __init__(self, dataset: Dataset):
        pass

    @abstractmethod
    def get(self, word: str) -> np.float32:
        pass


class StandardWordWeight(WordWeight):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def get(self, word: str) -> np.float32:
        return np.float32(1)


class IDFWordWeight(WordWeight):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.idf = IDFWordWeight._idf(dataset.get_docs())

    @staticmethod
    def _idf(docs: np.array) -> Dict[str, np.float32]:
        _idf = {}
        for doc in docs:
            doc = remove_string_special_characters(str(doc))
            words = set(filter(lambda word: word.isalnum(), word_tokenize(doc, language='english')))
            for word in words:
                word = word.lower()
                if word not in _idf:
                    _idf[word] = 0
                _idf[word] += 1
        for key, value in _idf.items():
            _idf[key] = np.log(len(docs) / value)
        return _idf

    def get(self, word: str) -> np.float32:
        if word in self.idf:
            return self.idf[word]
        return np.float32(0)
