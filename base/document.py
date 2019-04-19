from typing import Callable, List
import numpy as np

from .sentence import Sentence


class Document:
    def __init__(self, document: str):
        self.document = document

    def split_to_sentences(self, splitter: Callable) -> np.array:
        return np.array([Sentence(sentence) for sentence in splitter(self.document)], dtype=Sentence)
