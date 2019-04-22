from typing import Callable
import numpy as np

from .sentence import Sentence
from tools.textutils import normalize_text


class Document:
    def __init__(self, document: str, normalize=False):
        self.document = document
        if normalize:
            self.document = normalize_text(self.document)

    def __str__(self):
        return self.document

    def split_to_sentences(self, splitter: Callable) -> np.array:
        return np.array([Sentence(sentence) for sentence in splitter(self.document)], dtype=Sentence)
