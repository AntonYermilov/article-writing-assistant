from abc import ABC, abstractmethod
from pathlib import Path
import re

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from faiss import IndexFlatL2

from .textutils import normalize_text, text2vec


DATASET_FOLDER = Path('resources', 'datasets')


class Dataset:
    def __init__(self, sentences: list):
        self.sentences = sentences

    def apply_model(self, model: KeyedVectors) -> IndexFlatL2:
        dim = model.vector_size
        dataset_size = len(self.sentences)
        dataset = np.zeros((dataset_size, dim), dtype=np.float32)
        for i, sentence in enumerate(self.sentences):
            dataset[i] = text2vec(sentence, model)
        index = IndexFlatL2(dim)
        index.add(dataset)
        return index

    def get_sentence(self, index: int) -> str:
        return self.sentences[index]


class RawDataset(ABC):
    def __init__(self, name: str):
        self.path = DATASET_FOLDER / name

    @abstractmethod
    def _load(self) -> str:
        pass

    @staticmethod
    def _tokenize(text: str) -> list:
        text = normalize_text(text)
        return re.split(r'[\.\?\!]+', text)

    def load(self) -> Dataset:
        text = self._load()
        sentences = RawDataset._tokenize(text)
        return Dataset(sentences)


class NIPSPapersDataset(RawDataset):
    def _load(self) -> str:
        df = pd.read_table(self.path, compression='gzip', sep=',')
        return ' '.join(df['paper_text'])