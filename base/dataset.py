from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from tools.textutils import tokenize_to_sentences, tokenize_to_words

import pandas as pd
import numpy as np
from typing import Callable
from .document import Document
from .sentence import Sentence

from tools.textutils import normalize_text


DATASET_FOLDER = Path('resources', 'datasets')


class Dataset(ABC):
    def __init__(self, name: str):
        self.path = DATASET_FOLDER / name
        self.documents = None
        self.sentences = None

    def load(self, sentence_splitter: Callable):
        self.documents = np.array([Document(document) for document in self._load()])
        self.sentences = np.array([document.split_to_sentences(sentence_splitter)
                                   for document in self.documents], dtype=Sentence).reshape(-1)

    @abstractmethod
    def _load(self) -> np.array:
        pass

    def get_docs(self) -> np.array:
        return self.documents

    def get_sents(self) -> np.array:
        return self.sentences

    def __len__(self) -> int:
        return len(self.sentences)


class NIPSPapersDataset(Dataset):
    def _load(self) -> np.array:
        df = pd.read_csv(self.path, compression='gzip', sep=',')
        return df['paper_text'].to_numpy().astype(np.str)
