from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize

from .document import Document
from .sentence import Sentence

DATASET_FOLDER = Path('resources', 'datasets')


class Dataset(ABC):
    def __init__(self, name: str):
        self.path = DATASET_FOLDER / name
        self.documents = None
        self.sentences = None

    def load(self, sentence_splitter: Callable = sent_tokenize):
        # noinspection PyTypeChecker
        self.documents = np.array([Document(document, normalize=True) for document in self._load()])
        self.sentences = np.hstack([document.split_to_sentences(sentence_splitter)
                                   for document in self.documents]).astype(Sentence)
        return self

    def save(self, path: Path):
        with path.open('w') as out:
            for sentence in self.sentences:
                out.write(str(sentence))
                out.write(' ')

    @abstractmethod
    def _load(self) -> np.array:
        pass

    def get_docs(self) -> np.array:
        return self.documents

    def get_sentences(self) -> np.array:
        return self.sentences

    def __len__(self) -> int:
        return len(self.sentences)


class NIPSPapersDataset(Dataset):
    def _load(self) -> np.array:
        df = pd.read_csv(self.path, compression='gzip', sep=',')
        return df['paper_text'].to_numpy().astype(np.str)
