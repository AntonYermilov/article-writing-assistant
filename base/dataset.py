from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import List

import pandas as pd

from .textutils import normalize_text


DATASET_FOLDER = Path('resources', 'datasets')


class RawDataset(ABC):
    def __init__(self, name: str):
        self.path = DATASET_FOLDER / name
        self.sentences = None

    def load(self, size=None):
        texts = self._load()
        paper_text_sentences = RawDataset._tokenize(texts)
        self.sentences = list(set(filter(lambda x: len(x) > 10, map(lambda x: x.strip(), paper_text_sentences))))
        if size is not None:
            self.sentences = self.sentences[:size]
        print("len(sentences)", len(self.sentences))

    @abstractmethod
    def _load(self) -> str:
        pass

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = normalize_text(text)
        return re.split(r'[.?!]+', text)

    def get(self) -> List[str]:
        return self.sentences

    def __len__(self) -> int:
        return len(self.sentences)


class NIPSPapersDataset(RawDataset):
    def _load(self) -> str:
        df = pd.read_csv(self.path, compression='gzip', sep=',')
        return ' '.join(df['paper_text'])
