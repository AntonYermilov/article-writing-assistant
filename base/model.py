from gensim.models import KeyedVectors
from pathlib import Path
from abc import ABC, abstractmethod
from .utils import download_gz


MODEL_FOLDER = Path('resources', 'models')


class Model(ABC):
    @abstractmethod
    def load(self) -> KeyedVectors:
        pass


class FastTextModel(Model):
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{url}'

    def load(self) -> KeyedVectors:
        if not MODEL_FOLDER.exists:
            MODEL_FOLDER.mkdir()

        model = MODEL_FOLDER / self.name
        if not model.exists():
            download_gz(self.url, model)

        return KeyedVectors.load_word2vec_format(str(model))
