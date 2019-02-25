from gensim.models import KeyedVectors
from pathlib import Path
from .utils import download_gz


MODEL_FOLDER = Path('resources', 'models')


class PretrainedModel:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> KeyedVectors:
        if not self.path.exists():
            raise RuntimeError(f'Embedding model {str(self.path)} does not exist!')
        return KeyedVectors.load_word2vec_format(str(self.path))


class PretrainedFastTextModel(PretrainedModel):
    def __init__(self, name: str, url: str):
        super().__init__(str(MODEL_FOLDER / name))
        self.url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{url}'

    def load(self) -> KeyedVectors:
        if not MODEL_FOLDER.exists:
            MODEL_FOLDER.mkdir()

        if not self.path.exists():
            download_gz(self.url, self.path)

        return KeyedVectors.load_word2vec_format(str(self.path))
