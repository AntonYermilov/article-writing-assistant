from gensim.models import KeyedVectors
from pathlib import Path
from .utils import download_gz


MODEL_FOLDER = Path('resources', 'models')


class PretrainedModel:
    def __init__(self, name: str):
        self.name = name

    def load(self) -> KeyedVectors:
        model = MODEL_FOLDER / self.name
        return KeyedVectors.load_word2vec_format(str(model))


class PretrainedFastTextModel(PretrainedModel):
    def __init__(self, name: str, url: str):
        super().__init__(name)
        self.url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{url}'

    def load(self) -> KeyedVectors:
        if not MODEL_FOLDER.exists:
            MODEL_FOLDER.mkdir()

        model = MODEL_FOLDER / self.name
        if not model.exists():
            download_gz(self.url, model)

        return super().load()
