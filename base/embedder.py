from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from gensim.models import KeyedVectors
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from tools.downloader import download_gz


MODEL_FOLDER = Path('resources', 'models')


class Embedder(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def word_embedding(self, word: str) -> Union[np.ndarray, type(None)]:
        pass

    def embedding(self, sentence: List[str], word_ind: int) -> Union[np.ndarray, type(None)]:
        return self.word_embedding(sentence[word_ind])

    @abstractmethod
    def embeddings(self, sentence: List[str]) -> Union[np.ndarray, type(None)]:
        pass

    @abstractmethod
    def dim(self) -> int:
        pass


class GensimModel(Embedder):
    def __init__(self, name: str):
        self.path = MODEL_FOLDER / name
        self.keyed_vectors = None

    def load(self):
        self.keyed_vectors = KeyedVectors.load_word2vec_format(str(self.path))

    def word_embedding(self, word: str) -> Union[np.ndarray, type(None)]:
        if word in self.keyed_vectors.vocab:
            return self.keyed_vectors.get_vector(word)
        return None

    def embeddings(self, sentence: List[str]) -> Union[np.ndarray, type(None)]:
        emb = np.array([self.embedding(sentence, ind) for ind in range(len(sentence))])
        emb = emb[emb != np.array(None)]
        return emb if emb.shape[0] != 0 else None

    def dim(self) -> int:
        return self.keyed_vectors.vector_size


class FastText(GensimModel):
    def __init__(self, name: str, url: str):
        super().__init__(name)
        self.url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{url}'

    def load(self):
        if not MODEL_FOLDER.exists:
            MODEL_FOLDER.mkdir()

        if not self.path.exists():
            download_gz(self.url, self.path)
        self.keyed_vectors = KeyedVectors.load_word2vec_format(str(self.path))


class Elmo(Embedder):
    def __init__(self, weight_file, options_file):
        # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
        #                     "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
        #                    "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        # weight_file = "~/Downloads/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
        self.weight_file = weight_file
        self.options_file = options_file
        self.elmo = None
        self.cache = (None, None)

    def load(self):
        self.elmo = ElmoEmbedder(self.options_file, self.weight_file)

    def word_embedding(self, word: str):
        # TODO
        pass

    def embedding(self, sentence: List[str], word_ind: int) -> np.ndarray:
        if sentence != self.cache[0]:
            self.cache = (sentence, self.elmo.embed_sentence(sentence))
        return self.cache[1][2, word_ind]

    def embeddings(self, sentence: List[str]) -> np.ndarray:
        if sentence != self.cache[0]:
            self.cache = (sentence, self.elmo.embed_sentence(sentence))
        return self.cache[1][2]

    def dim(self) -> int:
        return 1024
