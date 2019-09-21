from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from gensim.models import KeyedVectors
# from allennlp.commands.elmo import ElmoEmbedder

from tools.downloader import download_gz
from .sentence import Sentence
from .word_weight import WordWeight


MODEL_FOLDER = Path('resources', 'models')


class EmbeddingModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def word_embedding(self, word: str) -> np.array:
        pass

    def sentence_embedding(self, sentence: Sentence, word_weight: WordWeight) -> np.array:
        embedding = None
        for word in sentence.get_tokens_by_indices(sentence.get_alphabetic_tokens()):
            word_embedding = self.word_embedding(word) * word_weight.get(word)
            embedding = embedding + word_embedding if embedding is not None else word_embedding
        return embedding if embedding is not None else np.zeros(self.dim(), dtype=np.float)

    def word_list_embedding(self, word_list: np.array, word_weight: WordWeight) -> np.array:
        return self.sentence_embedding(Sentence(word_list), word_weight)

    def word_embeddings_from_sentence(self, sentence: Sentence) -> np.array:
        tokens = sentence.get_tokens_by_indices(sentence.get_alphabetic_tokens())
        return np.array([self.word_embedding(token) for token in tokens], dtype=np.float)

    def word_embeddings_from_word_list(self, word_list: np.array) -> np.array:
        return self.word_embeddings_from_sentence(Sentence(word_list))

    @abstractmethod
    def dim(self) -> int:
        pass


class GensimModel(EmbeddingModel):
    def __init__(self, name: str):
        super().__init__()
        self.path = MODEL_FOLDER / name
        self.keyed_vectors = None

    def load(self):
        self.keyed_vectors = KeyedVectors.load_word2vec_format(str(self.path))
        return self

    def word_embedding(self, word: str) -> np.array:
        if word in self.keyed_vectors.vocab:
            return self.keyed_vectors.get_vector(word)
        return np.zeros(self.dim())

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
        return self


class Elmo(EmbeddingModel):
    def __init__(self):
        super().__init__()

    def load(self):
        pass
        return self

    def word_embedding(self, word: str) -> np.array:
        pass

    def dim(self) -> int:
        pass

"""
class Elmo(EmbeddingModel):
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
"""