import numpy as np

from .dataset import Dataset
from .embedding_model import EmbeddingModel
from .embedding_index import EmbeddingIndex
from .sentence_splitter import SentenceSplitter
from .word_weight import WordWeight
from .sentence import Sentence


class TextIndex:
    def __init__(self, dataset: Dataset, model: EmbeddingModel, index: EmbeddingIndex,
                 splitter: SentenceSplitter, weights: WordWeight, logger=None):
        self.dataset = dataset
        self.sentences = None
        self.model = model
        self.index = index
        self.splitter = splitter
        self.weights = weights
        self.logger = logger

    def _log(self, text: str):
        if self.logger is not None:
            self.logger.info(text)

    def build(self):
        self._log('Transforming dataset to a set of embeddings')
        self.sentences, matrix = [], []
        for i, sentence in enumerate(self.dataset.get_sentences()):
            parts = self.splitter.split(sentence)
            if parts is None:
                continue

            indexed_parts = np.hstack((np.array([[i] for _ in range(parts.shape[0])]), parts))
            self.sentences.append(*indexed_parts)

            parts = np.array([self.model.word_list_embedding(part, self.weights) for part in parts], dtype=np.float32)
            matrix.append(*parts)
        self.sentences, matrix = np.array(self.sentences, dtype=np.int32), np.array(matrix, dtype=np.float32)

        self._log(f'Dataset transformation finished. Dataset size: {matrix.shape[0]}')

        self._log('Creating embedding index')
        self.index.build(matrix)

    def search(self, sentence: Sentence, neighbours: int = 1) -> np.ndarray:
        query = self.model.sentence_embedding(sentence, self.weights)
        indices = self.index.search_by_vector(query, neighbours)

        response = []
        for i in indices:
            row = self.sentences[i]
            sentence_num, tokens = row[0], row[1:]
            sentence = self.dataset.get_sentences()[sentence_num]
            response.append(Sentence(sentence.get_tokens_by_indices(tokens)))
        return np.array(response, dtype=Sentence)
