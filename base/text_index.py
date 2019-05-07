from pathlib import Path
import numpy as np

from .dataset import Dataset
from .embedding_model import EmbeddingModel
from .embedding_index import EmbeddingIndex
from .sentence_splitter import SentenceSplitter
from .word_weight import WordWeight
from .sentence import Sentence


class TextIndex:
    INDEX_DIR = Path('resources', 'index')

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

    @staticmethod
    def _add_vectors_to_index(index_file: Path, vectors: np.array):
        with index_file.open(mode='ab') as out:
            out.write(vectors)

    def build(self, index_filename: str):
        index_file = TextIndex.INDEX_DIR / index_filename
        is_new_index = not index_file.exists()
        index_file.touch()

        if is_new_index:
            self._log(f'Creating new index: {index_file}')
        else:
            self._log(f'Using existing index: {index_file}')

        self.sentences = []

        number_of_sentences = len(self.dataset.get_sentences())
        percent_size = number_of_sentences // 100

        for i, sentence in enumerate(self.dataset.get_sentences()):
            if i % percent_size == 0:
                self._log(f'{i}/{number_of_sentences} sentences processed')

            parts = self.splitter.split(sentence)
            if parts is None:
                continue

            indexed_parts = np.hstack((np.array([[i] for _ in range(parts.shape[0])]), parts))
            self.sentences += list(indexed_parts)

            vectors = np.array([self.model.word_list_embedding(sentence.get_tokens_by_indices(part), self.weights)
                                for part in parts], dtype=np.float32)
            if is_new_index:
                TextIndex._add_vectors_to_index(index_file, vectors)

        self.sentences = np.array(self.sentences, dtype=np.int32)

        self._log(f'{number_of_sentences}/{number_of_sentences} sentences processed')
        self._log(f'Dataset size: {self.sentences.shape[0]} tokens')
        self._log(f'Index size: {index_file.stat().st_size / 1024**2:0.2f} MB')

        self._log('Creating embedding index')
        self.index.build(index_file, self.model.dim())

    def search(self, sentence: Sentence, neighbours: int = 1, splitter_neighbours: int = 10) -> np.array:
        sentence_parts = self.splitter.split(sentence)
        if sentence_parts is None:
            return None

        query = np.array([
            self.model.word_list_embedding(sentence.get_tokens_by_indices(part), self.weights) for part in sentence_parts
        ])

        indices = self.index.search_by_matrix(query, splitter_neighbours)
        inv_indices, indices = np.unique(indices.flatten(), return_inverse=True)

        bins = np.bincount(indices)
        most_frequent_indices = inv_indices[np.argsort(bins)[:-(neighbours+1):-1]]

        sentence_indices = np.array([self.sentences[i,0] for i in most_frequent_indices])
        response = self.dataset.get_sentences()[sentence_indices]
        return response
