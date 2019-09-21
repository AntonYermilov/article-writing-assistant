from pathlib import Path
import numpy as np

from .dataset import Dataset
from .embedding_model import EmbeddingModel
from .embedding_index import EmbeddingIndex, NSW
from .sentence_splitter import SentenceSplitter, KGramSplitter
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
        sentence_indices = np.array([self.sentences[ind, 0] for ind in indices.flatten()])
        inv_indices, sentence_indices = np.unique(sentence_indices, return_inverse=True)

        bins = np.bincount(sentence_indices)
        bin_args = np.argsort(bins)[:-(neighbours+1):-1]
        most_frequent_indices = inv_indices[bin_args]

        _newline, _space = '\n', ' '
        self.logger.info(f'Input sentence: {str(sentence).replace(_newline, _space)}')
        self.logger.info(f'Nearest sentences: {str(inv_indices).replace(_newline, _space)}')
        self.logger.info(f'Number of occurrences in sentences: {str(bins).replace(_newline, _space)}')
        self.logger.info(f'Best sentences: {str(inv_indices[bin_args]).replace(_newline, _space)}')
        self.logger.info(f'Number of occurrences in best sentences: {str(bins[bin_args]).replace(_newline, _space)}')

        response = self.dataset.get_sentences()[most_frequent_indices]
        return response


class KGramIndex:
    INDEX_DIR = Path('resources', 'index')

    def __init__(self, dataset: Dataset, model: EmbeddingModel, k_gram_size: int,
                 weights: WordWeight, logger=None):
        self.dataset = dataset
        self.sentences = None
        self.model = model
        self.k_gram_size = k_gram_size
        self.index = [NSW() for _ in range(self.k_gram_size)]
        self.splitter = KGramSplitter(k=self.k_gram_size)
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
        index_files = [KGramIndex.INDEX_DIR / index_filename.replace('.bin', f'_{i}.bin')
                       for i in range(self.k_gram_size)]

        is_new_index = False
        for index_file in index_files:
            is_new_index |= not index_file.exists()
            index_file.touch()

        if is_new_index:
            self._log(f'Creating new index: {index_files}')
        else:
            self._log(f'Using existing index: {index_files}')

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

            if is_new_index:
                for k, index_file in enumerate(index_files):
                    left = np.array([part[:k] for part in parts])
                    right = np.array([part[k+1:] for part in parts])
                    cur_parts = np.hstack((left, right))

                    vectors = np.array([self.model.word_list_embedding(sentence.get_tokens_by_indices(part), self.weights)
                                        for part in cur_parts], dtype=np.float32)
                    KGramIndex._add_vectors_to_index(index_file, vectors)

        self.sentences = np.array(self.sentences, dtype=np.int32)

        self._log(f'{number_of_sentences}/{number_of_sentences} sentences processed')
        self._log(f'Dataset size: {self.sentences.shape[0]} tokens')
        self._log(f'Index size: {sum(index_file.stat().st_size / 1024**2 for index_file in index_files):0.2f} MB')

        self._log('Creating embedding index')

        for i, index_file in enumerate(index_files):
            self.index[i].build(index_file, self.model.dim())

    def search(self, sentence: Sentence, neighbours: int = 1) -> np.array:
        sentence_parts = self.splitter.split(sentence)
        if sentence_parts is None:
            return None

        response = [[] for _ in range(len(sentence_parts) + self.k_gram_size - 1)]

        for i, part in enumerate(sentence_parts):
            emb_i = self.model.word_list_embedding(sentence.get_tokens_by_indices(part), self.weights)
            emb_i = emb_i / np.linalg.norm(emb_i)
            for k in range(self.k_gram_size):
                cur = np.hstack((part[:k], part[k+1:]))
                emb_ik = self.model.word_list_embedding(sentence.get_tokens_by_indices(cur), self.weights)
                indices = self.index[k].search_by_vector(emb_ik, neighbours)

                for ind in indices:
                    sent, k_gram = self.sentences[ind, 0], self.sentences[ind, 1:]
                    sent = self.dataset.get_sentences()[sent]
                    emb = self.model.word_list_embedding(sent.get_tokens_by_indices(k_gram), self.weights)
                    emb = emb / np.linalg.norm(emb)
                    response[i + k].append((emb_i.dot(emb), sent.get_tokens_by_indices(k_gram)[k]))

        return response
