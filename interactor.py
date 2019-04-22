import logging
from datetime import datetime
from pathlib import Path

from nltk.tokenize import sent_tokenize

from base import dataset, embedding_index, embedding_model, word_weight, sentence_splitter
from base.dataset import Dataset
from base.embedding_index import EmbeddingIndex
from base.embedding_model import EmbeddingModel
from base.word_weight import WordWeight
from base.sentence_splitter import SentenceSplitter
from base.text_index import TextIndex

class Interactor:
    DEFAULT_LOG_FOLDER = Path('logs')

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self._create_log_handler()

    def _create_log_handler(self):
        if not Interactor.DEFAULT_LOG_FOLDER.exists():
            Interactor.DEFAULT_LOG_FOLDER.mkdir()

        current_date = datetime.now().strftime('%Y.%m.%d %H.%M.%S')
        log_filename = f'interactor {current_date}'

        file_handler = logging.FileHandler(Interactor.DEFAULT_LOG_FOLDER / log_filename)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def interact(self):
        self.logger.info('Loading dataset')
        _dataset: Dataset = dataset.nips_papers
        _dataset.load(sentence_splitter=sent_tokenize)

        self.logger.info('Creating embedding index')
        _embedding_index: EmbeddingIndex = embedding_index.knn()

        self.logger.info('Creating embedding model')
        _embedding_model: EmbeddingModel = embedding_model.glove.load()

        self.logger.info('Initializing word weights')
        _word_weights: WordWeight = word_weight.idf_word_weight(_dataset)

        self.logger.info('Creating 5-gram sentence splitter')
        _sentence_splitter: SentenceSplitter = sentence_splitter.k_gram(5)

        self.logger.info('Creating text index')
        _text_index: TextIndex = TextIndex(_dataset, _embedding_model, _embedding_index,
                                           _sentence_splitter, _word_weights, self.logger)

        self.logger.info('Done')


if __name__ == '__main__':
    Interactor().interact()