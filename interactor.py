import sys
import logging
from datetime import datetime
from pathlib import Path

from nltk.tokenize import sent_tokenize

from base import dataset, embedding_index, embedding_model, word_weight, sentence_splitter
from base.dataset import Dataset
from base.document import Document
from base.embedding_index import EmbeddingIndex
from base.embedding_model import EmbeddingModel
from base.word_weight import WordWeight
from base.sentence_splitter import SentenceSplitter
from base.text_index import TextIndex
from evaluation.metrics import bleu_on_corpus


class Interactor:
    DEFAULT_LOG_FOLDER = Path('logs')

    def __init__(self, _dataset: Dataset, _embedding_model: EmbeddingModel, _embedding_index: EmbeddingIndex,
                 _sentence_splitter: SentenceSplitter, _word_weights: WordWeight, _documents_limit: int,
                 _text_index_bin: str):
        self.dataset: Dataset = _dataset
        self.embedding_model: EmbeddingModel = _embedding_model
        self.embedding_index: EmbeddingIndex = _embedding_index
        self.sentence_splitter: SentenceSplitter = _sentence_splitter
        self.word_weights: WordWeight = _word_weights
        self.documents_limit: int = _documents_limit
        self.text_index_bin: str = _text_index_bin

        self.text_index: TextIndex = None

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

    def _initialize(self):
        self.logger.info('Loading dataset')
        self.dataset.load(sentence_splitter=sent_tokenize, documents_limit=self.documents_limit)

        self.logger.info('Loading embedding model')
        self.embedding_model.load()

        self.logger.info('Initializing word weights')
        self.word_weights.initialize(self.dataset)

        self.logger.info('Creating text index')
        self.text_index = TextIndex(self.dataset, self.embedding_model, self.embedding_index,
                                    self.sentence_splitter, self.word_weights, self.logger)
        self.text_index.build(self.text_index_bin)

        self.logger.info('Initialization completed successfully')

    def _process_input(self, text: str):
        document = Document(text)
        sentences = document.split_to_sentences(sent_tokenize)
        for sentence in sentences:
            response = self.text_index.search(sentence, neighbours=5, splitter_neighbours=10)
            if response is None:
                continue
            for r in response:
                sys.stdout.write(f'{str(sentence)}   ->   {str(r)}\n')
                sys.stdout.flush()

    def interact(self):
        self._initialize()
        print(f"Bleu on corpus: {bleu_on_corpus(self.text_index.dataset.get_sentences(), self.text_index)}")
        while True:
            sys.stdout.write('> ')
            sys.stdout.flush()

            text = sys.stdin.readline().strip()
            if len(text) == 0:
                continue

            self._process_input(text)


if __name__ == '__main__':
    Interactor(
        _dataset=dataset.nips_papers,
        _embedding_model=embedding_model.glove128,
        # _embedding_index=embedding_index.knn,
        # _embedding_index=embedding_index.faiss,
        _embedding_index=embedding_index.nsw,
        _sentence_splitter=sentence_splitter.five_gram,
        _word_weights=word_weight.idf_word_weight,
        _documents_limit=100,
        _text_index_bin='nips_100doc_glove128_v1.bin'
    ).interact()
