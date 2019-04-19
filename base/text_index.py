import numpy as np
from .dataset import Dataset
from typing import List, Callable
from base.embedding import Embedding
from base.embedding_index import EmbeddingIndex
from tools.s2v import sentence2vec_idf


class DataEmbeddingFilter:
    def __init__(self, dataset: Dataset, embedding: Embedding, index: EmbeddingIndex, sentence_splitter: Callable,
                 piece_splitter: Callable):
        self.dataset = dataset
        self.index = index
        self.embedding = embedding

    def build(self):
        dim = self.embedding.dim()
        sentences = self.dataset.get()
        print("dataset size", len(sentences))
        dataset_size = len(sentences)
        dataset = np.zeros((dataset_size, dim), dtype=np.float32)
        print("make vectors")
        for i, sentence in enumerate(sentences):
            # dataset[i] = self.text2vec(sentence)
            dataset[i] = sentence2vec_idf(sentence, self.embedding)
        print("create index")
        print("adding to index")
        self.index.build(dataset)

    def search(self, sentence: str, neighbours: int = 1) -> List[str]:
        query = np.array([self.text2vec(sentence)], dtype=np.float32)
        inds = self.searcher.search(query, neighbours)
        return [self.dataset.sentences[ind] for ind in inds]
