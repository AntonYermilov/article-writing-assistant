from typing import List
import numpy as np
from base.dataset import RawDataset
from base.embedder import Embedder
from base.nearest_search import NearestSearcher


class DataEmbeddingFilter:
    def __init__(self, dataset: RawDataset, embedder: Embedder, searcher: NearestSearcher):
        self.dataset = dataset
        self.embedder = embedder
        self.searcher = searcher

    def build_index(self):
        dim = self.embedder.dim()
        sentences = self.dataset.get()
        print("dataset size", len(sentences))
        dataset_size = len(sentences)
        dataset = np.zeros((dataset_size, dim), dtype=np.float32)
        print("make vectors")
        for i, sentence in enumerate(sentences):
            dataset[i] = self.text2vec(sentence, normalize=False)
        print("create index")
        print("adding to index")
        self.searcher.add(dataset)

    def text2vec(self, text: str, normalize: bool = False) -> np.array:
        dim = self.embedder.dim()
        vec = np.zeros(dim, dtype=np.float32)
        words = text.split()
        for i, word in enumerate(words):
            embedding = self.embedder.embedding(words, i)
            if embedding is not None:
                vec += embedding
        if normalize:
            vec /= np.linalg.norm(vec)
        return vec

    def search(self, sentence: str, neighbours: int=1) -> List[str]:
        query = np.array([self.text2vec(sentence, normalize=False)], dtype=np.float32)
        inds = self.searcher.search(query, neighbours)
        return [self.dataset.sentences[ind] for ind in inds]
