from abc import ABC, abstractmethod
from typing import List

import numpy as np
import faiss


class NearestSearcher(ABC):
    @abstractmethod
    def add(self, vectors):
        pass

    @abstractmethod
    def search(self, vector, k=1) -> list:
        pass


class Faiss(NearestSearcher):
    def __init__(self, dim):
        # self.index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 16)

    def add(self, vectors):
        # self.index.add(vectors)
        sz = len(vectors)
        self.index.train(vectors[:sz // 2])
        self.index.hnsw.efConstruction = 40
        self.index.add(vectors[sz // 2:])
        self.index.hnsw.efSearch = 32

    def search(self, vector, k=1) -> List[int]:
        return self.index.search(vector, k)[1][0]


class KNN(NearestSearcher):
    def __init__(self, dim):
        self.dim = dim
        self.vectors = None
        self.norm = np.linalg.norm

    def add(self, vectors):
        if self.vectors is None:
            self.vectors = np.array(vectors)
        else:
            self.vectors = np.vstack((self.vectors, vectors))

    def search(self, vector, k=1) -> List[int]:
        dists = [(self.norm(vector - v), i) for i, v in enumerate(self.vectors)]
        return list(map(lambda x: x[1], sorted(dists)[:k]))
