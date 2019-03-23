from abc import ABC, abstractmethod

import numpy as np
from faiss import IndexFlatL2


class NearestSearcher(ABC):
    @abstractmethod
    def add(self, vectors):
        pass

    @abstractmethod
    def search(self, vector, k=1) -> list:
        pass


class Faiss(NearestSearcher):
    def __init__(self, dim):
        self.index = IndexFlatL2(dim)

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, vector, k=1) -> list:
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

    def search(self, vector, k=1) -> list:
        dists = [(self.norm(vector - v), i) for i, v in enumerate(self.vectors)]
        return list(map(lambda x: x[1], sorted(dists)[:k]))
