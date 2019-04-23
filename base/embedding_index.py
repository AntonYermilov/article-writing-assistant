from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import faiss


class EmbeddingIndex(ABC):
    def __init__(self):
        self.dim = None
        self.index_file = None

    @abstractmethod
    def build(self, index_file: Path, dim: int):
        pass

    @abstractmethod
    def search_by_matrix(self, matrix: np.array, neighbours: int = 1) -> np.array:
        pass

    @abstractmethod
    def search_by_vector(self, vector: np.array, neighbours: int = 1) -> np.array:
        pass


class KNN(EmbeddingIndex):
    def __init__(self):
        super().__init__()

    def build(self, index_file: Path, dim: int):
        self.dim = dim
        self.index_file = index_file

    def search_by_matrix(self, matrix: np.array, neighbours: int = 1) -> np.array:
        return np.array([self.search_by_vector(vector, neighbours) for vector in matrix])

    def search_by_vector(self, vector: np.array, neighbours: int = 1) -> np.array:
        index = np.memmap(self.index_file, dtype=np.float32, mode='r')
        index.reshape((-1, self.dim))
        dists = np.linalg.norm(index - vector, axis=1)
        return np.argsort(dists)[:neighbours]


class Faiss(EmbeddingIndex):
    def __init__(self):
        super().__init__()
        self.index = None

    def build(self, index_file: Path, dim: int):
        self.dim = dim
        self.index_file = index_file

        matrix = np.memmap(index_file, dtype=np.float32, mode='r+')
        matrix = matrix.reshape((-1, dim))

        self.index = faiss.IndexFlatL2(self.dim)
        # noinspection PyArgumentList
        self.index.add(matrix)

    def search_by_matrix(self, matrix: np.array, neighbours: int = 1) -> np.array:
        # noinspection PyArgumentList
        return self.index.search(matrix, neighbours)[1]

    def search_by_vector(self, vector: np.array, neighbours: int = 1) -> np.int32:
        # noinspection PyArgumentList
        return self.index.search(vector.reshape(1, -1), neighbours)[1][0]


class FaissHNSW(EmbeddingIndex):
    def __init__(self):
        super().__init__()
        self.index = None

    def build(self, index_file: Path, dim: int):
        self.dim = dim
        self.index_file = index_file

        matrix = np.memmap(index_file, dtype=np.float32, mode='r+')
        matrix = matrix.reshape((-1, dim))

        self.index = faiss.IndexHNSWSQ(self.dim, faiss.ScalarQuantizer.QT_8bit, 16)
        # noinspection PyArgumentList
        self.index.train(matrix)
        self.index.hnsw.efConstruction = 40
        # noinspection PyArgumentList
        self.index.add(matrix)
        self.index.hnsw.efSearch = 32

    def search_by_matrix(self, matrix: np.array, neighbours: int = 1) -> np.array:
        # noinspection PyArgumentList
        return self.index.search(matrix, neighbours)[1]

    def search_by_vector(self, vector: np.array, neighbours: int = 1) -> np.array:
        # noinspection PyArgumentList
        return self.index.search(vector.reshape(1, -1), neighbours)[1][0]
