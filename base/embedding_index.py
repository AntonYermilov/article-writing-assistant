from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import faiss
import nmsbind as nmslib


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

    def search_by_vector(self, vector: np.array, neighbours: int = 1) -> np.array:
        # noinspection PyArgumentList
        return self.index.search(vector.reshape(1, -1), neighbours)[1][0]


class NSW(EmbeddingIndex):
    def __init__(self):
        super().__init__()
        self.index = None

    def build(self, index_file: Path, dim: int):
        self.dim = dim
        self.index_file = index_file

        matrix = np.memmap(index_file, dtype=np.float32, mode='r+')
        matrix = matrix.reshape((-1, dim))

        self.index = nmslib.init(space='cosinesimil', method='sw-graph')
        nmslib.addDataPointBatch(self.index, np.arange(matrix.shape[0], dtype=np.int32), matrix)
        self.index.createIndex({}, print_progress=True)

    def search_by_matrix(self, matrix: np.array, neighbours: int = 1) -> np.array:
        return np.array([self.search_by_vector(vector, neighbours) for vector in matrix])

    def search_by_vector(self, vector: np.array, neighbours: int = 1) -> np.array:
        return nmslib.knnQuery(self.index, neighbours, vector)


class HNSW(EmbeddingIndex):
    def __init__(self):
        super().__init__()
        self.index = None

    def build(self, index_file: Path, dim: int):
        self.dim = dim
        self.index_file = index_file

        matrix = np.memmap(index_file, dtype=np.float32, mode='r+')
        matrix = matrix.reshape((-1, dim))

        self.index = nmslib.init(space='cosinesimil', method='hnsw')
        nmslib.addDataPointBatch(self.index, np.arange(matrix.shape[0], dtype=np.int32), matrix)
        self.index.createIndex({}, print_progress=True)

    def search_by_matrix(self, matrix: np.array, neighbours: int = 1) -> np.array:
        return np.array([self.search_by_vector(vector, neighbours) for vector in matrix])

    def search_by_vector(self, vector: np.array, neighbours: int = 1) -> np.array:
        return nmslib.knnQuery(self.index, neighbours, vector)
