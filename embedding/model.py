from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, corpus_name: str, model_name: str):
        self.corpus_name = corpus_name
        self.model_name = model_name + '.vec'

    @abstractmethod
    def fit(self):
        pass
