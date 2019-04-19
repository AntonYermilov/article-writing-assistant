from typing import List, Dict
import numpy as np
from base.embedding import Embedding


def sentence2vec(sentence: List[str], embedder: Embedding, normalize=False):
    sentence = [embedder.word_embedding(word) for word in sentence]
    vec = sum(sentence[sentence != np.array(None)])
    if normalize:
        vec /= np.linalg.norm(vec)
    return vec


def sentence2vec_idf(sentence: List[str], embedder: Embedding, idf: Dict[str, float], normalize=False):
    # TODO word from sentences could not be found in dataset
    idf = np.array([idf[word.lower()] for word in sentence])
    sentence = np.array([embedder.word_embedding(word) for word in sentence])
    vec = sum(idf[sentence != np.array(None)] * sentence[sentence != np.array(None)])
    if normalize:
        vec /= np.linalg.norm(vec)
    return vec
