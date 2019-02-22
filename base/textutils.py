import re
import numpy as np
from gensim.models import KeyedVectors


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'([0123456789\-\+\=\*\<\>\;\:\|\n])', r' ', text)
    text = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    return text


def text2vec(text: str, model: KeyedVectors, normalize: bool = False) -> np.array:
    dim = model.vector_size
    vec = np.zeros(dim, dtype=np.float32)
    for word in text.split():
        if word in model:
            vec += model[word]
    if normalize:
        vec /= np.linalg.norm(vec)
    return vec
