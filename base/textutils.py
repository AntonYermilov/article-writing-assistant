import re
import numpy as np
from gensim.models import KeyedVectors


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,\+\*])', r' \1 ', text)
    text = re.sub(r'([\;\:\|\n])', ' ', text)
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('0', ' zero ')
    text = text.replace('1', ' one ')
    text = text.replace('2', ' two ')
    text = text.replace('3', ' three ')
    text = text.replace('4', ' four ')
    text = text.replace('5', ' five ')
    text = text.replace('6', ' six ')
    text = text.replace('7', ' seven ')
    text = text.replace('8', ' eight ')
    text = text.replace('9', ' nine ')
    return text


def text2vec(text: str, model: KeyedVectors) -> np.array:
    dim = model.vector_size
    vec = np.zeros(dim, dtype=np.float32)
    for word in text.split():
        if word in model:
            vec += model[word]
    vec /= np.linalg.norm(vec)
    return vec
