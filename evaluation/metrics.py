from typing import List, Union
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from base.data_embedding_filter import DataEmbeddingFilter


def bleu(sent1: str, sent2: str) -> Union[float, type(None)]:
    sent1 = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), sent1.split(' '))))
    sent2 = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), sent2.split(' '))))
    return sentence_bleu([sent1], sent2, weights=(1, 0, 0, 0))


def bleu_on_corpus(sentences: List[str], data_filter: DataEmbeddingFilter) -> float:
    return float(np.mean([bleu(sentence, data_filter.search(sentence, 2)[1]) for sentence in sentences]))
