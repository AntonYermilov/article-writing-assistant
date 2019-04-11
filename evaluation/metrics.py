from typing import List, Union
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import time

from base.data_embedding_filter import DataEmbeddingFilter


def bleu(sent1: str, sent2: str) -> Union[float, type(None)]:
    sent1 = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), sent1.split(' '))))
    sent2 = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), sent2.split(' '))))
    return sentence_bleu([sent1], sent2, weights=(0.25, 0.25, 0.25, 0.25))


def bleu_on_corpus(sentences: List[str], data_filter: DataEmbeddingFilter, ratio=0.05) -> float:
    np.random.shuffle(sentences)
    samples = sentences[:int(ratio * len(sentences))]
    S = 0
    search, evaluate = 0, 0
    for i, sentence in enumerate(samples):
        if i > 0 and i % 100 == 0:
            print(f'search={search:.5f}, evaluate={evaluate:.5f}')
        t = time.time()
        q = data_filter.search(sentence, 2)[1]
        search += time.time() - t
        t = time.time()
        S += bleu(sentence, q)
        evaluate += time.time() - t
    return S / len(samples)
