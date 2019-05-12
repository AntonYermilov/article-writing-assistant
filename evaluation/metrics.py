import warnings
warnings.filterwarnings('ignore')
from typing import List, Union
from nltk.translate.bleu_score import sentence_bleu

from base.sentence import Sentence
from base.text_index import TextIndex


def bleu(sent1: str, sent2: str) -> Union[float, type(None)]:
    sent1 = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), sent1.split(' '))))
    sent2 = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), sent2.split(' '))))
    return sentence_bleu([sent1], sent2, weights=(0.25, 0.25, 0.25, 0.25))


def bleu_on_corpus(sentences: List[Sentence], text_index: TextIndex) -> float:
    cumulative_bleu = 0.
    count = 0.
    none_q = 0
    k = 10
    for i, sentence in enumerate(sentences):
        q = text_index.search(sentence, neighbours=k, splitter_neighbours=10)
        if q is None:
            none_q += 1
            continue
        ind = 0
        while ind < k and str(q[ind]) == str(sentence):
            ind += 1
        if ind < k:
            cumulative_bleu += bleu(str(sentence), str(q[ind]))
            count += 1
    print(f"Count: {count}, q is None = {none_q}, len = {len(sentences)}")
    return cumulative_bleu / count
