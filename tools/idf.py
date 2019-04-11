from nltk.tokenize import word_tokenize
import re
from typing import List
from numpy import log


def remove_string_special_characters(text: str) -> str:
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('_', '', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()


def idf(docs: List[str]):
    _idf = {}
    for doc in docs:
        doc = remove_string_special_characters(doc)
        words = set(filter(lambda word: word.isalnum(), word_tokenize(doc, language='english')))
        for word in words:
            word = word.lower()
            if word not in _idf:
                _idf[word] = 0
            _idf[word] += 1
    for key, value in _idf.items():
        _idf[key] = log(len(docs) / value)
    return _idf
