import re
import numpy as np
from typing import List
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import wordnet


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'([0123456789\-\+\=\*\<\>\;\:\|\n])', r' ', text)
    text = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    return text


def remove_string_special_characters(text: str) -> str:
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('_', '', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()


def tokenize_to_words(text: str) -> List[str]:
    return word_tokenize(text, language='english')


def tokenize_to_sentences(text: str) -> np.array:
    text = normalize_text(text)
    return np.array(map(str.strip, re.split(r'[ .?!]+', text)))


def spoil_text(text: str, modify_articles_rate=0.5, modify_prepositions_rate=0.25,
               modify_synonyms_rate=0.2) -> str:
    """
    Receives normalized text.
    Randomly odifies articles (removes or changes them).
    :param text: text in normalized form
    :param modify_articles_rate: probability with which articles are modified
    :param modify_prepositions_rate: probability with which prepositions are modified
    :param modify_synonyms_rate: probability with which ordinary words are changed to the synonyms
    :return: spoiled text
    """
    tokens = text.split(' ')
    tokens = list(filter(lambda token: len(token) > 0 and not token.isspace(), tokens))

    articles = ['a', 'an', 'the', '']
    prepositions = ['on', 'in', 'into', 'at']
    for i, token in enumerate(tokens):
        if  token in articles:
            if np.random.binomial(1, modify_articles_rate) == 1:
                tokens[i] = np.random.choice(articles)
        elif token in prepositions:
            if np.random.binomial(1, modify_prepositions_rate) == 1:
                tokens[i] = np.random.choice(prepositions)
        elif np.random.binomial(1, modify_synonyms_rate) == 1:
            synonyms = [l.name() for syn in wordnet.synsets(token)[:1] for l in syn.lemmas()]
            if len(synonyms) > 0:
                syn = np.random.choice(synonyms)
                tokens[i] = syn.replace('_', ' ')
    return ' '.join(tokens)
