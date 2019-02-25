import argparse
from embedding import models
from base import datasets
import json


def get_config():
    parser = argparse.ArgumentParser('Fit embeddings on the specified corpus of texts')
    parser.add_argument('--model', type=str, required=True, help='model to fit')
    parser.add_argument('--name', type=str, required=True, help='name of the model to save')
    parser.add_argument('--corpus', type=str, required=True, help='corpus of texts')
    parser.add_argument('--params', type=str, default='', help='parameters of model in json format')
    return parser.parse_args()


if __name__ == '__main__':
    config = get_config()

    model = config.model.lower()
    if model not in models:
        raise ValueError(f'Model {model} does not exist!')

    corpus_name = config.corpus.lower()
    if corpus_name not in datasets:
        raise ValueError(f'Corpus {corpus_name} does not exist!')

    model_name = config.name
    params = json.loads(config.params)

    model = models[model](corpus_name, model_name, **params)
    model.fit()
