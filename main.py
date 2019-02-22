import sys
import numpy as np
from base import models, datasets
from base.textutils import text2vec


def run():
    print('Loading dataset...', file=sys.stderr)
    dataset = datasets.nips_papers.load()
    assert dataset is not None
    print('Loading model...', file=sys.stderr)
    model = models.glove_nips.load()
    assert model is not None
    print('Creating index...', file=sys.stderr)
    index = dataset.apply_model(model)
    assert index is not None
    print('Done', file=sys.stderr)

    neighbours = 5
    while True:
        text = sys.stdin.readline()
        query = np.array([text2vec(sentence, model, normalize=False) for sentence in text.split('.')], dtype=np.float32)
        D, I = index.search(query, neighbours)
        for i in I:
            for j in i:
               print(dataset.get_sentence(j))
            print()


if __name__ == '__main__':
    run()
