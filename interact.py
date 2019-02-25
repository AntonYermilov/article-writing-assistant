import sys
import argparse
import numpy as np
from base import embeddings, datasets
from base.pretrained import PretrainedModel
from base.textutils import text2vec


def get_config():
    parser = argparse.ArgumentParser('Finding nearest sentences from the specified corpus of texts')
    parser.add_argument('--corpus', type=str, required=True, help='corpus of texts')
    parser.add_argument('--embedding', type=str, required=True, help='either a name of pretrained model' \
                                                                     'or a path to the existing .vec model')
    parser.add_argument('--neighbours', type=int, default=1, help='number of nearest sentences from corpus')
    return parser.parse_args()


if __name__ == '__main__':
    config = get_config()

    corpus_name = config.corpus.lower()
    if corpus_name not in datasets:
        raise ValueError(f'Corpus {corpus_name} does not exist!')

    embedding = config.embedding.lower()
    if embedding in embeddings:
        embedding = embeddings[embedding]
    else:
        embedding = PretrainedModel(embedding)

    neighbours = config.neighbours
    if neighbours < 1:
        raise ValueError(f'Invalid number of neighbours: {neighbours}')

    print('Loading dataset...', file=sys.stderr)
    corpus = datasets[corpus_name].load()

    print('Loading embedding model...', file=sys.stderr)
    model = embedding.load()

    print('Creating index...', file=sys.stderr)
    index = corpus.build_index(model)

    print('Done', file=sys.stderr)

    while True:
        sys.stdout.write('> ')
        sys.stdout.flush()

        text = sys.stdin.readline()
        sentences = list(filter(lambda s : not s.isspace(), text.split('.')))
        if len(sentences) == 0:
            continue

        query = np.array([text2vec(sentence, model, normalize=False) for sentence in sentences], dtype=np.float32)
        D, I = index.search(query, neighbours)
        for n, i in enumerate(I):
            for j in i:
               print(corpus.get_sentence(j))
            if n + 1 == len(I):
                print()
