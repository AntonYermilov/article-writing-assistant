import sys
import argparse
import time
from base import embeddings, datasets, searchers
from base.data_embedding_filter import DataEmbeddingFilter
from evaluation.metrics import bleu_on_corpus


def get_config():
    parser = argparse.ArgumentParser('Finding nearest sentences from the specified corpus of texts')
    parser.add_argument('--corpus', type=str, required=True, help='corpus of texts')
    parser.add_argument('--size', type=int, default=None, required=False, help='number of sentences')
    parser.add_argument('--embedding', type=str, required=True, help='either a name of pretrained model' \
                                                                     'or a path to the existing .vec model')
    parser.add_argument('--neighbours', type=int, default=1, help='number of nearest sentences from corpus')
    parser.add_argument('--searcher', type=str, default="Faiss", help='')
    return parser.parse_args()


def interact(data_filter):
    while True:
        sys.stdout.write('> ')
        sys.stdout.flush()

        text = sys.stdin.readline()
        sentences = list(filter(lambda s: not s.isspace(), text.split('.')))
        if len(sentences) == 0:
            continue

        for sentence in sentences:
            print(f"nearest for \"{sentence}\":")
            for nearest in data_filter.search(sentence, neighbours):
                print(nearest)
            print()


if __name__ == '__main__':
    config = get_config()
    start_time = time.time()

    corpus_name = config.corpus.lower()
    if corpus_name not in datasets:
        raise ValueError(f'Corpus {corpus_name} does not exist!')

    embedding = config.embedding.lower()
    if embedding in embeddings:
        embedding = embeddings[embedding]
    else:
        raise ValueError(f'Embedding {embedding} does not exist!')
        # embedding = GensimModel(embedding)

    searcher = config.searcher.lower()
    if searcher not in searchers:
        raise ValueError(f'Searcher {searcher} does not exist!')

    neighbours = config.neighbours
    if neighbours < 1:
        raise ValueError(f'Invalid number of neighbours: {neighbours}')
    size = config.size

    print('Loading dataset...', file=sys.stderr, end='')
    corpus = datasets[corpus_name]
    corpus.load(size)
    print(f" {(time.time() - start_time):.2f} seconds", file=sys.stderr)
    start_time = time.time()

    print('Loading embedding model...', file=sys.stderr, end='')
    embedding.load()
    print(f" {(time.time() - start_time):.2f} seconds", file=sys.stderr)
    start_time = time.time()

    nearest_searcher = searchers[searcher](embedding.dim())

    print('Creating index...', file=sys.stderr, end='')
    data_filter = DataEmbeddingFilter(corpus, embedding, nearest_searcher)
    data_filter.build_index()
    print(f" {(time.time() - start_time):.2f} seconds", file=sys.stderr)
    start_time = time.time()

    print('Done', file=sys.stderr)


    print(f"Bleu on corpus: {bleu_on_corpus(corpus.get(), data_filter)}")
    print(f" {(time.time() - start_time):.2f} seconds", file=sys.stderr)

    interact(data_filter)
