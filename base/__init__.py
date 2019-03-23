from .utils import AttributeDict
from .embedder import *
from .dataset import NIPSPapersDataset


embeddings = AttributeDict()
# embeddings.gensim = FastText('fasttext.en.300.vec', 'cc.en.300.vec.gz')
embeddings.fasttext_en_300 = FastText('fasttext.en.300.vec', 'cc.en.300.vec.gz')
embeddings.elmo = Elmo("~/Downloads/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                       "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                       "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json")

datasets = AttributeDict()
datasets.nips_papers = NIPSPapersDataset('nips-papers.csv.gz')
