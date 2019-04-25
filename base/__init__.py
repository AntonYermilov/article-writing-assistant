from tools import AttributeDict
from .embedding_index import Faiss, KNN, FaissHNSW
from .embedding_model import GensimModel, FastText
from .dataset import NIPSPapersDataset
from .word_weight import StandardWordWeight, IDFWordWeight
from .sentence_splitter import KGramSplitter



dataset = AttributeDict()
dataset.nips_papers = NIPSPapersDataset('nips-papers.csv.gz')

embedding_index = AttributeDict()
embedding_index.faiss = Faiss()
embedding_index.faiss_hnsw = FaissHNSW()
embedding_index.knn = KNN()

embedding_model = AttributeDict()
embedding_model.glove50 = GensimModel('glove.nips.50.vec')
embedding_model.glove128 = GensimModel('glove.nips.50.vec')
embedding_model.fasttext_en_300 = FastText('fasttext.en.300.vec', 'cc.en.300.vec.gz')

word_weight = AttributeDict()
word_weight.std_word_weight = StandardWordWeight()
word_weight.idf_word_weight = IDFWordWeight()

sentence_splitter = AttributeDict()
sentence_splitter.five_gram = KGramSplitter(5)

"""
embeddings.elmo = Elmo("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                       "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                       "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                       "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json")
"""