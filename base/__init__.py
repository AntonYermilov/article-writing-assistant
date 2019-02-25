from .utils import AttributeDict
from .pretrained import *
from .dataset import NIPSPapersDataset


embeddings = AttributeDict()
embeddings.fasttext_en_300 = PretrainedFastTextModel('fasttext.en.300.vec', 'cc.en.300.vec.gz')

datasets = AttributeDict()
datasets.nips_papers = NIPSPapersDataset('nips-papers.csv.gz')
