from .utils import AttributeDict
from .pretrained import *
from .dataset import NIPSPapersDataset


models = AttributeDict()
models.glove_nips = PretrainedModel('glove.nips.vec')
models.fasttext_en = PretrainedFastTextModel('cc.en.300.vec', 'cc.en.300.vec.gz')

datasets = AttributeDict()
datasets.nips_papers = NIPSPapersDataset('nips-papers.csv.gz')
