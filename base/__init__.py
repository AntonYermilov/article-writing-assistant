from .utils import AttributeDict
from .model import FastTextModel
from .dataset import NIPSPapersDataset


models = AttributeDict()
models.fasttext_en = FastTextModel('cc.en.300.vec', 'cc.en.300.vec.gz')

datasets = AttributeDict()
datasets.nips_papers = NIPSPapersDataset('nips-papers.csv')
