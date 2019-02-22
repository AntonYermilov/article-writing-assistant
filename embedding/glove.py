from base import datasets
from pathlib import Path
import subprocess


MODEL_FOLDER = Path('resources', 'models').absolute()
JAR_FOLDER = Path('resources', 'jar').absolute()
TEMP_FOLDER = Path('tmp').absolute()


class Glove:
    def __init__(self,
                 corpus_name: str,
                 model_name: str,
                 dim: int = 50,
                 min_freq: int = 5,
                 epochs: int = 25,
                 step: float = 0.1,
                 window_left: int = 15,
                 window_right: int = 15,
                 verbose: bool = False):
        self.corpus_name = corpus_name
        self.model_name = model_name
        self.dim = dim
        self.min_freq = min_freq
        self.epochs = epochs
        self.step = step
        self.window_left = window_left
        self.window_right = window_right
        self.verbose = verbose
        self.jar_path = JAR_FOLDER / 'glove.jar'

    def fit(self):
        if not TEMP_FOLDER.exists():
            TEMP_FOLDER.mkdir()

        if not self.corpus_name in datasets.keys():
            raise RuntimeError(f'Corpus {self.corpus_name} is not supported!')

        dataset_path = TEMP_FOLDER / 'dataset.txt'
        temp_model_path = TEMP_FOLDER / 'model.ss_decomp'

        dataset = datasets[self.corpus_name].load()
        dataset.save(dataset_path)

        out = subprocess.DEVNULL if not self.verbose else None
        params = ['java', '-Xmx8g', '-jar', str(self.jar_path),
                  '--corpus_path', str(dataset_path),
                  '--model_path', str(temp_model_path),
                  '--dim', str(self.dim),
                  '--min_freq', str(self.min_freq),
                  '--epochs', str(self.epochs),
                  '--step', str(self.step),
                  '--window_left', str(self.window_left),
                  '--window_right', str(self.window_right)]
        subprocess.run(params, stdout=out)

        if not MODEL_FOLDER.exists():
            MODEL_FOLDER.mkdir()
        model_path = MODEL_FOLDER / self.model_name
        count_words = 0
        with model_path.open('w') as dst:
            with temp_model_path.open('r') as src:
                for line in src.readlines():
                    count_words += 1
                    tokens = line.split()
                    dst.write(' '.join([tokens[0]] + tokens[2:-1]) + '\n')
            dst.seek(0)
            dst.write(f'{count_words} {self.dim}\n')

        for file in TEMP_FOLDER.iterdir():
            file.unlink()
