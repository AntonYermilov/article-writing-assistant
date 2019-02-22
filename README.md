# article-writing-assistant

## Installation

Python 3.7+ is required.

```
git clone https://github.com/AntonYermilov/article-writing-assistant.git
cd article-writing-assistant
cat requirements.txt | xargs -n 1 -L 1 pip3 install
```

## Usage

### Fitting embeddings

You may use `fit.py` in order to fit your own embedding model.

Currently we support the following options:

* `--model MODEL` – the name of the algorithm to fit embeddings
* `--name MODEL_NAME` – the name of the model to be saved in _.vec_ format
* `--corpus CORPUS` – the name of the corpus of texts to be used for fitting embeddings
* `--params PARAMS` – parameters of the algorithm in JSON format

### Interaction

**TBD**

## Examples

**TBD**

## Models

Currently we support the following models:

* glove

## Datasets

Currently we support the following corpuses of texts:

* nips-papers

## Embeddings

Currently we support the following pretrained embeddings:

* `fasttext_en`

**TBD**

## License
[MIT](LICENCE)
