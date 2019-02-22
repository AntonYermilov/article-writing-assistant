# article-writing-assistant

## Installation

Python 3.7+ is required.

```
git clone https://github.com/AntonYermilov/article-writing-assistant.git
cd article-writing-assistant
cat requirements.txt | xargs -n 1 -L 1 pip3 install
```

## Usage

**TBD**

In order to fit your own embedding model you may run
```
python3 fit.py --model {model} --name {model_name} --corpus {corpus} --params '{params in json format}'
```

Here `model` corresponds to the name of the algorithm to fit embeddings, `model_name` is a name of a model to be saved,
and `corpus` is a name of corpus of texts to be used for fitting embeddings.

As soon as model is fitted it would be saved in _.vec_ format under the specified name.

## Examples

**TBD**

## Models

Currently we support the following models:

* glove

## Datasets

Currently we support the following corpuses of texts:

* nips-papers

## License
[MIT](LICENCE)
