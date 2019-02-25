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

You may use `interact.py` to interact with _assistant_. This program shows you the sentences from the specified corpus
of texts, which are the nearest to the sentences you write to the console.

* `--corpus CORPUS` – the name of the corpus of texts to be used for finding similar sentences
* `--embedding EMBEDDING_MODEL` – either a name of pretrained embedding model, or a path to the existing model in _.vec_ format
* `--neighbours` – number of most similar sentences to be printed

## Examples

#### Fitting embedding model

```
$ python3 fit.py --model glove --name glove.nips.50.vec --corpus nips_papers --params '{"dim": 50, "epochs": 30, "step": 0.05, "verbose": true}'
[main] INFO com.expleague.ml.embedding.impl.EmbeddingBuilderBase - ==== Dictionary phase ====
[main] INFO com.expleague.ml.embedding.impl.EmbeddingBuilderBase - ==== 11s ====
[main] INFO com.expleague.ml.embedding.impl.EmbeddingBuilderBase - ==== Cooccurrences phase ====
[main] INFO com.expleague.ml.embedding.impl.EmbeddingBuilderBase - ==== 273s ====
[main] INFO com.expleague.ml.embedding.impl.EmbeddingBuilderBase - ==== Training phase ====
Iteration 0, score: 0.07275579261398023, count: 49915615: 22039(ms)
Iteration 1, score: 0.06419253349343526, count: 49915615: 19923(ms)
Iteration 2, score: 0.05773053618863979, count: 49915615: 20347(ms)
...
Iteration 29, score: 0.04445610218196396, count: 49915615: 20112(ms)
[main] INFO com.expleague.ml.embedding.impl.EmbeddingBuilderBase - ==== 629s ====

$ ls resources/models
... glove.nips.50.vec ...
```

#### Interaction

```
# The following results of the interaction are possible but not required
$ python3 interact.py --corpus nips_papers --embedding resources/models/glove.nips.50.vec --neighbours 2
Loading dataset...
Loading embedding model...
Creating index...
Done
> In this paper, we introduce a new approach to classification problem
We consider a bayesian approach to the problem
Here, we propose a principled probabilistic approach to this problem
> Our method is based on classical multinomial logistic regression boosting
We compared the boosting algorithms with multinomial naive bayes
Our third example is multinomial logistic regression for multiclass classification
```

## Models

Currently we support the following models:

* `glove`

## Datasets

Currently we support the following corpuses of texts:

* `nips-papers`

## Embeddings

Currently we support the following pretrained embeddings:

* `fasttext_en`

**TBD**

## License
[MIT](LICENCE)
