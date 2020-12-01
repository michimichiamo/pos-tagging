## pos-tagging
Training three different RNN models on a portion of Penn Treebank data to perform POS-tagging

This repository contains a **Python Notebook** project I developed for the Natural Language Processing exam in the AI Master Course I'm attending.
The aim of the project is performing **POS-tagging** (i.e. *sequence labelling*) on a portion of the Penn Treebank data, made public by [nltk.org](https://www.nltk.org/) and available on [this page](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip).
I performed this task through the help of **GloVe embeddings**, making use of the [Gensim](https://radimrehurek.com/gensim/) library to download the model.

## Outline

This notebook provides a `keras` implementation of **three RNN models** with the purpose of performing sequence labelling, in particular **POS-tagging**.

The main outline of the process is the following:
- Raw data (199 labelled documents consisting of a varying number of sentences) is split into **train**, **validation** and **test** partitions and stored in a `pandas.DataFrame` object.
- The words in the dataframe - after some preprocessing - are substituted with their *Glove embeddings* (and with custom embeddings, when needed) and their labels (POS tags) are one-hot encoded.
- Models have the **embeddings** as input and **tags** as output.
- To simulate a real-world scenario, the different models are trained on the first partition and evaluated on the second one; lastly, only the best-performing model is tested on the last partition.
Moreover, the partitions are kept independent with respect to every aspect except the word embeddings (either retrieved from GloVe or computed) and the labels encoding.
