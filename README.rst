===============
Wordbatch 1.4.9
===============

Overview
========

Python library for distributed AI processing pipelines, using swappable scheduler backends.

Wordbatch parallelizes task pipelines as minibatches processed by a chosen scheduler backend. This allows
the user to develop AI programs on a local workstation or laptop, and scale the same
solution on a cluster or the cloud, simply by changing the pipeline backend to a distributed scheduler such as Spark,
Dask or Ray. A backend can be chosen based on performance characteristics on a particular task, and swapped for
different situations. For example, an AI model can be trained using a distributed backend, and then debugged or
deployed using a single serial process.

The library is organized around the orchestrator class Batcher, and Sklearn-compatible components,
split into Pipelines, Transformers, Extractors and Models. These extend the Scikit-learn API with a
fit_partial()-method, that enables transformers and models to be used in a streaming fashion.
The current set of components has been developed mostly for text processing tasks, but components for other domains
can be developed based on the available classes.

Requirements
============
Linux / Windows / macOS. Python 3.6 / 3.7

Installation
============
pip install wordbatch

macOS: compile using GCC-7 (https://github.com/anttttti/Wordbatch/issues/1)

linux: make sure GCC and its required libraries are installed before installing Wordbatch
| sudo apt install gcc
| sudo apt-get update
| sudo apt-get install --reinstall build-essential

Getting started
===============

| from wordbatch.models import FTRL
| from wordbatch.extractors import WordBag
| from wordbatch.pipelines import WordBatch
| from wordbatch.batcher import Batcher
|
| wb= WordBatch(extractor=WordBag(hash_ngrams=0, norm= 'l2', tf= 'binary', idf= 50.0),
|               batcher=Batcher(backend="multiprocessing"))
|
| clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=1)
|
| train_texts= ["Cut down a tree with a herring? It can't be done.",
|              "Don't say that word.",
|              "How can we not say the word if you don't tell us what it is?"]
| train_labels= [1, 0, 1]
| test_texts= ["Wait! I said it! I said it! Ooh! I said it again!"]
|
| clf.fit(wb.fit_transform(train_texts), train_labels)
| print(clf.predict(wb.transform(test_texts)))
|
| import ray
| ray.init()
| wb.batcher.backend= "ray"
| wb.batcher.backend_handle= ray
|
| clf.fit(wb.fit_transform(train_texts), train_labels)
| print(clf.predict(wb.transform(test_texts)))


Components
==========

Batcher
-------
Batcher orchestrates MapReduce processing of tasks using a backend, by splitting input data into separately processed
minibatches. Currently three local (serial, multiprocessing, Loky) and three distributed backends (Spark, Dask,
Ray) are supported. Some distributed backends will process the tasks concurrently as a graph of lazily evaluated
futures, with Batcher dynamically sending the graph for the backend to process. All three supported distributed
backends allow real time monitoring of the processing pipeline using the backend's own GUI.


Pipelines
---------
Pipelines are classes that send functions, methods and classes to Batcher for processing. Unlike other components in
Wordbatch, pipelines contain a reference to Batcher, and are never referenced themselves in the calls sent to Batcher.
This prevents trying to serialize and send the backend handle itself. The simplest pipeline is Apply,
which processes a function or method over the input data row-by-row. WordBatch is a full complex pipeline for text
processing, with optional steps such as text normalization, spelling correction, stemming, feature extraction, and
LZ4-caching of results.


Transformers
------------
Transformers are transformer classes extending the Scikit-learn API, by accepting a Batcher instance as argument
of fit and transform methods. Transformers won't store Batcher references, allowing the transformer objects to be sent
to distributed workers. This allows transformers to do MapReduce operations as part of its methods, for example
gathering a dictionary of words from data when fitting a Dictionary. The current set of transformers are
text-specific classes, such as Dictionary, Tokenizer and TextNormalizer.


Extractors
----------
Extractors are transformer classes which don't directly call Batcher. Since extractors can't call Batcher directly,
they are mostly immutable and used for their transform() method calls distributed using a pipeline. The current set of
extractors is Cython-optimized, and aside from PandasHash intended for text feature extraction. These are:

- WordHash is wrapper for Scikit-learn HashingVectorizer, extended with option for LZ4-caching
- WordBag is a flexible alternative to Wordhash, with options such as IDF and per n-gram order weighting
- WordSeq provides sequences of word integers, as used by deep learning language models
- WordVec embeds words into word vector representations
- PandasHash extracts hashed features from a Pandas DataFrame, similar to VowpalWabbit's feature extraction


Models
------
Models are predictive models such as classifiers. Similar to extractors, they don't directly call Batcher, but are
Scikit-learn compatible and distributed using a pipeline if needed. Currently four
OpenMP-multithreaded L1&L2-regularized online learning models are provided, for single-label regression and
classification:

- FTRL : Linear model Proximal-FTRL that has become the most popular algorithm for online learning of linear models in Kaggle competions. The Cython-optimized implementation should be the fastest available version of FTRL.
- FM_FTRL : Factorization Machines. Linear effects estimated with FTRL and factor effects estimated with adaptive SGD. Prediction and estimation multithreaded across factors.
- NN_Relu_H1 : Neural Network with 1 hidden layer and Rectified Linear Unit activations, estimated with adaptive SGD. Prediction and estimation multithreaded across hidden layer.
- NN_Relu_H2: Neural Network with 2 hidden layers and Rectified Linear Unit activations, estimated with adaptive SGD. Prediction multithreaded across 2nd hidden layer, estimation across 1st hidden layer outputs.

The adaptive SGD optimizer works like Adagrad, but pools the adaptive learning rates across hidden nodes using the same
feature. This makes learning more robust and requires less memory. FM_FTRL uses AVX2-optimization, so that processors
supporting AVX2 will run the factorization model up to four times faster.

Example scripts
===============

The directory /scripts/ contains scripts for demonstrating and testing basic uses of the toolkit. To run the scripts
one should first install the dependencies: Keras, NLTK, TextBlob, Pandas, Ray, Dask Distributed and PySpark.
The scripts also use the TripAdvisor dataset (http://times.cs.uiuc.edu/~wang296/Data/), and the
precomputed word embeddings glove.twitter.27B.100d and glove.6B.50d (http://nlp.stanford.edu/projects/glove/). Test
data from Crowdflower Open data & Kaggle is provided in the /data directory.

Airline Classification Example
------------------------------
classify_airline_sentiment.py shows training and combining predictions with four classifier scripts that use the
Wordbatch extractors and models: wordhash_regressor.py, wordbag_regressor.py, wordseq_regressor.py and
wordvec_regressor.py. The header part of the script can be modified to choose the backend. By default Ray is used and
passed to the other scripts.

Backends Benchmark Example
--------------------------
backends_benchmark.py shows how to benchmark different backends on two simple pipeline tasks:
using ApplyBatch with Scikit-learn HashingVectorizer, and running WordBatch Pipeline with most of its possible
processing steps. Dask and Spark are commented out by default, as these need command-line configuration.
All three distributed backends can be configured to run across a distributed cluster, as done in the
commented-out code.


Contributors
============
Antti Puurula

Anders Topper

Cheng-Tsung Liu
