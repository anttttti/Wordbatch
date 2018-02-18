Wordbatch
=========

Parallel text feature extraction for machine learning.

Wordbatch produces parallel feature extraction from raw text data for uses such as deep learning and text analytics. The most basic use for Wordbatch is as a drop-in replacement for the more basic non-parallelized extraction available in toolkits such as Scikit-learn, TfLearn and Keras. Wordbatch additionally provides customizable preprocessing and feature extractors that improve predictive performance.

Unlike text preprocessing in these toolkits that either deal with text as a single batch or as a stream, Wordbatch works best with large minibatches of text data. Wordbatch internally stores per-batch statistics of the data, and applies these for uses such as dictionary selection, spelling correction, and online IDF weighting. The larger the batches, the better choices Wordbatch can make in extracting features.

The current text preprocessing options include passing any function as text normalization to be parallelized, a constant-time adaptive version of Norvig spelling correction, and passing any function for parallel stemming.

Currently four basic feature extractor classes are provided:

- WordHash is simply the Scikit-learn HashingVectorizer wrapped with the Wordbatch parallelization, providing multiplied processing speeds
- WordBag is a flexible alternative to Wordhash, providing cababilities missing from Scikit-learn, such as IDF and per n-gram order weighting of hashed features, windowed and distance-weighted polynomial interactions, and more transforms for counts.
- WordSeq provides sequences of word integers, as used by the deep learning toolkits for input into LSTM models.
- WordVec provides embedding transforms from words into wordvectors

A list of extractors can be defined. For example, word vector sequences can be projected into per-document vectors, and concatenated with the vectors from other word vector embeddings.

Four basic OpenMP-parallelized L1&L2-regularized online learning models are provided, for single-label regression and classification:

- FTRL : Linear model Proximal-FTRL that has become the most popular algorithm for online learning of linear models in Kaggle competions. The Cython-optimized implementation should be the fastest available version of FTRL. 
- FM_FTRL : Factorization Machines. Linear effects estimated with FTRL and factor effects estimated with adaptive SGD. Prediction and estimation multithreaded across factors.
- NN_Relu_H1 : Neural Network with 1 hidden layer and Rectified Linear Unit activations, estimated with adaptive SGD. Prediction and estimation multithreaded across hidden layer.
- NN_Relu_H2: Neural Network with 2 hidden layers and Rectified Linear Unit activations, estimated with adaptive SGD. Prediction multithreaded across 2nd hidden layer, estimation across 1st hidden layer outputs. 
The adaptive SGD optimizer works like Adagrad, but pools the adaptive learning rates across hidden nodes using the same feature. This makes learning more robust and requires less memory.
  
Wordbatch is written with Cython, and uses concurrent threading, multiprocessing and OpenMP parallelization for circumventing the Python GIL. License is GNU GPL 2.0, and less restrictive licenses are available on request.

Requirements
============
Linux/Windows. Python 2.7/3.6

Installation
============
pip install wordbatch

Getting started
===============

| #from sklearn.feature_extraction.text import HashingVectorizer
| #from sklearn.linear_model import *
| #vct= HashingVectorizer()
| #clf= SGDRegressor()
|
| import wordbatch
| from wordbatch.models import FTRL
| from wordbatch.extractors import WordBag
| wb= wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams":2, "hash_ngrams_weights":[0.5, -1.0], "hash_size":2**23, "norm":'l2', "tf":'log', "idf":50.0}))
| clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=1)
|
| train_texts= ["Cut down a tree with a herring? It can't be done.", "Don't say that word.", "How can we not say the word if you don't tell us what it is?"]
| train_labels= [1, 0, 1]
| test_texts= ["Wait! I said it! I said it! Ooh! I said it again!"]
|
| clf.fit(wb.transform(train_texts), train_labels)
| preds= clf.predict(wb.transform(test_texts))


Example scripts
===============

The directory /scripts/ contains four scripts for demonstrating the basic extractors, and a Scikit-learn ensemble model to combine predictions. To run the scripts you should first install the dependencies: Keras, NLTK, TextBlob and Pandas. The scripts also use the TripAdvisor dataset (http://times.cs.uiuc.edu/~wang296/Data/) for training models, and the precomputed word embeddings glove.twitter.27B.100d and glove.6B.50d (http://nlp.stanford.edu/projects/glove/). The test data from Crowdflower Open data & Kaggle is provided in the /data directory.

- wordhash_regressor.py shows wordbatch.extractors.WordHash, and feature extraction concurrent with file reading
- wordbag_regressor.py shows wordbatch.extractors.WordBag, and online feature extraction and parallel FTRL training
- wordseq_regressor.py shows wordbatch.extractors.WordSeq, and training a 1D-convnet regression model
- wordvec_regressor.py shows wordbatch.extractors.WordVec, and combining word vector embeddings for FTRL training
- classify_airline_sentiment.py show how to combine predictions from the four scripts using a Random Forest Regressor on the airline sentiment data

Spark integration
=================
Starting from 1.2, Wordbatch has full Spark integration. All processing steps will be parallelized by Spark, simply by setting wb.use_sc=True and providing data in the RDD format produced by wb.lists2rddbatches(texts). 

A basic script using this is wordbag_regressor_spark.py, which is the wordbag_regressor.py script modified to run on Spark. This converts each minibatch of training data into an RDD, does feature extraction on the RDD, and collects the resulting features for local FTRL model training. A more practical script should read the data from parallelized storage, and implement model training on the RDD as well.

Parallel prediction is also demonstrated in wordbag_regressor_spark.py. By calling the class with predict_parallel(), it will parallelize prediction either locally or on Spark, depending on whether a SparkContext has been set for the class.
