Wordbatch
=========

Parallel text feature extraction for machine learning.

Wordbatch produces parallel feature extraction from raw text data for uses such as deep learning and text analytics. The most basic use for Wordbatch is as a drop-in replacement for the more basic non-parallelized extraction available in toolkits such as Scikit-learn, TfLearn, Keras, and Neon. Wordbatch additionally provides customizable preprocessing and feature extractors that improve predictive performance.

Unlike text preprocessing in these toolkits that either deal with text as a single batch or as a stream, Wordbatch works best with large minibatches of text data. Wordbatch internally stores per-batch statistics of the data, and applies these for uses such as dictionary selection, spelling correction, and online IDF weighting. The larger the batches, the better choices Wordbatch can make in extracting features.

The current text preprocessing options include passing any function as text normalization to be parallelized, a constant-time adaptive version of Norvig spelling correction, and passing any function for parallel stemming.

Currently four basic feature extractor classes are provided:

- WordHash is simply the Scikit-learn HashingVectorizer wrapped with the Wordbatch parallelization, providing multiplied processing speeds
- WordBag is a flexible alternative to Wordhash, providing cababilities missing from Scikit-learn, such as IDF and per n-gram order weighting of hashed features, windowed and distance-weighted polynomial interactions, and more transforms for counts.
- WordSeq provides sequences of word integers, as used by the deep learning toolkits for input into LSTM models.
- WordVec provides embedding transforms from words into wordvectors

A list of extractors can be defined. For example, word vector sequences can be projected into per-document vectors, and concatenated with the vectors from other word vector embeddings.

Starting with 1.1, Wordbatch provides an OpenMP-parallelized version of the FTRL that has become the most popular algorithm for online learning of linear models in Kaggle competions. The implementation should be the fastest available version of FTRL.

Wordbatch is written with Cython, and uses concurrent threading, multiprocessing and OpenMP parallelization for circumventing the Python GIL. License is GNU GPL 2.0, and less restrictive licenses are available on request.


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
| vct= wordbatch.WordBatch(extractors=[(wordbatch.WordBag, {"hash_ngrams":2, "hash_ngrams_weights":[0.5, -1.0], "hash_size":2**23, "norm":'l2', "tf":'log', "idf":50.0})])
| clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=1)
|
| train_texts= ["Some text in this form", "words..."]
| train_labels= [0, 1]
| test_texts= ["more text"] 
|
| clf.fit(vct.transform(train_texts), train_labels)
| preds= clf.predict(vct.transform(test_texts))


Example scripts
===============

The directory /scripts/ contains four scripts for demonstrating the basic extractors, and a Scikit-learn ensemble model to combine predictions. To run the scripts you should first install the dependencies: Nervana Neon, NLTK, TextBlob and Pandas. The scripts also use the TripAdvisor dataset (http://times.cs.uiuc.edu/~wang296/Data/) for training the scripts, and the precomputed word embeddings glove.twitter.27B.100d and glove.6B.50d (http://nlp.stanford.edu/projects/glove/). The test data from Crowdflower Open data & Kaggle is provided in the /data directory.

- wordhash_regressor.py shows wordbatch.WordHash, and feature extraction concurrent with file reading
- wordhash_regressor.py shows wordbatch.WordBag, and online feature extraction and parallel FTRL training
- wordseq_regressor.py shows wordbatch.WordSeq, and training a Bi-LSTM regression model
- wordvec_regressor.py shows wordbatch.WordVec, and combining word vector embeddings for FTRL training
- classify_airline_sentiment.py show how to combine predictions from the four scripts using a Random Forest Regressor on the airline sentiment data
