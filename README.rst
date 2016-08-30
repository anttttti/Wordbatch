Wordbatch
=======================

Parallel text feature extraction for machine learning.

Wordbatch produces parallel feature extraction from raw text data for uses
such as deep learning and text analytics. The most basic use for Wordbatch
is as a drop-in replacement for the more basic non-parallelized extraction
available in toolkits such as Scikit-learn, TfLearn, Keras, and Neon.
Wordbatch additionally provides customizable preprocessing and feature
extractors that improve predictive performance.

Unlike text preprocessing in these toolkits that either deal with text as
a single batch or as a stream, Wordbatch works best with large minibatches
of text data. Wordbatch internally stores per-batch statistics of the data,
and applies these for uses such as dictionary selection, spelling correction,
and online IDF weighting. The larger the batches, the better choices
Wordbatch can make in extracting features.

The current text preprocessing options include passing any function as text
normalization to be parallelized, a constant-time adaptive version of Norvig
spelling correction, and passing any function for parallel stemming.

Currently four basic feature extractor classes are provided:
- Wordhash is simply the Scikit-learn HashingVectorizer wrapped with the
  Wordbatch parallelization, providing multiplied processing speeds.
- Wordbag is a flexible alternative to Wordhash, providing cababilities
  missing from Sciikit-learn, such as IDF and per n-gram order weighting of
  hashed features, windowed and distance-weighted polynomial interactions,
  and more transforms for counts.
- Wordseq provides sequences of word integers, as used by the deep learning
  toolkits for input into LSTM models.
- Wordvec provides embedding transforms from words into wordvectors

A list of extractors can be defined. For example, word vector sequences can be
projected into per-document vectors, and concatenated with the vectors from
other word vector embeddings.

Wordbatch is written with Cython, and uses multiprocessing for circumventing
the Python GIL. License is GNU GPL 2.0, and less restrictive licenses are
available on request.
