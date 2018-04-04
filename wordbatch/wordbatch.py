#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import types
from collections import Counter, defaultdict
import operator
#from nltk.metrics import edit_distance
import Levenshtein #python-Levenshtein
import re
import os
import sys
import wordbatch.batcher as batcher

WB_DOC_CNT= u'###DOC_CNT###' #Used for Spark document counting across RDFs

def batch_get_dfs(args):
    """Count Word/Token Occurrences over a batch of documents along with total document counts.

    Parameters
    ----------
    args: list
        The first entry(args[0]) is a list of string representing each document.

    Returns
    -------
    dft: Counter
        Keep records of occurrence(s) of each word in the corpus and the total
        document counts. The occurence(s) of a single word can be accessed with
        the word itself as a dictionary key, and the total document counts can
        be accessed with WB_DOC_CNT, which is '###DOC_CNT###' in this case.
    """
    dft= Counter()
    for text in args[0]:
        for word in set(text.split(" ")):  dft[word]+= 1
    dft[WB_DOC_CNT]+= len(args[0]) #Avoid Spark collect() by counting here
    return dft

def batch_normalize_texts(args):
    """Apply a specified normalization function to a batch of documents.

    Parameters
    ----------
    args: list
        The first entry(args[0]) is a list of string containing every document
        to normalize. And the second entry(args[1]) is the normalization
        function to apply, which accepts a single string as input and output
        a single string.

    Returns
    -------
    (anonymous): list
        List of normalized documents.
    """
    normalize_text= args[1]
    return [normalize_text(text) for text in args[0]]

def batch_predict(args):
    """Make Predictions on specified data with a specified estimator/model.

    Parameters
    ----------
    args: list
        The first entry(args[0] is an estimator with a "predict" method
        implemented. And the second entry(args[1]) is a list-like object,
        representing the data(n_samples x n_features) to make predictions on.

    Returns
    -------
    (anonymous): list-like
        The predictions given by the estimator using the data passed.
    """
    return args[1].predict(args[0])

def correct_spelling(word, dft, spell_index, spellcor_count, spellcor_dist):
    """Get the optimal suggestion for correcting the specified word.

    Parameters
    ----------
    word: str
        Original word/token to correct.

    dft: Counter | dict
        Global occurrence counter to look up. This matters in two ways:
        (1) If the occurrence frequency is less than or equal to a given
        threshold(i.e. the "spellcor_count" parameter), then no correction
        searching will be triggered and the word will just be "corrected" by
        itself. (2) If more than one suggestions are found, the one with the
        highest occurrence frequency will be selected as the best suggestion.

    spell_index: dict
        Auxilliary dictionary for suggestion searching, with string as keys and
        lists of string as values. Each key represents a word/token and each value
        represents a list of possible correction suggestions for the word/token.

    spellcor_count: int
        The threshold of word occurrence frequency for the word to be corrected.
        If occurrence number of the specified word is less than or equal to this
        number, no correction program will be triggered for this word.

    spellcor_dist: int
        Max distance within which the correction suggestions are searched. The
        distance metrics is defined by "Levenshtein Distance", i.e., the number
        of characters to insert/delete/alter to turn one given word into the other.

    Returns
    -------
    max_word: str
        The best word/token found to correct the original word. If no qualified
        words/tokens found, this will be set to the original word itself.
    """
    #T. Bocek, E. Hunt, B. Stiller: Fast Similarity Search in Large Dictionaries, 2007
    if dft.get(word, 0)>spellcor_count or len(word)<3:  return word
    max_df= -100000000000000
    max_word= word
    spell_suggestions= get_deletions(word, spellcor_dist)
    candidates= {}
    for x in spell_suggestions:
        if x in spell_index:
            for y in spell_index[x]:  candidates[y]= 1
    #for word2 in list(candidates.keys()):
    for word2 in candidates:
        #score= edit_distance(word, word2, True)
        score= Levenshtein.distance(word, word2)
        if score>spellcor_dist:  continue
        #score = float(dft[word2]) / score
        score= dft[word2]
        #score = Levenshtein.jaro_winkler(word, word2)
        #score= dft[word2]*Levenshtein.jaro_winkler(word, word2)
        if score > max_df:
            max_df= score
            max_word= word2
    return max_word

def batch_correct_spellings(args):
    """Correct spelling and add POS information for a batch of documents.

    Parameters
    ----------
    args: list
        List of arguments. The first entry(args[0]) should be a list(or list-like container)
        of string representing the documents. The second entry(args[1]) should be a dictionary
        for correction, which maps raw words/tokens to the best correction suggestions. The
        third entry(args[2]) is a POS tagging method (input and output form to be checked).

    Returns
    -------
    res: list
        List of processed documents. The processing includes: (1)correct words using given
        suggestions. (2)concatenate the POS tag behind the corrected word to achieve finer-grained
        representation.
    """
    corrs= args[1]
    if args[2]== None:
        return [u" ".join([corrs.get(word, word) for word in text.split(" ")]) for text in args[0]]
    #Tagger provided. Tag each word and return word_tag sequence
    res= []
    pos_tagger= args[2]
    for text in args[0]:
        text2= []
        tags= pos_tagger(text)  # Placed after the text splitting? It works wrong while the input is a plain string
        text= text.split(" ")
        for y in range(len(text)):
            word= text[y]
            text2.append(corrs.get(word, word)+"_"+tags[y][1])
        res.append(" ".join(text2))
    return res

def get_deletions(word, order):
    """Get tokens derived from a word by deleting at most a particular number of characters

    Parameters
    ----------
    word: str
        Original word/token to start the deletion with.

    order: int
        The maximum number of characters to delete.

    Returns
    -------
    (anonymous): list
        List containing all tokens that can be derived the the specified word
        by deleting some characters, where the number of characters deleted is
        not greater than the number specified. Note that the original word itself
        will not be included in the return.

    Examples
    --------
    >>> get_deletions("hello", 3)
    ['ell', 'hll', 'heo', 'ello', 'eo', 'hell', 'lo', 'hel', 'el', 'hlo', 'll',
    'he', 'ho', 'llo', 'hl', 'hllo', 'helo', 'elo']
    """
    stack = {word: order}
    results = {}
    while len(stack) > 0:
        stack2 = {}
        for word2 in stack:
            order2 = stack[word2] - 1
            for x in range(len(word2)):
                if order2 != 0:  stack2[word2[:x] + word2[x + 1:]] = order2
                results[word2[:x] + word2[x + 1:]] = 1
        stack = stack2
    return list(results.keys())

non_alphanums= re.compile(u'[^A-Za-z0-9]+')
def default_normalize_text(text):
    return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x)>1])

class WordBatch(object):
    """

    Parameters
    ----------
    normalize_text:

    spellcor_count:

    spellcor_dist:

    n_words:

    min_df: int | float in range [0.0, 1.0]

    max_df: int | float in range [0.0, 1.0]

    raw_min_df:

    procs: int

    minibatch_size:

    stemmer:

    pos_tagger:

    extractor: type | list | tuple

    timeout: int | float

    use_sc: boolean

    freeze: boolean

    method: {'serial', 'multiprocessing', 'threading'}

    verbose:

    Attributes
    ----------
    batcher: batcher.Batcher

    dictionary:

    dft: Counter

    raw_dft: Counter

    preserve_raw_dft: bool

    doc_count: int

    extractor: extractors.WordBag

    """
    def __init__(self, normalize_text= default_normalize_text, spellcor_count=0, spellcor_dist= 2, n_words= 10000000,
                 min_df= 0, max_df= 1.0, raw_min_df= -1, procs= 0, minibatch_size= 20000,
                 stemmer= None, pos_tagger= None, extractor=None, timeout= 600, use_sc= False, freeze= False,
                 method= "multiprocessing", verbose= 1):
        if procs==0:  procs= multiprocessing.cpu_count()
        self.verbose= verbose
        self.use_sc = use_sc

        self.batcher= batcher.Batcher(procs, minibatch_size, timeout, use_sc, method, verbose)
        self.freeze= freeze
        self.dictionary= {}
        self.dft= Counter()
        self.raw_dft= Counter()
        self.preserve_raw_dft= False

        self.normalize_text= normalize_text
        if spellcor_count==0:  spellcor_dist= 0
        elif spellcor_dist==0:  spellcor_count= 0
        self.spellcor_count= spellcor_count
        self.spellcor_dist= spellcor_dist
        self.stemmer= stemmer
        if raw_min_df==-1:  self.raw_min_df= min_df
        else:  self.raw_min_df= raw_min_df
        self.pos_tagger= pos_tagger

        self.doc_count= 0
        self.n_words= n_words
        self.min_df= min_df
        self.max_df= max_df

        self.set_extractor(extractor)

    def reset(self):
        """Clear the instance's dictionary and word counters"""
        self.dictionary = {}
        self.dft = Counter()
        self.raw_dft = Counter()
        return self

    def set_extractor(self, extractor=None):
        """Set the extractor associated with the instance.

        Parameters
        ----------
        extractor: type | list | tuple | None
            If type, it specified the class of the extractor to use, and an extractor
            instance with default settings will be constructed. If list or tuple, the
            first entry(extractor[0]) should be type indicating the class of the extractor
            to use, and the second entry(extractor[1]) should be a dictionary representing
            the setting of the extractor. If None, the instance's extractor will be
            simply set as None.
        """
        if extractor != None:
            if type(extractor) != tuple and type(extractor) != list:  self.extractor = extractor(self, {})
            else:  self.extractor = extractor[0](self, extractor[1])
        else: self.extractor = None

    def get_pruning_dft(self, dft):
        """Prepare meta-data for further dictionary pruning.

        Parameters
        ----------
        dft: Counter
            Dictionary to be pruned. As a Counter instance it should have words
            as keys and occurrence frequencies as values. Note that it will not be
            pruned here, what this method is responsible for is just to prepare
            meta-data for the pruning process in the next steps.

        Returns
        -------
        sorted_dft: Counter
            Dictionary sorted from the original one in descending order, i.e.
            words with highest frequencies will be placed at the top.

        min_df2: int
            The lower threshold to be used in further pruning. Words with a document
            frequency strictly lower than the given value will be pruned out.

        max_df2 : int
            The upper threshold to be used in further pruning. Words with a document
            frequency strictly higher than the given value will be pruned out.
        """
        sorted_dft = sorted(list(dft.items()), key=operator.itemgetter(1), reverse=True)
        if type(self.min_df) == type(1):  min_df2 = self.min_df
        else:  min_df2 = self.doc_count * self.min_df
        if type(self.max_df) == type(1):   max_df2 = self.max_df
        else:  max_df2 = self.doc_count * self.max_df
        return sorted_dft, min_df2, max_df2


    def update_dictionary(self, texts, dft, dictionary, min_df, input_split= False):
        """Update word frequency counter and word-index dictionary with given documents.

        Parameters
        ----------
        texts: list
            List of list-like objects representing document mini-batches. Each entry
            should be a list-like object with each single document, which should be
            a string, as an entry.

        dft: Counter
            Original counter to update. New Occurrences found in given documents
            will be added to this instance inplace.

        dictionary: dict | None
            Dictionary mapping words to unique indices. It contains string keys
            and integer values. If specified as None, no tracking of the words and
            corresponding indices will be performed.

        min_df: (To removed? this parameter is not used)

        input_split: boolean, default False
            If True, data will be splitted into single samples before applying task in parellel, otherwise
            data will processed in mini-batches.
        """
        #Update document frequencies.
        dfts2= self.parallelize_batches(batch_get_dfs, texts, [], input_split= input_split,
                                        merge_output=False)
        if self.use_sc==True:  dfts2= [batch[1] for batch in dfts2.collect()]
        if dictionary!=None:  self.doc_count+= sum([dft2.pop(WB_DOC_CNT) for dft2 in dfts2])
        for dft2 in dfts2:  dft.update(dft2)

        if dictionary!=None:
            #Add entries. Online pruning only used to prevent inclusion into dictionary
            sorted_dft, min_df2, max_df2 = self.get_pruning_dft(dft)
            for word, df in sorted_dft:
                if len(dictionary)>= self.n_words: break
                if df<min_df2 or df>max_df2: continue
                if word in dictionary:  continue
                dictionary[word] = len(dictionary)+1
                if self.verbose>2: print("Add word to dictionary:", word, dft[word], dictionary[word])

    def prune_dictionary(self, n_words=None, min_df=None, max_df=None, re_encode= False, prune_dfs= True,
                         set_n_words= True):
        #Prune dictionary. Optionally prune document frequency table as well
        if n_words!=None: self.n_words= n_words
        if min_df!=None: self.min_df= min_df
        if max_df!= None: self.max_df= max_df
        n_words= self.n_words
        dictionary = self.dictionary
        dft = self.dft
        sorted_dft, min_df2, max_df2 = self.get_pruning_dft(dft)
        c= 0
        print(len(sorted_dft), len(self.dictionary), len(self.raw_dft))
        for word, df in sorted_dft:
            if word not in dictionary:
                if re_encode:  dictionary[word]= -1
                else:  continue
            c+= 1
            if c > n_words or df < min_df2 or df > max_df2:
                if prune_dfs: dft.pop(word)
                dictionary.pop(word)
            elif re_encode:
                dictionary[word]= c
        if set_n_words:  self.n_words= len(dictionary)

    def normalize_texts(self, texts, input_split=False, merge_output=True):
        """Normalize documents with basic non-semantic rules in parallel.

        The normalization here is some cleaning regardless of the corpora or any semantic
        knowledge. For example, removing or replacing some special characters.

        Parameters
        ----------
        texts: list-like
            List-like object containing every single document to normalize as an entry.

        input_split: boolean, default False
            If True, data will be splitted into single samples before applying task in parellel, otherwise
            data will just be splitted into mini-batches with specified size.

        merge_output: boolean, default True
            If True, results from mini-batches will be merged into one single instance before being
            returned, otherwise they will be returned in form of list.

        Returns
        -------
        texts2: list-like | list of list-like
            If merge_output is specified as True, this will be a list-like object representing
            the normalized documents, with each entry as a normalized document. Otherwise this
            will be a list of list-like objects, with each entry representing the results
            from a mini-batch.
        """
        texts2= self.parallelize_batches(batch_normalize_texts, texts, [self.normalize_text],
                                          input_split=input_split, merge_output=merge_output)
        return texts2

    def normalize_wordforms(self, texts, input_split= False, merge_output= True):
        """Normalize documents with semantic rules in parallel.

        The normalization here is some cleaning with corpora or some semantic
        knowledge involved. For example, correction of words using other high-frequency
        words discovered in the corpora, stemming.

        Parameters
        ----------
        texts: list-like
            List-like object containing every single document to normalize or
            every mini-batch of documents to normalize as an entry.

        input_split: boolean, default False
            If True, every single entry in 'texts' will be regarded as a mini-batch and processed
            by an individual process/thread, otherwise, it will be splitted into mini-batches
            before processed by the workers. Note that if your 'texts' is already a list of
            mini-batches, you should set this parameters as True so that no over-splitting
            will happen.

        merge_output: boolean, default True
            If True, results from mini-batches will be merged into one single instance before being
            returned, otherwise they will be returned in form of list.

        Returns
        -------
        (anonymous): list-like | list of list-like
            If merge_output is specified as True, this will be a list-like object representing
            the documents, with each entry as a single document. Otherwise this will be a list
            of list-like objects, with each entry representing the processed results from
            a mini-batch of documents.
        """
        if self.verbose > 0:  print("Make word normalization dictionary")
        if self.spellcor_dist>0:
            raw_dft2= {word:self.raw_dft[word]
                                for word in self.raw_dft if self.raw_dft[word] > self.spellcor_count}
            spell_index= defaultdict(list)
            for word in raw_dft2:
                if len(word)>15:  continue
                for word2 in get_deletions(word, self.spellcor_dist):
                    spell_index[word2].append(word)
        if self.stemmer!=None:
            if self.spellcor_count>0:
                corrs= {word:self.stemmer.stem(correct_spelling(
                      word, raw_dft2, spell_index, self.spellcor_count, self.spellcor_dist)) for word in self.raw_dft}
            else: corrs= {word:self.stemmer.stem(word) for word in self.raw_dft}
        else:
            corrs = {word:correct_spelling(
                word, raw_dft2, spell_index, self.spellcor_count, self.spellcor_dist) for word in self.raw_dft}
        corrs= {key:value for key, value in corrs.items() if key!=value}
        if self.verbose > 0:  print("Make word normalizations")
        return self.parallelize_batches(batch_correct_spellings, texts, [corrs, self.pos_tagger],
                                        input_split=input_split, merge_output=merge_output)

    def process(self, texts, input_split= False, reset= True, update= True):
        """Normalize documents and update the instance's states accordingly.

        A complete normalization process will be applied to the documents, including
        general normalization(e.g. removing and replacing special characters) and
        normalization with corpora information and domain knowledge. Then the instance's
        data structures, i.e. word counter and word-index dictionary will be updated
        using the normalized documents. Also similar data structures keeping the information
        of the raw documents may be kept, according the setting during the instance
        construction.

        Parameters
        ----------
        texts: list-like
            List-like object containing every single document to process as an entry.

        input_split: boolean, default False
            If True, before the first normalization process, every single document
            in 'text' will be regarded as a mini-batch and processed by an individual
            process/thread, otherwise it will be splitted into mini-batches before
            processed by workers. It only affects the way how the first normalization
            process is executed, and does not matter at all in how the subsequent processing
            is done.


        reset: boolean, default True
            If True, the instance's word counter and dictionary will be cleared
            before any processing, otherwise the previous records will be kept
            while new records being added.

        update: boolean, default True
            If True, the instance's word counter and dictionary will be updated with
            the processed documents after the normalization.

        Returns
        -------
        texts: list
            List containing mini-batches of normalized documents as entries.
        """
        if reset:  self.reset()
        if self.freeze:  update= False

        if self.normalize_text != None:
            if self.verbose > 0:  print("Normalize text")
            texts= self.normalize_texts(texts, input_split= input_split, merge_output= False)
            input_split= True

        if self.spellcor_count> 0 or self.stemmer!=None:
            if self.verbose > 0:  print("Update raw dfts")
            if update:  self.update_dictionary(texts, self.raw_dft, None, self.raw_min_df, input_split= input_split)
            if self.verbose > 0:  print("Normalize wordforms")
            texts= self.normalize_wordforms(texts, input_split= input_split, merge_output= False)
            input_split= True
            if self.preserve_raw_dft==False:  self.raw_dft= Counter()

        if update:
            self.update_dictionary(texts, self.dft, self.dictionary, self.min_df, input_split= input_split)

        if self.verbose> 2: print("len(self.raw_dft):", len(self.raw_dft), "len(self.dft):", len(self.dft))
        return texts

    def fit(self, texts, input_split= False, reset= True):
        """
        Parameters
        ----------
        texts: list-like
            List-like object containing the documents to fit into the model.

        input_split: boolean, default False
            If True, before the first normalization process, every single document
            in 'text' will be regarded as a mini-batch and processed by an individual
            process/thread, otherwise it will be splitted into mini-batches before
            processed by workers. It only affects the way how the first normalization
            process is executed, and does not matter at all in how the subsequent processing
            is done.

        reset: boolean, default True
            If True, the instance's word counter and dictionary will be cleared
            before fitting, otherwise the previous records will be kept
            while new records being added.
        """
        self.process(texts, input_split, reset=reset, update= True)
        return self

    def transform(self, texts, extractor= None, cache_features= None, input_split= False, reset= False, update= False):
        """

        Parameters
        ----------
        texts:

        extractor:

        cache_features:

        input_split:

        reset:

        update:

        Returns
        -------
        """
        if self.use_sc==True:  cache_features= None  #No feature caching with Spark
        if extractor== None:  extractor= self.extractor
        if cache_features != None and os.path.exists(cache_features):  return extractor.load_features(cache_features)
        if not(input_split):  texts= self.split_batches(texts)

        texts= self.process(texts, input_split=True, reset=reset, update= update)
        if extractor!= None:
            texts= extractor.transform(texts, input_split= True, merge_output= True)
            if cache_features!=None:  extractor.save_features(cache_features, texts)
            return texts  # Antti: would you mind unify the symbol of return? It will be better for documentation
        else:
            return self.merge_batches(texts)

    def partial_fit(self, texts, input_split=False):
        """Fit the model with the previous records will always be cleared at the beginning"""
        return self.fit(texts, input_split, reset=False)

    def fit_transform(self, texts, extractor=None, cache_features=None, input_split=False, reset=True):
        return self.transform(texts, extractor, cache_features, input_split, reset, update=True)

    def partial_fit_transform(self, texts, extractor=None, cache_features=None, input_split=False):
        return self.transform(texts, extractor, cache_features, input_split, reset=False, update=True)

    def predict_parallel(self, texts, clf, procs=None):
        """Make Predictions with a specified estimator/model in parallel

        Parameters
        ----------
        texts: list-like
            List of samples to make predictions on.

        clf: object
            Estimator/Model to make predictions with a "predict" function
            implemented.

        procs: int | None, default None
            If specified as an integer, it indicates the number of
            process(es)/thread(s) to use to make prediction in parallel.
            If None, half of process(es)/thread(s) of the associated batcher
            will be used as default.

        Returns
        -------
        (anonymous): list-like
            List-like object containing every single prediction as
            an entry.

        """
        if procs==None: procs= int(self.batcher.procs / 2)
        return self.merge_batches(self.parallelize_batches(batch_predict, texts, [clf], procs=procs))

    def parallelize_batches(self, *args, **kwargs):
        """Apply a specified function/task to the data specified in parallel"""
        return self.batcher.parallelize_batches(*args, **kwargs)

    def split_batches(self, *args, **kwargs):
        """Split data into mini-batches"""
        return self.batcher.split_batches(*args, **kwargs)

    def merge_batches(self, *args, **kwargs):
        """Merge a list of data mini-batches into one single data instance"""
        return self.batcher.merge_batches(*args, **kwargs)

    def shuffle_batch(self, *args, **kwargs):
        """Shuffle a list of samples, as well as the labels if specified"""
        return self.batcher.shuffle_batch(*args, **kwargs)

    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items())

    def __setstate__(self, params):
        for key in params:  setattr(self, key, params[key])
