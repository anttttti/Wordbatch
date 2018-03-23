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
    dft= Counter()
    for text in args[0]:
        for word in set(text.split(" ")):  dft[word]+= 1
    dft[WB_DOC_CNT]+= len(args[0]) #Avoid Spark collect() by counting here
    return dft

def batch_normalize_texts(args):
    normalize_text= args[1]
    return [normalize_text(text) for text in args[0]]

def batch_predict(args):
    return args[1].predict(args[0])

def correct_spelling(word, dft, spell_index, spellcor_count, spellcor_dist):
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
    corrs= args[1]
    if args[2]== None:
        return [u" ".join([corrs.get(word, word) for word in text.split(" ")]) for text in args[0]]
    #Tagger provided. Tag each word and return word_tag sequence
    res= []
    pos_tagger= args[2]
    for text in args[0]:
        text2= []
        tags= pos_tagger(text)
        text= text.split(" ")
        for y in range(len(text)):
            word= text[y]
            text2.append(corrs.get(word, word)+"_"+tags[y][1])
        res.append(" ".join(text2))
    return res

def get_deletions(word, order):
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
    def __init__(self, normalize_text= default_normalize_text, spellcor_count=0, spellcor_dist= 2, max_words= 10000000,
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

        import wordbatch.transformers.dictionary as dictionary
        self.dictionary= dictionary.Dictionary(self.batcher, min_df=min_df, max_df=max_df, max_words= 10000000000000,
                                               freeze= False, verbose=1)
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
        self.max_words= max_words
        self.min_df= min_df
        self.max_df= max_df

        self.set_extractor(extractor)

    def reset(self):
        self.dictionary.reset()
        return self

    def set_extractor(self, extractor=None):
        if extractor != None:
            if type(extractor) != tuple and type(extractor) != list:
                self.extractor = extractor(self.batcher, self.dictionary,  {})
            else:  self.extractor = extractor[0](self.batcher, self.dictionary, extractor[1])
        else: self.extractor = None

    def normalize_texts(self, texts, input_split=False, merge_output=True):
        texts2= self.parallelize_batches(batch_normalize_texts, texts, [self.normalize_text],
                                          input_split=input_split, merge_output=merge_output)
        return texts2

    def normalize_wordforms(self, texts, input_split= False, merge_output= True):
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
            #self.update_dictionary(texts, self.dft, self.dictionary, self.min_df, input_split= input_split)
            self.dictionary.fit(texts, input_split=input_split, reset= reset)

        if self.verbose> 2: print("len(self.raw_dft):", len(self.raw_dft), "len(self.dft):", len(self.dft))
        return texts

    def fit(self, texts, input_split= False, reset= True):
        self.process(texts, input_split, reset=reset, update= True)
        return self

    def transform(self, texts, extractor= None, cache_features= None, input_split= False, reset= False, update= False):
        if self.use_sc==True:  cache_features= None  #No feature caching with Spark
        if extractor== None:  extractor= self.extractor
        if cache_features != None and os.path.exists(cache_features):  return extractor.load_features(cache_features)
        if not(input_split):  texts= self.split_batches(texts)

        texts= self.process(texts, input_split=True, reset=reset, update= update)
        if extractor!= None:
            texts= extractor.transform(texts, input_split= True, merge_output= True)
            if cache_features!=None:  extractor.save_features(cache_features, texts)
            return texts
        else:
            return self.merge_batches(texts)

    def partial_fit(self, texts, input_split=False):
        return self.fit(texts, input_split, reset=False)

    def fit_transform(self, texts, extractor=None, cache_features=None, input_split=False, reset=True):
        return self.transform(texts, extractor, cache_features, input_split, reset, update=True)

    def partial_fit_transform(self, texts, extractor=None, cache_features=None, input_split=False):
        return self.transform(texts, extractor, cache_features, input_split, reset=False, update=True)

    def predict_parallel(self, texts, clf, procs=None):
        if procs==None: procs= int(self.batcher.procs / 2)
        return self.merge_batches(self.parallelize_batches(batch_predict, texts, [clf], procs=procs))

    def parallelize_batches(self, *args, **kwargs):
        return self.batcher.parallelize_batches(*args, **kwargs)

    def split_batches(self, *args, **kwargs):
        return self.batcher.split_batches(*args, **kwargs)

    def merge_batches(self, *args, **kwargs):
        return self.batcher.merge_batches(*args, **kwargs)

    def shuffle_batch(self, *args, **kwargs):
        return self.batcher.shuffle_batch(*args, **kwargs)

    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items())

    def __setstate__(self, params):
        for key in params:  setattr(self, key, params[key])
