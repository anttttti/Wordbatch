#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import types
from contextlib import closing
from collections import Counter, defaultdict
import operator
#from nltk.metrics import edit_distance
import Levenshtein #python-Levenshtein
import scipy.sparse as ssp
import random
import re
import os
import pandas as pd
import sys
from math import ceil
if sys.version_info.major == 3:
    import copyreg as copy_reg
else:
    import copy_reg

WB_DOC_CNT= u'###DOC_CNT###'
def _pickle_method(m):
    if sys.version_info.major == 3:
        if m.im_self is None:  return getattr, (m.im_self.__class__, m.im_func.__name__)
        else:  return getattr, (m.im_self, m.im_func.__name__)
    else:
        if m.__self__ is None:  return getattr, (m.__self__.__class__, m.__func__.__name__)
        else:  return getattr, (m.__self__, m.__func__.__name__)
copy_reg.pickle(types.MethodType, _pickle_method)

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
    def __init__(self, normalize_text= default_normalize_text, spellcor_count=0, spellcor_dist= 2, n_words= 10000000,
                 min_df= 0, max_df= 1.0, raw_min_df= -1, procs= 0, verbose= 1, minibatch_size= 20000,
                 stemmer= None, pos_tagger= None, extractor=None, timeout= 600, use_sc= False,
                 method= "multiprocessing"):
        if procs==0:  procs= multiprocessing.cpu_count()
        self.procs= procs
        self.verbose= verbose
        self.minibatch_size= minibatch_size
        self.timeout= timeout

        self.dictionary_freeze= False
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
        self.use_sc= use_sc
        self.method= method

    def set_extractor(self, extractor=None):
        if extractor != None:
            if type(extractor) != tuple and type(extractor) != list:  self.extractor = extractor(self, {})
            else:  self.extractor = extractor[0](self, extractor[1])
        else: self.extractor = None

    def update_dictionary(self, texts, dft, dictionary, min_df, input_split= False):
        dfts2= self.parallelize_batches(self.procs, batch_get_dfs, texts, [], input_split= input_split,
                                        merge_output=False)
        if self.use_sc==True:  dfts2= [batch[1] for batch in dfts2.collect()]
        if dictionary!=None:  self.doc_count+= sum([dft2.pop(WB_DOC_CNT) for dft2 in dfts2])
        for dft2 in dfts2:  dft.update(dft2)

        if dictionary!=None:
            sorted_dft = sorted(list(dft.items()), key=operator.itemgetter(1), reverse=True)
            if type(self.min_df) == type(1): min_df2 = self.min_df
            else: min_df2= self.doc_count * self.min_df
            if type(self.max_df) == type(1):  max_df2 = self.max_df
            else:  max_df2= self.doc_count * self.max_df
            for word, df in sorted_dft:
                if len(dictionary)>= self.n_words: break
                if df<min_df2 or df>max_df2: continue
                if word in dictionary:  continue
                dictionary[word] = len(dictionary)+1
                if self.verbose>2: print("Add word to dictionary:", word, dft[word], dictionary[word])

        if min_df>0:
            if self.verbose>1: print("Document Frequency Table size:", len(dft))
            if type(min_df) == type(1):
                for word in list(dft.keys()):
                    if dft[word]<min_df:  dft.pop(word)
            else:
                for word in list(dft.keys()):
                    if float(dft[word])/self.doc_count < min_df:  dft.pop(word)
            if self.verbose > 1: print("Document Frequency Table pruned size:", len(dft))

    def normalize_texts(self, texts, input_split=False, merge_output=True):
        texts2= self.parallelize_batches(self.procs, batch_normalize_texts, texts, [self.normalize_text],
                                          input_split=input_split, merge_output=merge_output)
        #if self.use_sc==False:  return [item for sublist in texts2 for item in sublist]
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
        return self.parallelize_batches(self.procs, batch_correct_spellings, texts, [corrs, self.pos_tagger],
                                        input_split=input_split, merge_output=merge_output)
        #if self.use_sc== False:  return [item for sublist in texts for item in sublist]
        #return texts

    def fit(self, texts, labels= None, return_texts= False, input_split= False, merge_output= True):
        if self.verbose > 0:  print("Normalize text")
        if self.normalize_text != None:
            texts= self.normalize_texts(texts, input_split= input_split, merge_output= False)
            input_split= True
        if self.spellcor_count> 0 or self.stemmer!=None:
            if self.verbose > 0:  print("Update raw dfts")
            self.update_dictionary(texts, self.raw_dft, None, self.raw_min_df, input_split= input_split)
            if self.verbose > 0:  print("Normalize wordforms")
            texts= self.normalize_wordforms(texts, input_split= input_split, merge_output= False)
            input_split= True
            if self.preserve_raw_dft==False:  self.raw_dft= Counter()
        if not(self.dictionary_freeze):  self.update_dictionary(texts, self.dft, self.dictionary, self.min_df,
                                                                input_split= input_split)
        if self.verbose> 2: print("len(self.raw_dft):", len(self.raw_dft), "len(self.dft):", len(self.dft))
        if return_texts:
            if merge_output:  return self.merge_batches(texts)
            else:  return texts

    def fit_transform(self, texts, labels=None, extractor= None, cache_features= None, input_split= False):
        return self.transform(texts, labels, extractor, cache_features, input_split)

    def partial_fit(self, texts, labels=None, input_split= False, merge_output= True):
        return self.fit(texts, labels, input_split, merge_output)

    def transform(self, texts, labels= None, extractor= None, cache_features= None, input_split= False):
        if self.use_sc==True:  cache_features= None  #No feature caching with Spark
        if extractor== None:  extractor= self.extractor
        if cache_features != None and os.path.exists(cache_features):  return extractor.load_features(cache_features)
        if not(input_split):  texts= self.split_batches(texts, self.minibatch_size)
        texts= self.fit(texts, return_texts=True, input_split=True, merge_output=False)
        if extractor!= None:
            texts= extractor.transform(texts, input_split= True, merge_output= True)
            if cache_features!=None:  extractor.save_features(cache_features, texts)
            return texts
        else:
            return self.merge_batches(texts)

    def lists2rddbatches(self, lists, sc, minibatch_size=0):
        if minibatch_size==0:  minibatch_size= self.minibatch_size
        start= 0; len_data= len(lists[0]); batch_count= 0
        batches= []
        while start < len_data:
            batches.append([batch_count]+[column[start:start + minibatch_size] for column in lists])
            start+= minibatch_size
            batch_count+= 1
        return sc.parallelize(batches)

    def rddbatches2lists(self, rddbatches, sort= True):
        batches= rddbatches.collect()
        if sort:  batches= sorted(batches)
        texts= []
        labels= []
        for batch in batches:
            texts.append(batch[1])
            labels.append(batch[2])
        texts= self.merge_batches(texts)
        labels= self.merge_batches(labels)
        return texts, labels

    def split_batches(self, data, minibatch_size= None):
        if minibatch_size==None: minibatch_size= self.minibatch_size
        data_type= type(data)
        if data_type is list or data_type is tuple:  len_data= len(data)
        else:  len_data= data.shape[0]
        if minibatch_size> len_data:  minibatch_size= len_data
        if data_type == pd.DataFrame:
            data_split = [data.iloc[x * minibatch_size:(x + 1) * minibatch_size] for x in
                          range(int(ceil(len_data / minibatch_size)))]
        else:
            data_split= [data[x* minibatch_size:(x+1)*minibatch_size]
                         for x in range(int(ceil(len_data/minibatch_size)))]
        return data_split

    def merge_batches(self, data):
        if isinstance(data[0], ssp.csr_matrix):
            return ssp.vstack(data)
        return [item for sublist in data for item in sublist]

    def parallelize_batches(self, procs, task, data, args, method=None, timeout=-1, rdd_col= 1, input_split=False,
                            merge_output= True, minibatch_size= None):
        if minibatch_size== None: minibatch_size= self.minibatch_size
        if method == None: method= self.method
        if self.verbose > 1: print("Parallel task:", task, " Method:", method, " Procs:", self.procs, " input_split:",
                                   input_split)
        if self.use_sc==True:
            def apply_func(batch):  return batch[:rdd_col]+[task([batch[rdd_col]]+args)]+batch[rdd_col+1:]
            results= data.map(apply_func)
            return results

        if timeout==-1:  timeout= self.timeout
        attempt= 0
        if not(input_split):
            paral_params= [[data_batch]+ args for data_batch in self.split_batches(data, minibatch_size)]
        else:
            paral_params=  [[data_batch]+ args for data_batch in data]
        if method == "serial":
            results = [task(minibatch) for minibatch in paral_params]
        else:
            while (attempt != -1):
                try:
                    if method=="multiprocessing":
                        with closing(multiprocessing.Pool(max(1, procs), maxtasksperchild=2)) as pool:
                            results= pool.map_async(task, paral_params)
                            if timeout==0:
                                pool.close()
                                pool.join()
                                results= results.get()
                            else:
                                results.wait(timeout=timeout)
                                if results.ready():  results= results.get()
                                else:  raise ValueError('Parallelization timeout')
                        break
                    elif method=="threading":
                        with closing(multiprocessing.dummy.Pool(max(1,procs))) as pool:
                           results= pool.map(task, paral_params)
                           pool.close()
                           pool.join()
                    #elif method == "parallelpython":
                    #    job_server= pp.Server()
                    #    jobs= [job_server.submit(task, (x,), (), ()) for x in paral_params]
                    #    results= [x() for x in jobs]
                except:
                    print("Parallelization fail. Method:", method, "Task:", task)
                    attempt+= 1
                    if timeout!=0: timeout*= 2
                    if attempt>=5:  return None
                    print("Retrying, attempt:", attempt, "timeout limit:", timeout, "seconds")
                    continue
                attempt= -1
        if merge_output:  return self.merge_batches(results)
        return results

    def shuffle_batch(self, texts, labels= None, seed= None):
        if seed!=None:  random.seed(seed)
        index_shuf= list(range(len(texts)))
        random.shuffle(index_shuf)
        texts= [texts[x] for x in index_shuf]
        if labels==None:  return texts
        labels= [labels[x] for x in index_shuf]
        return texts, labels

    def predict_parallel(self, texts, clf):
        return self.merge_batches(self.parallelize_batches(self.procs / 2, batch_predict, texts, [clf]))

    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items())

    def __setstate__(self, params):
        for key in params:  setattr(self, key, params[key])
