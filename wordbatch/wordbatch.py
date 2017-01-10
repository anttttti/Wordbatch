#!python
import multiprocessing
import copy_reg
import types
from contextlib import closing
from collections import Counter
import operator
#from nltk.metrics import edit_distance
import Levenshtein #python-Levenshtein
import scipy.sparse as ssp
import scipy as sp
import random
import re

def _pickle_method(m):
   if m.im_self is None:  return getattr, (m.im_self.__class__, m.im_func.__name__)
   else:  return getattr, (m.im_self, m.im_func.__name__)
copy_reg.pickle(types.MethodType, _pickle_method)

def batch_get_dfs(args):
    dft= Counter()
    for text in args[0]:
        for word in set(text.split()):  dft[word]+= 1
    dft['###DOC_CNT###']+= len(args[0]) #Avoid Spark collect() by counting here
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
    for word2 in list(candidates.keys()):
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
    if args[2]== None: return [u" ".join([corrs.get(word, word) for word in text.split()]) for text in args[0]]
    res= []
    pos_tagger= args[2]
    for text in args[0]:
        text2= []
        tags= pos_tagger(text)
        text= text.split()
        for y in range(len(text)):
            word= text[y]
            text2.append(corrs.get(word, word)+"_"+tags[y][1])
        res.append(u" ".join(text2))
    return res

def batch_find_normalizations(args):
    dft = args[1]
    stemmer = args[2]
    spell_index= args[3]
    spellcor_dist= args[4]
    return dict((word, stemmer.stem(correct_spelling(word,dft,spell_index,spellcor_dist))) for word in args[0])

def get_deletions(word, order):
    stack = {word: order}
    results = {}
    while len(stack) > 0:
        stack2 = {}
        for word2 in stack:
            order2 = stack[word2] - 1
            for x in xrange(len(word2)):
                if order2 != 0:  stack2[word2[:x] + word2[x + 1:]] = order2
                results[word2[:x] + word2[x + 1:]] = 1
        stack = stack2
    return list(results.keys())

non_alphanums= re.compile('[^A-Za-z0-9]+')
def default_normalize_text(text):
    return " ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split()] if len(x)>1])

class WordBatch(object):
    def __init__(self, normalize_text= default_normalize_text, spellcor_count=0, spellcor_dist= 2, n_words= 10000000,
                 min_df= 0, max_df= 1.0, raw_min_df= 0, procs= 0, verbose= 1, minibatch_size= 20000,
                 stemmer= None, pos_tagger= None, extractor=None, timeout= 300, use_sc= False):
    #             stemmer=None, pos_tagger=None, extractors=None, timeout=300, use_sc=None):
        if procs==0:  procs= multiprocessing.cpu_count()
        self.procs= procs
        self.verbose= verbose
        self.minibatch_size= minibatch_size
        self.timeout= timeout

        self.dictionary_freeze= False
        self.dictionary= {u'':0}
        self.dft= Counter()
        self.raw_dft= Counter()

        self.normalize_text= normalize_text
        if spellcor_count==0:  spellcor_dist= 0
        elif spellcor_dist==0:  spellcor_count= 0
        self.spellcor_count= spellcor_count
        self.spellcor_dist= spellcor_dist
        self.stemmer= stemmer
        self.raw_min_df= raw_min_df
        self.pos_tagger= pos_tagger

        self.doc_count= 0
        self.n_words= n_words
        self.min_df= min_df
        self.max_df= max_df

        if extractor!=None:
            if type(extractor) != tuple and type(extractor) != list:   self.extractor= extractor(self, {})
            else:  self.extractor= extractor[0](self, extractor[1])
        self.use_sc= use_sc

    def update_dictionary(self, texts, dft, dictionary, min_df):
        dfts2= self.parallelize_batches(self.procs, batch_get_dfs, texts, [])
        if self.use_sc==True:  dfts2= [batch[1] for batch in dfts2.collect()]
        if dictionary!=None:  self.doc_count+= sum([dft2.pop("###DOC_CNT###") for dft2 in dfts2])
        for dft2 in dfts2:  dft.update(dft2)

        if dictionary!=None:
            sorted_dft = sorted(list(dft.items()), key=operator.itemgetter(1), reverse=True)
            for word, df in sorted_dft:
                if len(dictionary)>= self.n_words: break
                if word in dictionary:  continue
                if type(self.min_df)==type(1):
                    if df<self.min_df:  continue
                else:
                    if float(df)/self.doc_count < self.min_df:  continue
                if type(self.max_df)==type(1):
                    if df > self.max_df:  continue
                else:
                    if float(df)/self.doc_count > self.max_df:  continue
                dictionary[word] = len(dictionary)
                if self.verbose>1: print "Add word to dictionary:", word, dft[word], dictionary[word]

        if min_df>0:
            if self.verbose>2: print "Document Frequency Table size:", len(dft)
            if type(min_df) == type(1):
                for word in list(dft.keys()):
                    if dft[word]<min_df:  dft.pop(word)
            else:
                for word in list(dft.keys()):
                    if float(dft[word])/self.doc_count < min_df:  dft.pop(word)
            if self.verbose > 2: print "Document Frequency Table pruned size:", len(dft)

    def normalize_texts(self, texts):
        texts2= self.parallelize_batches(self.procs, batch_normalize_texts, texts, [self.normalize_text])
        if self.use_sc==False:  return [item for sublist in texts2 for item in sublist]
        return texts2

    def normalize_wordforms(self, texts):
        if self.verbose > 0:  print "Make word normalization dictionary"
        if self.spellcor_dist>0:
            raw_dft2 = Counter(
                dict((word, self.raw_dft[word]) for word in self.raw_dft if self.raw_dft[word] > self.spellcor_count))
            spell_index= {}
            for word in raw_dft2:
                if len(word)>15: continue
                for word2 in get_deletions(word, self.spellcor_dist):
                    if word2 not in spell_index:  spell_index[word2]= [word]
                    else:  spell_index[word2].append(word)
        if self.stemmer!=None:
            if self.spellcor_count>0:
                corrs= dict((word, self.stemmer.stem(correct_spelling(
                      word, raw_dft2, spell_index, self.spellcor_count, self.spellcor_dist))) for word in self.raw_dft)
            else: corrs= dict((word, self.stemmer.stem(word)) for word in self.raw_dft)
        else:
            corrs = dict((word, correct_spelling(
                word, raw_dft2, spell_index, self.spellcor_count, self.spellcor_dist)) for word in self.raw_dft)
        if self.verbose > 0:  print "Make word normalizations"
        texts= self.parallelize_batches(self.procs, batch_correct_spellings, texts, [corrs, self.pos_tagger])
        if self.use_sc== False:  return [item for sublist in texts for item in sublist]
        return texts

    def fit(self, texts):
        if self.verbose > 0:  print "Normalize text"
        if self.normalize_text != None:  texts= self.normalize_texts(texts)
        if self.spellcor_count> 0 or self.stemmer!=None:
            if self.verbose > 0:  print "Update raw dfts"
            self.update_dictionary(texts, self.raw_dft, None, self.raw_min_df)
            if self.verbose > 0:  print "Normalize wordforms"
            texts= self.normalize_wordforms(texts)
        if not(self.dictionary_freeze):  self.update_dictionary(texts, self.dft, self.dictionary, self.min_df)
        return texts

    def fit_transform(self, texts, labels=None, cache_features= None):
        return self.transform(texts, labels, cache_features)

    def partial_fit(self, texts, labels=None):  return self.fit(texts, labels)

    def transform(self, texts, labels= None, extractor= None, cache_features= None):
        if self.use_sc==True:  cache_features= None  #No feature caching with Spark
        if extractor== None:  extractor= self.extractor
        texts= self.fit(texts)
        if extractor!= None:  features= extractor.transform(texts)
        if cache_features!=None: extractor.save_features(cache_features + ".lz4", features)
        return features

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
        if isinstance(texts[0], ssp.csr_matrix):  texts= ssp.vstack(texts)
        else:  texts= [item for sublist in texts for item in sublist]
        labels= [item for sublist in labels for item in sublist]
        return texts, labels

    def parallelize_batches(self, procs, task, data, args, method="multiprocessing", timeout=-1, rdd_col= 1):
        if self.use_sc==True:
            if self.verbose>1: print task, method
            def apply_func(batch):  return batch[:rdd_col]+[task([batch[rdd_col]]+args)]+batch[rdd_col+1:]
            results= data.map(apply_func)
            return results

        if timeout==-1:  timeout= self.timeout
        attempt= 0; start= 0; len_data= 0; minibatch_size= self.minibatch_size

        paral_params= []
        if type(data) is list or type(data) is tuple:  len_data= len(data)
        else:
            len_data= data.shape[0]
            if minibatch_size> len_data: minibatch_size= len_data
        while start < len_data:
            paral_params.append([data[start:start + minibatch_size]] + args)
            start += minibatch_size
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
                    return results
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
                print "Parallelization fail. Method:", method, "Task:", task
                attempt+= 1
                if timeout!=0: timeout*= 2
                if attempt>=5:  return None
                print "Retrying, attempt:", attempt, "timeout limit:", timeout, "seconds"
                continue
            attempt= -1
        return results

    def shuffle_batch(self, texts, labels= None, seed= None):
        if seed!=None:  random.seed(seed)
        index_shuf= list(range(len(texts)))
        random.shuffle(index_shuf)
        texts= [texts[x] for x in index_shuf]
        if labels==None:  return texts
        labels= [labels[x] for x in index_shuf]
        return texts, labels

    #def batch_predict(self, args):
    #    fnc= args[1]
    #    return fnc(args[0])


    def predict_parallel(self, texts, clf):
        #return sp.vstack(self.parallelize_batches(self.procs / 2, self.batch_apply_func, texts, [clf.predict]))[0]
        results= self.parallelize_batches(self.procs / 2, batch_predict, texts, [clf])
        if self.use_sc== True:  return results
        return sp.vstack(results)[0]

    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.iteritems())

    def __setstate__(self, params):
        for key in params:  setattr(self, key, params[key])
