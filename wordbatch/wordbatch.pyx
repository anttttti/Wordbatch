#!python
#cython: boundscheck=False, infer_types=True, wraparound=False
import os
import multiprocessing
import copy_reg
import types
from sklearn.utils.murmurhash import murmurhash3_32
from sklearn.feature_extraction.text import HashingVectorizer
from contextlib import closing
from collections import Counter
import operator
#from nltk.metrics import edit_distance
import Levenshtein #python-Levenshtein
import scipy.sparse as ssp
import scipy as sp
import numpy as np
import gzip
import lz4framed
import array
import sys
import random
import time
import re

from cpython cimport array
cimport cython
from libc.stdlib cimport abs
from libc.math cimport log
cimport numpy as np
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

np.import_array()

cdef extern:
    void MurmurHash3_x86_32(void *key, int len, np.uint32_t seed, void *out)

cpdef np.int32_t murmurhash3_bytes_s32(bytes key, unsigned int seed):
    cdef np.int32_t out
    MurmurHash3_x86_32(<char*> key, len(key), seed, &out)
    return out

def _pickle_method(m):
    if m.im_self is None:  return getattr, (m.im_self.__class__, m.im_func.__name__)
    else:  return getattr, (m.im_self, m.im_func.__name__)
copy_reg.pickle(types.MethodType, _pickle_method)

def batch_get_dfs(args):
    dft= Counter()
    for text in args[0]:
        for word in set(text.split()):  dft[word]+= 1
    return dft

def batch_normalize_texts(args):
    normalize_text= args[1]
    return [normalize_text(text) for text in args[0]]

def correct_spelling(word, dft, spell_index, spellcor_count, spellcor_dist):
    #T. Bocek, E. Hunt, B. Stiller: Fast Similarity Search in Large Dictionaries, 2007
    if dft.get(word, 0)>spellcor_count or len(word)<3:  return word
    max_count= -100000000000000
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
        if score > max_count:
            max_count= score
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

def save_to_lz4(file, input, dtype, level= 0):
    with open(file, 'wb') as f:  f.write(lz4framed.compress(np.array(input, dtype=dtype).tostring(), level))

def load_from_lz4(file, dtype):
    with open(file, 'rb') as f:  input= np.fromstring(lz4framed.decompress(f.read()), dtype=dtype)
    return input

non_alphanums= re.compile('[^A-Za-z0-9]+')
def default_normalize_text(text):
    return " ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split()] if len(x)>1])

class WordBatch(object):
    def __init__(self, normalize_text= default_normalize_text, spellcor_count=0, spellcor_dist= 2, n_words= 10000000,
                 min_count= 0, max_count= 1.0, raw_min_count= 0, procs= 0, verbose= 1, minibatch_size= 20000,
                 stemmer= None, pos_tagger= None, extractors=[("wordbag",{})], timeout= 300):
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
        self.raw_min_count= raw_min_count
        self.pos_tagger= pos_tagger

        self.doc_count= 0
        self.n_words= n_words
        self.min_count= min_count
        self.max_count= max_count

        for x in range(len(extractors)):
            if type(extractors[x])!=tuple and type(extractors[x])!=list:
                extractor= extractors[x]
                fea_cfg= {}
            else:
                extractor= extractors[x][0]
                fea_cfg= extractors[x][1]
            extractors[x]= extractor(self, fea_cfg)
        self.extractors= extractors

    def update_dictionary(self, texts, dft, dictionary, min_count):
        self.doc_count+= len(texts)
        for dft2 in self.parallelize_batches(self.procs, batch_get_dfs, texts, []): dft.update(dft2)

        if dictionary!=None:
            sorted_dft = sorted(list(dft.items()), key=operator.itemgetter(1), reverse=True)
            for word, count in sorted_dft:
                if len(dictionary)>= self.n_words: break
                if word in dictionary:  continue
                if type(self.min_count)==type(1):
                    if count<self.min_count:  continue
                else:
                    if float(count)/self.doc_count < self.min_count:  continue
                if type(self.max_count)==type(1):
                    if count > self.max_count:  continue
                else:
                    if float(count)/self.doc_count > self.max_count:  continue
                dictionary[word] = len(dictionary)
                if self.verbose>1: print "Add word to dictionary:", word, dft[word], dictionary[word]

        if min_count>0:
            if self.verbose>2: print "Document Frequency Table size:", len(dft)
            if type(min_count) == type(1):
                for word in list(dft.keys()):
                    if dft[word]<min_count:  dft.pop(word)
            else:
                for word in list(dft.keys()):
                    if float(dft[word])/self.doc_count < min_count:  dft.pop(word)
            if self.verbose > 2: print "Document Frequency Table pruned size:", len(dft)

    def normalize_texts(self, texts):
        return [item for sublist in self.parallelize_batches(self.procs, batch_normalize_texts, texts,
                                     [self.normalize_text]) for item in sublist]

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
        texts= [item for sublist in self.parallelize_batches(self.procs, batch_correct_spellings,
                                                              texts, [corrs, self.pos_tagger])  for item in sublist]
        return texts

    def fit(self, texts):
        if self.verbose > 0:  print "Normalize text"
        if self.normalize_text != None:  texts= self.normalize_texts(texts)
        if self.spellcor_count> 0 or self.stemmer!=None:
            if self.verbose > 0:  print "Update raw dfts"
            self.update_dictionary(texts, self.raw_dft, None, self.raw_min_count)
            if self.verbose > 0:  print "Normalize wordforms"
            texts= self.normalize_wordforms(texts)
        if not(self.dictionary_freeze):  self.update_dictionary(texts, self.dft, self.dictionary, self.min_count)
        return texts

    def fit_transform(self, texts, labels=None):  return self.transform(texts, labels, cache_file= None)

    def partial_fit(self, texts, labels=None):  return self.fit(texts, labels)

    def transform(self, texts, labels= None, extractors= None, cache_features= None):
        if extractors== None:  extractors= self.extractors
        if cache_features!=None and os.path.exists(cache_features + "_0.lz4"):
            features= [extractors[x].load_features(cache_features + "_" + str(x)+".lz4")
                        for x in range(len(extractors))]
            if len(extractors)== 1:  features= features[0]
            return features
        texts= self.fit(texts)
        if extractors!= []:  features= [extractor.transform(texts) for extractor in extractors]
        else:  features= texts
        if cache_features!=None:
            for x in range(len(extractors)):
                extractors[x].save_features(cache_features + "_" + str(x)+".lz4", features[x])
        if len(extractors) == 1:  features= features[0]
        return features

    def parallelize_batches(self, procs, task, texts, argss, method="multiprocessing", timeout=-1):
        if timeout==-1:  timeout= self.timeout
        cdef int attempt= 0, start= 0, len_texts= 0, minibatch_size= self.minibatch_size
        paral_params= []
        if type(texts) is list or type(texts) is tuple:  len_texts= len(texts)
        else:
            len_texts= texts.shape[0]
            if minibatch_size> len_texts: minibatch_size= len_texts
        while start < len_texts:
            paral_params.append([texts[start:start + minibatch_size]] + argss)
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

    def batch_apply_func(self, args):
        fnc= args[1]
        return fnc(args[0])

    def predict_parallel(self, texts, clf):
        return sp.vstack(self.parallelize_batches(self.procs / 2, self.batch_apply_func, texts, [clf.predict]))[0]

cdef class TextRow:
    cdef list indices, data
    cdef dict fea_weights

    def __init__(self):
        self.indices= []
        self.data= []
        self.fea_weights= {}

    cdef append(self, int index, int value, float weight):
        self.indices.append(index)
        self.data.append(value)
        self.fea_weights[index]= weight

class WordBag():
    def __init__(self, wb, fea_cfg):
        self.wb= wb
        fea_cfg.setdefault("norm", 'l2')
        fea_cfg.setdefault("tf", 'log')
        fea_cfg.setdefault("idf", 0.0)
        fea_cfg.setdefault("hash_ngrams", 0)
        fea_cfg.setdefault("hash_ngrams_weights", [-1.0, -1.0])
        fea_cfg.setdefault("hash_size", 10000000)
        fea_cfg.setdefault("hash_polys_window", 0)
        fea_cfg.setdefault("hash_polys_mindf", 5)
        fea_cfg.setdefault("hash_polys_maxdf", 0.5)
        fea_cfg.setdefault("hash_polys_weight", 0.1)
        for key, value in fea_cfg.items():  setattr(self, key, value)

    def get_wordbag(self, text):
        wb= self.wb
        cdef int fc_hash_ngrams= self.hash_ngrams, word_id, df= 1, df2, hashed, doc_count= wb.doc_count, use_idf= 0
        cdef float idf_lift= 0.0, idf= 1.0, weight, norm= 1.0
        if self.idf!= None:
            use_idf= True
            idf_lift= self.idf
        cdef int fc_hash_size= self.hash_size
        fc_hash_ngrams_weights= self.hash_ngrams_weights
        fc_tf= self.tf
        fc_norm= self.norm
        cdef int fc_hash_polys_window= self.hash_polys_window
        cdef int fc_hash_polys_mindf= self.hash_polys_mindf
        cdef float fc_hash_polys_maxdf= self.hash_polys_maxdf

        text= text.split()
        cdef TextRow textrow= TextRow()
        for x from 0 <= x < len(text):
            word= text[x]
            if not(wb.dictionary_freeze):  df= wb.dft[word]
            if df==0: continue
            if use_idf:
                idf= log(max(1.0, idf_lift + doc_count / df))
                if idf== 0.0:  continue

            if fc_hash_ngrams==0:
               word_id= wb.dictionary.get(word, -1)
               if word_id == -1:  continue
               textrow.append(word_id, 1, idf)

            for y from 0 <= y < min(fc_hash_ngrams, x+1):
                hashed= murmurhash3_bytes_s32((" ".join(text[x-y:x+1])).encode("utf-8"), 0)
                weight= fc_hash_ngrams_weights[y]
                if weight < 0: weight*= -idf
                textrow.append(abs(hashed) % fc_hash_size, (hashed >= 0) * 2 - 1, weight)

            if fc_hash_polys_window!=0:
                if df< fc_hash_polys_mindf or float(df)/wb.doc_count> fc_hash_polys_maxdf:  continue
                for y from max(1, fc_hash_ngrams) <= y < min(fc_hash_polys_window, x+1):
                    word2= text[x-y]
                    df2= wb.dft[word2]
                    if df2< fc_hash_polys_mindf or float(df2)/wb.doc_count> fc_hash_polys_maxdf:  continue
                    hashed= murmurhash3_bytes_s32(word+"#"+word2,0) if word<word2 else \
                                                  murmurhash3_bytes_s32(word2+"#"+word, 0)
                    weight= self.hash_polys_weight
                    #if weight<0.0: weight= np.abs(weight) * 1.0/np.log(1+y)
                    if weight < 0.0: weight= np.abs(weight) * 1.0 / log(1 + y)
                    #print word, word2, df, df2, x, y, abs(hashed) % fc_hash_size, (hashed >= 0) * 2 - 1, weight
                    textrow.append(abs(hashed) % fc_hash_size, (hashed >= 0) * 2 - 1, weight)

        cdef np.int32_t size= len(textrow.data)
        cdef int rowdim= fc_hash_size if (fc_hash_ngrams!=0 or fc_hash_polys_window!=0) else wb.n_words

        wordbag = ssp.csr_matrix((textrow.data, textrow.indices, array.array("i", ([0, size]))),
                                 shape=(1, rowdim), dtype=np.float64)
        wordbag.sum_duplicates()

        if fc_tf== 'log':  wordbag.data= np.log(1.0+np.abs(wordbag.data)) *np.sign(wordbag.data)
        elif fc_tf== 'binary':  np.sign(wordbag.data, out=wordbag.data)
        elif type(fc_tf)== type(1.0):
            wordbag.data= ((fc_tf+1.0)*np.abs(wordbag.data))/(fc_tf+np.abs(wordbag.data))*np.sign(wordbag.data)

        size= wordbag.data.shape[0]
        fea_weights= textrow.fea_weights
        cdef int [:] indices_view= wordbag.indices
        cdef double [:] data_view= wordbag.data

        for x from 0 <= x < size: data_view[x]*= fea_weights[indices_view[x]]

        if fc_norm== 'l0':  norm= size
        elif fc_norm == 'l1':  norm= np.sum(np.abs(data_view))
        elif fc_norm== 'l2':  norm= np.sqrt(np.sum([w*w for w in data_view]))
        if norm != 0.0:  norm = 1.0 / norm
        if fc_norm!=None:  wordbag.data*= norm
        return wordbag

    def batch_get_wordbags(self, args):
        return ssp.vstack([self.get_wordbag(text) for text in args[0]])

    def transform(self, texts):
        if self.wb.verbose > 0:  print "Extract wordbags"
        return ssp.vstack(self.wb.parallelize_batches(int(self.wb.procs / 2),  self.batch_get_wordbags, texts, []))

    def save_features(self, file, features):
        #with open(file, 'wb') as f:  f.write(lz4framed.compress(np.array(features.indptr, dtype=int).tostring()))
        #with open(file+".i", 'wb') as f:  f.write(lz4framed.compress(np.array(features.indices, dtype=int).tostring()))
        #with open(file+".d", 'wb') as f:  f.write(lz4framed.compress(np.array(features.data, dtype=np.float64).tostring()))
        save_to_lz4(file, features.indptr, dtype=int)
        save_to_lz4(file+".i", features.indices, dtype=int)
        save_to_lz4(file+".d", features.indices, dtype=np.float64)
        #sp.io.mmwrite(file, features)

    def load_features(self, file):
        #with open(file, 'rb') as f:  indptr= np.fromstring(lz4framed.decompress(f.read()), dtype=int)
        #with open(file+".i", 'rb') as f:  indices= np.fromstring(lz4framed.decompress(f.read()), dtype=int)
        #with open(file+".d", 'rb') as f:  data= np.fromstring(lz4framed.decompress(f.read()), dtype=np.float64)
        indptr= load_from_lz4(file, int)
        indices= load_from_lz4(file+".i", int)
        data= load_from_lz4(file+".d", np.float64)
        return ssp.csr_matrix((data, indices, indptr))
        #return ssp.csr_matrix(sp.io.mmread(file))

class WordHash():
    def __init__(self, wb, fea_cfg):
        self.wb= wb
        self.hv= HashingVectorizer(**fea_cfg)

    def batch_get_wordhashes(self, args):  return self.hv.transform(args[0])

    def transform(self, texts):
        if self.wb.verbose> 0:  print "Extract wordhashes"
        return ssp.vstack(self.wb.parallelize_batches(int(self.wb.procs / 2), self.batch_get_wordhashes, texts, []))

class WordSeq():
    def __init__(self, wb, fea_cfg):
        self.wb = wb
        fea_cfg.setdefault("seq_maxlen", None)
        fea_cfg.setdefault("seq_padstart", True)
        fea_cfg.setdefault("seq_truncstart", True)
        fea_cfg.setdefault("remove_oovs", False)
        for key, value in fea_cfg.items():  setattr(self, key, value)

    def get_wordseq(self, text):
        wb = self.wb
        dictionary = wb.dictionary
        if self.remove_oovs:
            wordseq = [dictionary.get(word, 0) for word in text.split() if word in dictionary]
        else:
            wordseq = [dictionary.get(word, wb.n_words - 1) for word in text.split()]
        if self.seq_maxlen != None:
            if len(wordseq) > self.seq_maxlen:
                if self.seq_truncstart:  wordseq= wordseq[-self.seq_maxlen:]
                else:  wordseq= wordseq[:self.seq_maxlen]
            else:
                if self.seq_padstart == True:  wordseq= [0] * (self.seq_maxlen - len(wordseq)) + wordseq
                else:  wordseq+= [0] * (wb.seq_maxlen - len(wordseq))
        return wordseq

    def batch_get_wordseqs(self, args):
        return [self.get_wordseq(text) for text in args[0]]

    def transform(self, texts):
        if self.wb.verbose > 0:  print "Extract wordseqs"
        return [item for sublist in self.wb.parallelize_batches(self.wb.procs, self.batch_get_wordseqs, texts, [])
                for item in sublist]

class WordVec():
    def __init__(self, wb, fea_cfg):
        self.wb = wb
        if "w2v" not in fea_cfg: fea_cfg["w2v"] = self.load_w2v(fea_cfg["wordvec_file"], fea_cfg["normalize_text"])
        fea_cfg.setdefault("w2v_dim", len(fea_cfg["w2v"].values()[0]))
        fea_cfg.setdefault("merge_vectors", "mean")
        for key, value in fea_cfg.items():  setattr(self, key, value)

    def load_w2v(self, w2v_file, normalize_text=None, stemmer=None):
        w2v= {}
        if w2v_file.endswith(".gz"):  opn= gzip.open
        else:  opn= open
        for line in opn(w2v_file):
            line = line.decode("ISO-8859-1").strip().split(" ", 1)
            vec= np.array([np.float64(x) for x in line[1].split(" ")])
            if len(vec)<2: continue
            word= line[0]
            if normalize_text!=None:  word= normalize_text(word)
            if stemmer!=None: word= stemmer.stem(word)
            w2v[word]= vec
        return w2v

    def get_wordvec(self, text):
        text= text.split()
        if len(text)==0:  return np.zeros(self.w2v_dim)
        w2v= self.w2v
        vecs= []
        for word in text:
            if word in w2v:  vecs.append(w2v[word])
            else:  vecs.append(np.zeros(self.w2v_dim))
        if self.merge_vectors=="mean":
            mean_vec= np.zeros(self.w2v_dim)
            for vec in vecs:  mean_vec+= vec/ len(vecs)
            return mean_vec
        return vecs

    def batch_get_wordvecs(self, args):  return [self.get_wordvec(text) for text in args[0]]

    def transform(self, texts):
        if self.wb.verbose > 0:  print "Extract wordvecs"
        return [item for sublist in self.wb.parallelize_batches(self.wb.procs, self.batch_get_wordvecs, texts, [])
                for item in sublist]
