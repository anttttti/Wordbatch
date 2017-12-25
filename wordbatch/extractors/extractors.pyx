#!python
#cython: boundscheck=False, infer_types=True, wraparound=False
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import types
#from sklearn.utils.murmurhash import murmurhash3_32
from sklearn.feature_extraction.text import HashingVectorizer
#from nltk.metrics import edit_distance
import scipy.sparse as ssp
import scipy as sp
import numpy as np
import gzip
import lz4framed
import array
import sys
if sys.version_info.major == 3:
    import copyreg as copy_reg
else:
    import copy_reg

from cpython cimport array
cimport cython
from libc.stdlib cimport abs
from libc.math cimport log
cimport numpy as np

np.import_array()

cdef extern:
    void MurmurHash3_x86_32(void *key, int len, np.uint32_t seed, void *out)

cpdef np.int32_t murmurhash3_bytes_s32(bytes key, unsigned int seed= 0):
    cdef np.int32_t out
    MurmurHash3_x86_32(<char*> key, len(key), seed, &out)
    return out

def _pickle_method(m):
    if sys.version_info.major == 3:
        if m.im_self is None:  return getattr, (m.im_self.__class__, m.im_func.__name__)
        else:  return getattr, (m.im_self, m.im_func.__name__)
    else:
        if m.__self__ is None:  return getattr, (m.__self__.__class__, m.__func__.__name__)
        else:  return getattr, (m.__self__, m.__func__.__name__)
copy_reg.pickle(types.MethodType, _pickle_method)

def save_to_lz4(file, input, dtype, level= 0):
    with open(file, 'wb') as f:  f.write(lz4framed.compress(np.array(input, dtype=dtype).tostring(), level))

def load_from_lz4(file, dtype):
    with open(file, 'rb') as f:  input= np.fromstring(lz4framed.decompress(f.read()), dtype=dtype)
    return input

def csr_to_lz4(file, features):
    save_to_lz4(file, features.indptr, dtype=int)
    save_to_lz4(file+".i", features.indices, dtype=int)
    save_to_lz4(file+".d", features.data, dtype=np.float64)

def lz4_to_csr(file):
    indptr= load_from_lz4(file, int)
    indices= load_from_lz4(file+".i", int)
    data= load_from_lz4(file+".d", np.float64)
    return ssp.csr_matrix((data, indices, indptr))

def batch_transform(args):
    return args[1].batch_transform(args[0])

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

class WordBag:
    def __init__(self, wb, fea_cfg):
        self.wb= wb
        fea_cfg.setdefault("norm", 'l2')
        fea_cfg.setdefault("tf", 'log')
        fea_cfg.setdefault("idf", 0.0)
        fea_cfg.setdefault("hash_ngrams", 0)
        fea_cfg.setdefault("hash_ngrams_weights", None)
        fea_cfg.setdefault("hash_size", 10000000)
        fea_cfg.setdefault("hash_polys_window", 0)
        fea_cfg.setdefault("hash_polys_mindf", 5)
        fea_cfg.setdefault("hash_polys_maxdf", 0.5)
        fea_cfg.setdefault("hash_polys_weight", 0.1)
        fea_cfg.setdefault("seed", 0)
        for key, value in fea_cfg.items():  setattr(self, key, value)
        if self.hash_ngrams_weights==None: self.hash_ngrams_weights= [1.0 for x in range(self.hash_ngrams)]

    def transform_single(self, text):
        wb= self.wb
        cdef int fc_hash_ngrams= self.hash_ngrams, word_id, df= 1, df2, hashed, doc_count= wb.doc_count, \
                                    use_idf= 0, seed= self.seed
        cdef float idf_lift= 0.0, idf= 1.0, weight, norm= 1.0, norm_idf= 1.0
        if self.idf!= None:
            use_idf= True
            idf_lift= self.idf
            norm_idf= 1.0 / log(max(1.0, idf_lift + doc_count))
        cdef int fc_hash_size= self.hash_size
        fc_hash_ngrams_weights= self.hash_ngrams_weights
        fc_tf= self.tf
        fc_norm= self.norm
        cdef int fc_hash_polys_window= self.hash_polys_window, fc_hash_polys_mindf= self.hash_polys_mindf
        cdef float fc_hash_polys_maxdf= self.hash_polys_maxdf, fc_hash_polys_weight= self.hash_polys_weight

        text= text.split(" ")
        cdef TextRow textrow= TextRow()
        for x from 0 <= x < len(text):
            word= text[x]
            if not(wb.dictionary_freeze):  df= wb.dft[word]
            if df==0: continue
            if use_idf:
                idf= log(max(1.0, idf_lift + doc_count / df)) * norm_idf
                if idf== 0.0:  continue

            if fc_hash_ngrams==0:
               word_id= wb.dictionary.get(word, -1)
               if word_id== -1:  continue
               textrow.append(word_id, 1, idf)

            for y from 0 <= y < min(fc_hash_ngrams, x+1):
                hashed= murmurhash3_bytes_s32((" ".join(text[x-y:x+1])).encode("utf-8"), seed)
                weight= fc_hash_ngrams_weights[y]
                if weight < 0: weight*= -idf
                textrow.append(abs(hashed) % fc_hash_size, (hashed >= 0) * 2 - 1, weight)

            if fc_hash_polys_window!=0:
                if doc_count!=0:
                    if df< fc_hash_polys_mindf or float(df)/wb.doc_count> fc_hash_polys_maxdf:  continue
                #for y from max(1, fc_hash_ngrams) <= y < min(fc_hash_polys_window, x+1):
                for y from 1 <= y < min(fc_hash_polys_window, x+1):
                    word2= text[x-y]
                    if doc_count!=0:
                        df2= wb.dft[word2]
                        if df2< fc_hash_polys_mindf or float(df2)/wb.doc_count> fc_hash_polys_maxdf:  continue
                    hashed= murmurhash3_bytes_s32((word+"#"+word2).encode("utf-8"), seed) if word<word2 \
                        else murmurhash3_bytes_s32((word2+"#"+word).encode("utf-8"), seed)
                    weight= fc_hash_polys_weight
                    #if weight<0.0: weight= np.abs(weight) * 1.0/np.log(1+y)
                    if weight < 0.0: weight= np.abs(weight) * 1.0 / log(1 + y)
                    #print word, word2, df, df2, x, y, abs(hashed) % fc_hash_size, (hashed >= 0) * 2 - 1, weight
                    textrow.append(abs(hashed) % fc_hash_size, (hashed >= 0) * 2 - 1, weight)

        cdef np.int32_t size= len(textrow.data)
        cdef int rowdim= fc_hash_size if (fc_hash_ngrams!=0 or fc_hash_polys_window!=0) else wb.n_words

        wordbag= ssp.csr_matrix((textrow.data, textrow.indices, array.array("i", ([0, size]))),
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
        elif fc_norm== 'l1':  norm= np.sum(np.abs(data_view))
        elif fc_norm== 'l2':  norm= np.sqrt(np.sum([w*w for w in data_view]))
        if norm != 0.0:  norm= 1.0 / norm
        if fc_norm!=None:  wordbag.data*= norm
        return wordbag

    def batch_transform(self, texts):
        return ssp.vstack([self.transform_single(text) for text in texts])

    def transform(self, texts, input_split= False, merge_output= True):
        if self.wb.verbose > 0:  print("Extract wordbags")
        return self.wb.parallelize_batches(batch_transform, texts, [self], input_split= input_split,
                                           merge_output= merge_output, procs= int(self.wb.batcher.procs / 2))

    def save_features(self, file, features):
        csr_to_lz4(file, features)

    def load_features(self, file):
        return lz4_to_csr(file)

class WordHash:
    def __init__(self, wb, fea_cfg):
        self.wb= wb
        self.hv= HashingVectorizer(**fea_cfg)

    def batch_transform(self, texts):  return self.hv.transform(texts)

    def transform(self, texts, input_split= False, merge_output= True):
        if self.wb.verbose> 0:  print("Extract wordhashes")
        return self.wb.parallelize_batches(batch_transform, texts, [self], input_split= input_split,
                                           merge_output= True, procs= int(self.wb.batcher.procs / 2))

    def save_features(self, file, features):
        csr_to_lz4(file, features)

    def load_features(self, file):
        return lz4_to_csr(file)

class WordSeq:
    def __init__(self, wb, fea_cfg):
        self.wb= wb
        fea_cfg.setdefault("seq_maxlen", None)
        fea_cfg.setdefault("seq_padstart", True)
        fea_cfg.setdefault("seq_truncstart", True)
        fea_cfg.setdefault("remove_oovs", False)
        fea_cfg.setdefault("pad_id", 0)
        fea_cfg.setdefault("oov_id", wb.n_words+1)
        for key, value in fea_cfg.items():  setattr(self, key, value)

    def transform_single(self, text):
        wb= self.wb
        dictionary= wb.dictionary
        if self.remove_oovs:  wordseq= [dictionary[word] for word in text.split(" ") if word in dictionary]
        else:  wordseq= [dictionary.get(word, self.oov_id) for word in text.split(" ")]
        if self.seq_maxlen != None:
            if len(wordseq) > self.seq_maxlen:
                if self.seq_truncstart:  wordseq= wordseq[-self.seq_maxlen:]
                else:  wordseq= wordseq[:self.seq_maxlen]
            else:
                if self.seq_padstart== True:  wordseq= [self.pad_id] * (self.seq_maxlen - len(wordseq)) + wordseq
                else:  wordseq+= [self.pad_id] * (wb.seq_maxlen - len(wordseq))
        return wordseq

    def batch_transform(self, texts):  return [self.transform_single(text) for text in texts]

    def transform(self, texts, input_split= False, merge_output= True):
        if self.wb.verbose > 0:  print("Extract wordseqs")
        return self.wb.parallelize_batches(batch_transform, texts, [self], input_split=input_split,
                                           merge_output=merge_output, procs= int(self.wb.batcher.procs / 2))

    def save_features(self, file, features):
        save_to_lz4(file, features, dtype=int)
        i= 0
        indices= []
        for x in features:
            i+= len(x)
            indices.append(i)
        save_to_lz4(file + ".i", indices, dtype=int)

    def load_features(self, file):
        words= load_from_lz4(file, int).tolist()
        indices= [0]+load_from_lz4(file + ".i", int).tolist()
        return [words[indices[i]:indices[i+1]] for i in range(len(indices)-1)]

class WordVec:
    def __init__(self, wb, fea_cfg):
        self.wb= wb
        fea_cfg.setdefault("normalize_text", None)
        fea_cfg.setdefault("stemmer", None)
        fea_cfg.setdefault("merge_dict", True)
        fea_cfg.setdefault("normalize_dict", False)
        fea_cfg.setdefault("verbose", 0)
        fea_cfg.setdefault("merge_vectors", "mean")
        fea_cfg.setdefault("normalize_merged", "L2")
        for key, value in fea_cfg.items():  setattr(self, key, value)
        self.w2v= self.load_w2v(fea_cfg["wordvec_file"])
        self.w2v_dim= len(list(self.w2v.values())[0])

    def load_w2v(self, w2v_file):
        w2v= {}
        from collections import Counter
        w2v_counts= Counter()
        opn= gzip.open if w2v_file.endswith(".gz") else open
        for line in opn(w2v_file):
            line= line.decode("ISO-8859-1").strip().split(" ", 1)
            vec= np.array([np.float64(x) for x in line[1].split(" ")])
            if len(vec)<2: continue
            word= line[0]
            if self.normalize_text!=None:  word= self.normalize_text(word)
            if self.stemmer!=None:  word= self.stemmer.stem(word)
            if not(self.merge_dict):  w2v[word]= vec
            else:
                w2v_counts[word] += 1
                if word in w2v:
                    w2v[word]+= (vec - w2v[word]) / w2v_counts[word]
                    if self.verbose>0:
                        print("Merged entry:", word, w2v_counts[word])
                else:  w2v[word]= vec
        if self.normalize_dict!=False:
            for word in w2v:
                if self.normalize_dict=="L1":
                    norm= sum(np.abs(w2v[word]))
                else:
                    norm = np.sqrt(sum(w2v[word] **2))
                if norm!=0:
                    w2v[word]/= norm
        return w2v

    def transform_single(self, text):
        text= text.split(" ")
        if len(text)==0:  return np.zeros(self.w2v_dim)
        w2v= self.w2v
        vecs= []
        for word in text:
            if word in w2v:  vecs.append(w2v[word])
            else:  vecs.append(np.zeros(self.w2v_dim))
        if self.merge_vectors!=None:
            if self.merge_vectors=="mean":
                vec= np.mean(vecs, axis=0)
            if self.normalize_merged!=None:
                if self.normalize_merged == "L1":
                    norm = sum(np.abs(vec))
                else:
                    norm = np.sqrt(sum(vec ** 2))
                if norm != 0:
                    vec /= norm
            return vec
        return vecs

    def batch_transform(self, texts):  return [self.transform_single(text) for text in texts]

    def transform(self, texts, input_split= False, merge_output= True):
        if self.wb.verbose > 0:  print("Extract wordvecs")
        return self.wb.parallelize_batches(batch_transform, texts, [self], input_split=input_split,
                                           merge_output=merge_output, procs= int(self.wb.batcher.procs / 2))

class Hstack:
    def __init__(self, wb, fea_cfg):
        self.wb= wb
        t= [x[0](wb, x[1]) for x in fea_cfg]
        self.extractors= list(t)

    def transform_single(self, text):
        return sp.hstack([x.transform_single(text) for x in self.extractors])

    def batch_transform(self, texts):  return [self.transform_single(text) for text in texts]

    def transform(self, texts, input_split= False, merge_output= True):
        if self.wb.verbose> 0:  print("Extract concatenated dense features")
        return self.wb.parallelize_batches(batch_transform, texts, [self], input_split=input_split,
                                           merge_output=merge_output, procs= int(self.wb.batcher.procs / 2))
