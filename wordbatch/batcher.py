#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import types
from contextlib import closing
import scipy.sparse as ssp
import random
import pandas as pd
import sys
from math import ceil
if sys.version_info.major == 3:
    import copyreg as copy_reg
else:
    import copy_reg

def _pickle_method(m):
    if sys.version_info.major == 3:
        if m.im_self is None:  return getattr, (m.im_self.__class__, m.im_func.__name__)
        else:  return getattr, (m.im_self, m.im_func.__name__)
    else:
        if m.__self__ is None:  return getattr, (m.__self__.__class__, m.__func__.__name__)
        else:  return getattr, (m.__self__, m.__func__.__name__)
copy_reg.pickle(types.MethodType, _pickle_method)

#def batch_object_call(args):
#    return args[1].predict(args[0])
#    return eval(args[1]+"."+args[2]+'('+args[0]+')')

class Batcher(object):
    def __init__(self, procs= 0, minibatch_size= 20000, timeout= 600, use_sc= False, method= "multiprocessing",
                 verbose= 1):
        if procs==0:  procs= multiprocessing.cpu_count()
        self.procs= procs
        self.verbose= verbose
        self.minibatch_size= minibatch_size
        self.timeout= timeout
        self.use_sc= use_sc
        self.method= method

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
        if data_type == pd.DataFrame:
            data_split = [data.iloc[x * minibatch_size:(x + 1) * minibatch_size] for x in
                          range(int(ceil(len_data / minibatch_size)))]
        else:
            data_split= [data[x* minibatch_size:min(len_data, (x+1)*minibatch_size)]
                         for x in range(int(ceil(len_data/minibatch_size)))]
        return data_split

    def merge_batches(self, data):
        if isinstance(data[0], ssp.csr_matrix):  return ssp.vstack(data)
        return [item for sublist in data for item in sublist]

    def parallelize_batches(self, task, data, args, method=None, timeout=-1, rdd_col= 1, input_split=False,
                            merge_output= True, minibatch_size= None, procs=None):
        if procs==None: procs= self.procs
        if method == None: method= self.method
        if self.verbose > 1:
            print("Parallel task:", task, " Method:", method, " Procs:", self.procs, " input_split:",
                                   input_split)
        if self.use_sc==True:
            def apply_func(batch):  return batch[:rdd_col]+[task([batch[rdd_col]]+args)]+batch[rdd_col+1:]
            results= data.map(apply_func)
            return results

        if self.verbose> 10:
            print("len(data):", len(data), "len(args):", len(args), "[type(x) for x in data]:",
                  [type(x) for x in data], "[type(x) for x in args]:", [type(x) for x in args])

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

    #def object_call_parallel(self, data, object, call):
    #    return self.merge_batches(self.parallelize_batches(self.procs / 2, object_call, data, [object, call]))

    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items())

    def __setstate__(self, params):
        for key in params:  setattr(self, key, params[key])
