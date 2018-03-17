import randomstate.prng.xoroshiro128plus as rnd
import numpy as np
import time
import multiprocessing
from contextlib import contextmanager
from functools import partial
from multiprocessing.pool import ThreadPool
import itertools
from scipy.sparse import csr_matrix

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def shuffle(*objects, seed=0):
    #Faster than inplace, but uses more memory
    shuffled= rnd.RandomState(seed).permutation(len(objects[0]))
    return [x[shuffled] for x in objects]

def inplace_shuffle(*objects, seed=0):
    #Slower than shuffle, but uses no extra memory
    rand = rnd.RandomState()
    for x in objects:
        rand.seed(seed)
        rand.shuffle(x)

def inplace_shuffle_threaded(*objects, threads= 0, seed=0):
    #Faster than inplace for very large array sizes, > 10000000
    if threads== 0:  threads= min(len(objects), multiprocessing.cpu_count())
    with ThreadPool(processes=threads) as pool:
        pool.map(partial(inplace_shuffle, seed=seed), objects)

def indlist2csrmatrix(indlist, datalist= None, shape= None):
    #Convert a list indicator lists to a scipy.sparse.csr_matrix
    indptr= [0]
    c= 0
    for x in indlist:
        c+= len(x)
        indptr.append(c)
    indices = list(itertools.chain.from_iterable(indlist))
    if datalist!=None:
        data= list(itertools.chain.from_iterable(datalist))
    else:
        data= np.ones(len(indices), dtype=np.float64)
    if shape==None:  shape= (len(indlist), max(indices))
    X= csr_matrix((data, indices, indptr), shape=shape)
    return X