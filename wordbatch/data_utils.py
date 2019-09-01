import randomgen
import numpy as np
import time
import multiprocessing
from contextlib import contextmanager
from functools import partial
from multiprocessing.pool import ThreadPool
import itertools
import scipy.sparse as ssp

# @contextmanager
# def timer(name):
#     t0 = time.time()
#     yield
#     print(name + " done in " + str(time.time() - t0) + "s")
#
# def shuffle(*objects, seed=0):
#     #Faster than inplace, but uses more memory
#     if isinstance(objects[0], ssp.base.spmatrix):  lenn= objects[0].shape[0]
#     else: lenn= len(objects[0])
#     shuffled= randomgen.xoroshiro128.Xoroshiro128(seed).generator.permutation(lenn)
#     return [[x[z] for z in shuffled] if type(x)==list else x[shuffled] for x in objects]
#
# def inplace_shuffle(*objects, seed=0):
#     #Slower than shuffle, but uses no extra memory
#     rand = randomgen.xoroshiro128.Xoroshiro128(seed).generator
#     for x in objects:
#         rand.seed(seed)
#         rand.shuffle(x)
#
# def inplace_shuffle_threaded(*objects, threads= 0, seed=0):
#     #Faster than inplace for very large array sizes, > 10000000
#     if threads== 0:  threads= min(len(objects), multiprocessing.cpu_count())
#     with ThreadPool(processes=threads) as pool:
#         pool.map(partial(inplace_shuffle, seed=seed), objects)

def indlist2csrmatrix(indlist, datalist= None, shape= None):
    #Convert a list of indicator lists to a scipy.sparse.csr_matrix
    indptr= [0]
    c= 0
    for x in indlist:
        c+= len(x)
        indptr.append(c)
    indices = list(itertools.chain.from_iterable(indlist))
    if datalist is not None:
        data= list(itertools.chain.from_iterable(datalist))
    else:
        data= np.ones(len(indices), dtype=np.float64)
    if shape==None:  shape= (len(indlist), max(indices)+1)
    X= ssp.csr_matrix((data, indices, indptr), shape=shape)
    return X

# x= np.array(range(10000000))
# y= np.array(range(10000000))
#
# print(x)
# print(y)
#
# with timer('shuffle'):
#     for z in range(10):
#         x, y= shuffle(x,y)
# print(x)
# print(y)
#
# with timer('inplace_shuffle'):
#     for z in range(10):
#         inplace_shuffle(x,y)
# print(x)
# print(y)
#
# with timer('inplace_shuffle_threaded'):
#     for z in range(10):
#         inplace_shuffle_threaded(x,y)
# print(x)
# print(y)
#
# from sklearn.utils import shuffle as shuffle2
# with timer('sklearn_shuffle'):
#     for z in range(10):
#         x, y= shuffle2(x,y)
# print(x)
# print(y)
