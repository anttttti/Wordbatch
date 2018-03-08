import randomstate.prng.xoroshiro128plus as rnd
import numpy as np
import time
import multiprocessing
from contextlib import contextmanager
from functools import partial
from multiprocessing.pool import ThreadPool

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

