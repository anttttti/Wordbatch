# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
import cPickle as pkl
import gzip
cimport cython
from cpython cimport array
import scipy.sparse as ssp
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, log, fmax, fmin, sqrt
import multiprocessing

np.import_array()

cdef double inv_link_f(double e, int inv_link):
    if inv_link==1:  return 1.0 / (1.0 + exp(-fmax(fmin(e, 35.0), -35.0)))
    return e

cdef double predict_single(int[:] inds, double[:] vals, int lenn, double L1, double L2, double alpha, double beta,
                double[:] w, double[:] z, double[:] n, int threads) nogil:
    cdef int i, ii
    cdef double sign, v
    cdef double e= 0.0
    for ii in prange(lenn+1, nogil=True, num_threads= threads):
        if ii!=lenn:
            i= inds[ii]
            v= vals[ii]
        else:
            i= 0
            v= 1.0
        sign = -1.0 if z[i] < 0 else 1.0
        if sign * z[i] <= L1:  w[i] = 0.0
        else:  w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
        e += w[i] * v
    return e

cdef void update_single(int[:] inds, double[:] vals, int lenn, double e, double alpha, double[:] w, double[:] z,
                        double[:] n, int threads) nogil:
    cdef int i, ii
    cdef double g, g2, v
    for ii in prange(lenn+1, nogil=True, num_threads= threads):
        if ii!=lenn:
            i= inds[ii]
            v= vals[ii]
        else:
            i= 0
            v= 1.0
        g = e * v
        g2 = g * g
        z[i] += g - ((sqrt(n[i] + g2) - sqrt(n[i])) / alpha) * w[i]
        n[i] += g2

cdef class FTRL:
    cdef double[:] w
    cdef double[:] z
    cdef double[:] n

    cdef unsigned int threads
    cdef unsigned int iters
    cdef unsigned int D
    cdef double L1
    cdef double L2
    cdef double alpha
    cdef double beta
    cdef int inv_link

    def __init__(self,
                 double alpha=0.1,
                 double beta=1.0,
                 double L1=1.0,
                 double L2=1.0,
                 unsigned int D=2**25,
                 unsigned int iters=1,
                 int threads= 0,
                 inv_link= "sigmoid"):

        self.alpha= alpha
        self.beta= beta
        self.L1= L1
        self.L2= L2
        self.D= D
        self.iters= iters
        if threads==0:  threads= multiprocessing.cpu_count()-1
        self.threads= threads
        if inv_link=="sigmoid":  self.inv_link= 1
        if inv_link=="identity":  self.inv_link= 0
        self.w= np.zeros((self.D,), dtype=np.float64)
        self.z= np.zeros((self.D,), dtype=np.float64)
        self.n= np.zeros((self.D,), dtype=np.float64)

    def predict(self, X, int threads= 0):
        if threads==0:  threads= self.threads
        if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
        p= np.zeros((X.shape[0], ), dtype=np.float64)
        cdef double[:] pp= p, Xvals= X.data, vals
        cdef int i, lenn, row_count= X.shape[0], row
        cdef int[:] Xindptr= X.indptr, Xinds= X.indices, inds

        for row in range(row_count):
            ptr= Xindptr[row]
            lenn= Xindptr[row + 1] - ptr
            inds= Xinds[ptr:ptr + lenn]
            vals= Xvals[ptr:ptr + lenn]
            pp[row]= inv_link_f(predict_single(inds, vals, lenn, self.L1, self.L2, self.alpha, self.beta, self.w, self.z,
                                             self.n, threads), self.inv_link)
        return p

    def fit(self, X, y, int threads= 0):
        if threads == 0:  threads= self.threads
        if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
        if type(y) != np.array:  y= np.array(y, dtype=np.float64)
        cdef double alpha= self.alpha, beta= self.beta, L1= self.L1, L2= self.L2
        cdef double[:] w= self.w, z= self.z, n= self.n, vals, ys= y, Xvals= X.data
        cdef int lenn, i, ptr, row_count= X.shape[0], row
        cdef int[:] Xindptr= X.indptr, Xinds= X.indices, inds

        for iters in range(self.iters):
            for row in range(row_count):
                ptr= Xindptr[row]
                lenn= Xindptr[row+1]-ptr
                inds= Xinds[ptr:ptr+lenn]
                vals= Xvals[ptr:ptr+lenn]
                update_single(inds, vals, lenn,
                              inv_link_f(predict_single(inds, vals, lenn, L1, L2, alpha, beta, w, z, n, threads),
                                       self.inv_link)-ys[row],
                              alpha, w, z, n, threads)

    def pickle_model(self, filename):
        with gzip.open(filename, 'wb') as model_file:
            pkl.dump(self.get_params(), model_file, protocol=2)

    def unpickle_model(self, filename):
        self.set_params(pkl.load(gzip.open(filename, 'rb')))

    def __getstate__(self):
        return (self.alpha, self.beta, self.L1, self.L2, self.D, self.iters,
                np.asarray(self.w), np.asarray(self.z), np.asarray(self.n), self.inv_link)

    def __setstate__(self, params):
        (self.alpha, self.beta, self.L1, self.L2, self.D, self.iters, self.w, self.z, self.n, self.inv_link)= params
