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

cdef double predict_single(int* inds, double* vals, int lenn, double L1, double L2, double alpha, double beta,
                double[::1] w, double[::1] z, double[::1] n, bint bias_term, int threads) nogil:
    cdef int i, ii, lenn2= lenn
    cdef double sign, v
    cdef double e= 0.0
    if bias_term:  lenn2+= 1
    for ii in prange(lenn2, nogil=True, num_threads= threads):
        if ii!=lenn:
            i= inds[ii]
            v= vals[ii]
        else:
            i= 0
            v= 1.0
        sign = -1.0 if z[i] < 0 else 1.0
        if sign * z[i] <= L1:  w[i] = 0.0
        else:
            w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
            e += w[i] * v
    return e

cdef void update_single(int* inds, double* vals, int lenn, double e, double alpha, double[::1] w, double[::1] z,
                        double[::1] n, bint bias_term, int threads) nogil:
    cdef int i, ii, lenn2= lenn
    cdef double g, g2, v
    if bias_term:  lenn2+= 1
    for ii in prange(lenn2, nogil=True, num_threads= threads):
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
    cdef double[::1] w
    cdef double[::1] z
    cdef double[::1] n
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

    def predict(self, X, bint bias_term= True, int threads= 0):
        if threads==0:  threads= self.threads
        if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
# return self.predict_f(X, np.ascontiguousarray(X.data), np.ascontiguousarray(X.indices),
#               np.ascontiguousarray(X.indptr), threads)
        return self.predict_f(X.data, X.indices, X.indptr, bias_term, threads)

    def predict_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
                    np.ndarray[int, ndim=1, mode='c'] X_indices,
                    np.ndarray[int, ndim=1, mode='c'] X_indptr, bint bias_term, int threads):
        p= np.zeros(X_indptr.shape[0]-1, dtype= np.float64)
        cdef int lenn, row_count= X_indptr.shape[0]-1, row, ptr

        cdef int* inds2, indptr2
        cdef double* vals2
        for row in range(row_count):
            ptr= X_indptr[row]
            lenn= X_indptr[row + 1] - ptr
            inds= <int*> X_indices.data + ptr
            vals= <double*> X_data.data + ptr
            p[row]= inv_link_f(predict_single(inds, vals, lenn, self.L1, self.L2, self.alpha, self.beta, self.w, self.z,
                                             self.n, bias_term, threads), self.inv_link)
        return p

    def fit(self, X, y, bint bias_term= True, int threads= 0):
        if threads == 0:  threads= self.threads
        if type(X) != ssp.csr.csr_matrix:  X = ssp.csr_matrix(X, dtype=np.float64)
        if type(y) != np.array:  y = np.array(y, dtype=np.float64)
        # self.fit_f(X, np.ascontiguousarray(X.data), np.ascontiguousarray(X.indices),
        #           np.ascontiguousarray(X.indptr), y, threads)
        self.fit_f(X.data, X.indices, X.indptr, y, threads, bias_term)

    def fit_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
                    np.ndarray[int, ndim=1, mode='c'] X_indices,
                    np.ndarray[int, ndim=1, mode='c'] X_indptr, y, bint bias_term, int threads):
        cdef double alpha= self.alpha, beta= self.beta, L1= self.L1, L2= self.L2
        cdef double[::1] w= self.w, z= self.z, n= self.n, ys= y
        cdef int lenn, ptr, row_count= X_indptr.shape[0]-1, row

        cdef int* inds, indptr
        cdef double* vals

        for iters in range(self.iters):
            for row in range(row_count):
                ptr= X_indptr[row]
                lenn= X_indptr[row+1]-ptr
                inds= <int*> X_indices.data+ptr
                vals= <double*> X_data.data+ptr
                update_single(inds, vals, lenn,
                              inv_link_f(predict_single(inds, vals, lenn, L1, L2, alpha, beta, w, z, n, bias_term,
                                                        threads),
                                         self.inv_link)-ys[row], alpha, w, z, n, bias_term, threads)

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
