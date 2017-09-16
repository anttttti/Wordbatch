# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
import gzip
cimport cython
from cpython cimport array
import scipy.sparse as ssp
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, log, fmax, fmin, sqrt, fabs
import multiprocessing
import sys
if sys.version_info.major == 3:
	import pickle as pkl
else:
	import cPickle as pkl

np.import_array()

cdef double inv_link_f(double e, int inv_link) nogil:
	if inv_link==1:  return 1.0 / (1.0 + exp(-fmax(fmin(e, 35.0), -35.0)))
	return e

cdef double predict_single(int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
				double[:] w, double[:] z, double[:] n, bint bias_term, int threads) nogil:
	cdef int i, ii
	cdef double sign, zi, wi
	cdef double e= 0.0
	if bias_term:
		wi= w[0]= -z[0] / ((beta+sqrt(n[0])) * ialpha)
		e += wi

	for ii in prange(lenn, nogil=True, num_threads= threads):
		i= inds[ii]
		zi= z[i]
		sign = -1.0 if zi < 0 else 1.0
		if sign * zi  > L1:
			wi= w[i] = (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2)
			e+= wi * vals[ii]
		else:  w[i] = 0.0
	return e

cdef void update_single(int* inds, double* vals, int lenn, double e, double ialpha, double[:] w, double[:] z,
						double[:] n, bint bias_term, int threads) nogil:
	cdef int i, ii
	cdef double g, g2, ni
	if bias_term:
		g2= e ** 2
		ni= n[0]
		z[0]+= e - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[0]
		n[0]+= g2

	for ii in prange(lenn, nogil=True, num_threads= threads):
		i= inds[ii]
		g= e * vals[ii]
		g2= g ** 2
		ni= n[i]
		z[i]+= g - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[i]
		n[i]+= g2

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
	cdef bint bias_term

	def __init__(self,
				 double alpha=0.1,
				 double beta=1.0,
				 double L1=1.0,
				 double L2=1.0,
				 unsigned int D=2**25,
				 unsigned int iters=1,
				 int threads= 0,
				 inv_link= "sigmoid",
				 bint bias_term=1):

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
		self.bias_term= bias_term
		self.w= np.zeros((D,), dtype=np.float64)
		self.z= np.zeros((D,), dtype=np.float64)
		self.n= np.zeros((D,), dtype=np.float64)

	def predict(self, X, int threads= 0):
		if threads==0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
		# return self.predict_f(X, np.ascontiguousarray(X.data), np.ascontiguousarray(X.indices),
		#               np.ascontiguousarray(X.indptr), threads)
		return self.predict_f(X.data, X.indices, X.indptr, threads)

	def predict_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr, int threads):
		cdef double ialpha= 1.0/self.alpha, L1= self.L1, beta= self.beta, baL2= beta * ialpha + self.L2
		p= np.zeros(X_indptr.shape[0]-1, dtype= np.float64)
		cdef double[:] w= self.w, z= self.z, n= self.n
		cdef double[:] pp= p
		cdef int lenn, row_count= X_indptr.shape[0]-1, row, ptr
		cdef bint bias_term= self.bias_term
		for row in range(row_count):
			ptr= X_indptr[row]
			lenn= X_indptr[row + 1] - ptr
			inds= <int*> X_indices.data + ptr
			vals= <double*> X_data.data + ptr
			pp[row]= inv_link_f(predict_single(inds, vals, lenn, L1, baL2, ialpha, beta, w, z, n,
											   bias_term, threads), self.inv_link)
		return p

	def fit(self, X, y, int threads= 0, int verbose=0):
		if threads == 0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X = ssp.csr_matrix(X, dtype=np.float64)
		if type(y) != np.array:  y = np.array(y, dtype=np.float64)
		# self.fit_f(X, np.ascontiguousarray(X.data), np.ascontiguousarray(X.indices),
		#           np.ascontiguousarray(X.indptr), y, threads)
		self.fit_f(X.data, X.indices, X.indptr, y, threads, verbose)

	def fit_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr, y, int threads, int verbose):
		cdef double ialpha= 1.0/self.alpha, L1= self.L1, beta= self.beta, baL2= beta * ialpha + self.L2, e, e_total= 0
		cdef double[:] w= self.w, z= self.z, n= self.n, ys= y
		cdef unsigned int lenn, ptr, row_count= X_indptr.shape[0]-1, row, inv_link= self.inv_link, j=0, jj
		cdef bint bias_term= self.bias_term
		cdef int* inds, indptr
		cdef double* vals
		for iter in range(self.iters):
			e_total= 0.0
			for row in range(row_count):
				ptr= X_indptr[row]
				lenn= X_indptr[row+1]-ptr
				inds= <int*> X_indices.data+ptr
				vals= <double*> X_data.data+ptr
				e= inv_link_f(
				predict_single(inds, vals, lenn, L1, baL2, ialpha, beta, w, z, n, bias_term,
														threads), inv_link)-ys[row]
				update_single(inds, vals, lenn, e, ialpha, w, z, n, bias_term, threads)
				e_total+= fabs(e)
			if verbose > 0:  print "Total e:", e_total

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
