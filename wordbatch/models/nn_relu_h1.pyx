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
import randomgen

np.import_array()

cdef double inv_link_f(double e, int inv_link) nogil:
	if inv_link==1:  return 1.0 / (1.0 + exp(-fmax(fmin(e, 35.0), -35.0))) #Sigmoid + logloss
	return e

cdef double predict_single(int* inds, double* vals, int lenn, int D, int D_nn,
				double* w0, double* w1, double* z, int threads) nogil:
	cdef int j, i, ii, DD_nn= D*D_nn
	cdef double p, v, zj
	p= w1[D_nn]
	for j in prange(D_nn, nogil=True, num_threads= threads):
		zj= w0[DD_nn+j]
		for ii in range(lenn):
			zj= zj+ w0[inds[ii]*D_nn+j] * vals[ii]
		if zj<=0:  z[j]= 0
		else:
			z[j]= zj
			p+= w1[j] * zj
	return p

cdef void update_single(int* inds, double* vals, int lenn, int D, int D_nn, double e, double alpha,
						double L2,  double* w0, double* w1, double* z, double* c0, double* c1, int threads) nogil:
	cdef int i, ii, j, DD_nn= D*D_nn, iDnnj
	cdef double dldy= e, dldz, dldw1, dldw0
	w1[D_nn]-= (dldy+ L2 * w1[D_nn]) * alpha
	#for j in prange(D_nn, nogil=True, num_threads=threads):
	for j in range(D_nn):
		if z[j]==0:  continue
		dldw1= dldy * z[j]
		w1[j]-= (dldw1 + L2 * w1[j]) * alpha/(sqrt(c1[j])+1)
		dldz= dldy * w1[j]
		w0[DD_nn+j]-= (dldz+ L2 *w0[DD_nn+j]) * alpha/(sqrt(c1[j])+1)

		#for ii in range(lenn):
		for ii in prange(lenn, nogil=True, num_threads=threads):
			i= inds[ii]
			dldw0= dldz * vals[ii]
			w0[i * D_nn + j]-= (dldw0 + L2 *w0[i * D_nn + j]) * alpha/(sqrt(c0[i])+1)
			c0[i]+= fabs(dldw0)
		c1[j]+= fabs(dldw1)

cdef class NN_ReLU_H1:
	cdef const double[:] w0
	cdef const double[:] w1
	cdef const double[:] z
	cdef const double[:] c0
	cdef const double[:] c1

	cdef unsigned int threads
	cdef unsigned int iters
	cdef int D
	cdef int D_nn
	cdef double init_nn

	cdef double L2
	cdef double alpha
	cdef double e_noise
	cdef double e_clip
	cdef int inv_link
	cdef int seed
	cdef int verbose

	def __init__(self,
				 double alpha=0.1,
				 double L2=0.001,
				 int D=0,
				 int D_nn=30,
				 double init_nn=0.01,
				 double e_noise=0.0001,
				 double e_clip=1.0,
				 unsigned int iters=4,
				 inv_link= "identity",
				 int threads= 0,
				 int seed= 0,
				 int verbose=1):

		self.alpha= alpha
		self.L2= L2
		self.D= D
		self.D_nn= D_nn
		self.init_nn= init_nn
		self.e_noise= e_noise
		self.e_clip= e_clip
		self.iters= iters
		if threads==0:  threads= multiprocessing.cpu_count()-1
		self.threads= threads
		if inv_link=="sigmoid":  self.inv_link= 1
		if inv_link=="identity":  self.inv_link= 0
		self.seed = seed
		self.verbose= verbose
		self.reset()

	def reset(self):
		init_nn= self.init_nn
		D= self.D
		D_nn= self.D_nn
		rand= randomgen.xoroshiro128.Xoroshiro128(seed= self.seed).generator
		self.w0 = (rand.rand((D + 1) * D_nn) - 0.5) * init_nn
		self.w1 = (rand.rand(D_nn + 1) - 0.5) * init_nn
		self.z = np.zeros((D_nn,), dtype=np.float64)
		self.c0 = np.zeros((D,), dtype=np.float64)
		self.c1 = np.zeros((D_nn,), dtype=np.float64)

	def predict(self, X, int threads= 0):
		if threads==0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
		if X.shape[1] != self.D:
			print("Dimension mismatch! self.D=", self.D, "X.shape[1]=", X.shape[1])
		return self.predict_f(X.data, X.indices, X.indptr, threads)

	def predict_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr, int threads):
		cdef double alpha= self.alpha, L2= self.L2
		p= np.zeros(X_indptr.shape[0]-1, dtype= np.float64)
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *z= &self.z[0]
		cdef double[:] pp= p
		cdef int lenn, D= self.D, D_nn= self.D_nn, row_count= X_indptr.shape[0]-1, row, ptr
		for row in range(row_count):
			ptr= X_indptr[row]
			lenn= X_indptr[row + 1] - ptr
			inds= <int*> X_indices.data + ptr
			vals= <double*> X_data.data + ptr
			pp[row]= inv_link_f(predict_single(inds, vals, lenn, D, D_nn, w0, w1, z, threads), self.inv_link)
		return p

	def partial_fit(self, X, y, sample_weight= None, int threads = 0, int seed = 0):
		return self.fit(X, y, sample_weight= sample_weight, threads = threads, seed = seed, reset= False)

	def fit(self, X, y, sample_weight= None, int threads= 0, int seed= 0, reset= True):
		if threads == 0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X = ssp.csr_matrix(X, dtype=np.float64)
		if reset or self.D==0:
			self.D= X.shape[1]
			self.reset()
		elif X.shape[1] != self.D:
			print("Dimension mismatch! self.D=", self.D, "X.shape[1]=", X.shape[1])
		if type(y) != np.array:  y = np.array(y, dtype=np.float64)
		# self.fit_f(X, np.ascontiguousarray(X.data), np.ascontiguousarray(X.indices),
		#           np.ascontiguousarray(X.indptr), y, threads)
		if sample_weight is not None and type(sample_weight) != np.array:
			sample_weight= np.array(sample_weight, dtype=np.float64)
		return self.fit_f(X.data, X.indices, X.indptr, y, sample_weight, threads, seed)

	def fit_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr,
					np.ndarray[double, ndim=1, mode='c'] y,
					sample_weight,
					int threads, int seed):
		cdef double alpha= self.alpha, L2= self.L2, e_noise= self.e_noise, e, e_total= 0, e_clip= self.e_clip, abs_e
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *z= &self.z[0], *c0= &self.c0[0], *c1= &self.c1[0]
		cdef double *ys= <double*> y.data
		cdef unsigned int lenn, D= self.D, D_nn= self.D_nn, ptr, row_count= X_indptr.shape[0]-1, row, \
																			inv_link= self.inv_link, j=0, jj
		cdef int* inds, indptr
		cdef double* vals
		rand= randomgen.xoroshiro128.Xoroshiro128(seed= seed).generator

		for iter in range(self.iters):
			e_total= 0.0
			for row in range(row_count):
				ptr= X_indptr[row]
				lenn= X_indptr[row+1]-ptr
				inds= <int*> X_indices.data+ptr
				vals= <double*> X_data.data+ptr
				e= inv_link_f(predict_single(inds, vals, lenn, D, D_nn, w0, w1, z, threads), self.inv_link) -ys[row]
				abs_e= fabs(e)
				e_total+= abs_e
				e += (rand.rand() - 0.5) * e_noise
				if abs_e> e_clip:
					if e>0:  e= e_clip
					else:  e= -e_clip
				if sample_weight is not None:
					e*= sample_weight[row]
				update_single(inds, vals, lenn, D, D_nn, e, alpha, L2, w0, w1, z, c0, c1, threads)
			if self.verbose > 0:  print "Total e:", e_total
		return self

	def predict_layer(self, X, int layer, int threads= 0):
		if threads==0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
		return self.predict_layer_f(X.data, X.indices, X.indptr, layer, threads)

	def predict_layer_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr, int layer, int threads):
		cdef double alpha= self.alpha, L2= self.L2
		p = np.zeros(((X_indptr.shape[0] - 1), self.D_nn), dtype=np.float64)
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *z= &self.z[0]
		cdef double[:,:] pp= p
		cdef unsigned int lenn, D= self.D, D_nn= self.D_nn, row_count= X_indptr.shape[0]-1, row, ptr
		for row in range(row_count):
			ptr= X_indptr[row]
			lenn= X_indptr[row + 1] - ptr
			inds= <int*> X_indices.data + ptr
			vals= <double*> X_data.data + ptr
			predict_single(inds, vals, lenn, D, D_nn, w0, w1, &pp[row][0], threads)
		return p

	def __getstate__(self):
		return (self.alpha,
			self.L2,
			self.e_noise,
			self.e_clip,
			self.init_nn,
			self.D,
			self.D_nn,
			self.iters,
			self.threads,
			np.asarray(self.w0),
			np.asarray(self.w1),
			np.asarray(self.z),
			np.asarray(self.c0),
			np.asarray(self.c1),
			self.inv_link,
			self.seed,
			self.verbose)

	def __setstate__(self, params):
		(self.alpha,
		 self.L2,
		 self.e_noise,
		 self.e_clip,
		 self.init_nn,
		 self.D,
		 self.D_nn,
		 self.iters,
		 self.threads,
		 self.w0,
		 self.w1,
		 self.z,
		 self.c0,
		 self.c1,
		 self.inv_link,
		 self.seed,
		 self.verbose)= params
