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

cdef double predict_single(int* inds, double* vals, int lenn,  int D,  int D_nn,  int D_nn2,
				double* w0, double* w1, double* w2, double* z1, double* z2, int threads) nogil:
	cdef int i, ii, j, k, DD_nn= D*D_nn, DD_nn2= D_nn*D_nn2
	cdef double p, v, z1j, z2k
	p= w2[D_nn2]
	for k in prange(D_nn2, nogil=True, num_threads= threads):
	#for k in range(D_nn2):
		z2k= w1[DD_nn2+k]
		for j in range(D_nn):
			z1j= w0[DD_nn+j]
			for ii in range(lenn):	z1j= z1j+ w0[inds[ii]*D_nn+j] * vals[ii]
			if z1j<0: z1[j]= 0
			else:
				z1[j] = z1j
				z2k= z2k+ w1[j * D_nn2+k] * z1j
		if z2k<0:  z2[k]= 0
		else:
			z2[k]= z2k
			p+= w2[k] * z2k
	return p

cdef void update_single(int* inds, double* vals, int lenn,  int D,  int D_nn,  int D_nn2,
						double e, double alpha, double L2, double* w0, double* w1, double* w2, double* z1,
						double* z2, double* c0, double* c1, double* c2, int threads) nogil:
	cdef int i, ii, j, k, DD_nn= D*D_nn, DD_nn2= D_nn*D_nn2
	cdef double dldy= e, dldz1, dldz2, dldw0, dldw1, dldw2
	w2[D_nn2]-= (dldy + L2 *w2[D_nn2]) * alpha
	for k in range(D_nn2):
		if z2[k]==0:  continue
		dldw2= dldy * z2[k]
		w2[k]-= (dldw2 + L2 * w2[k]) * alpha / (sqrt(c2[k])+1)
		dldz2= dldy * w2[k]
		w1[DD_nn2+k]-= (dldz2 + L2 * w1[DD_nn2 + k]) * alpha / (sqrt(c2[k])+1)
		for j in range(D_nn):
			if z1[j]==0:  continue
			dldw1= dldz2 * z1[j]
			w1[j*D_nn2+k]-= (dldw1 + L2 * w1[j]) * alpha / (sqrt(c1[j])+1)
			dldz1= dldz2 * w1[j*D_nn2+k]
			w0[DD_nn+j]-= (dldz1 + L2 * w0[DD_nn+j]) * alpha /  (sqrt(c1[j])+1)
			for ii in prange(lenn, nogil=True, num_threads= threads):
				i= inds[ii]
				dldw0= dldz1 * vals[ii]
				w0[i*D_nn+j]-= (dldw0 + L2 * w0[i * D_nn + j]) * alpha/(sqrt(c0[i])+1)
				c0[i] += fabs(dldw0)
			c1[j] += fabs(dldw1)
		c2[k] += fabs(dldw2)

cdef class NN_ReLU_H2:
	cdef const double[:] w0
	cdef const double[:] w1
	cdef const double[:] w2
	cdef const double[:] z1
	cdef const double[:] z2
	cdef const double[:] c0
	cdef const double[:] c1
	cdef const double[:] c2

	cdef unsigned int threads
	cdef unsigned int iters
	cdef int D
	cdef int D_nn
	cdef int D_nn2
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
				 double L2=0.00001,
				 int D=0,
				 int D_nn=12,
				 int D_nn2=4,
				 double init_nn=0.01,
				 double e_noise=0.001,
				 double e_clip=1.0,
				 unsigned int iters=3,
				 inv_link= "identity",
				 int threads= 0,
				 int seed= 0,
				 int verbose=1):

		self.alpha= alpha
		self.L2= L2
		self.e_noise= e_noise
		self.D= D
		self.D_nn= D_nn
		self.D_nn2= D_nn2
		self.init_nn= init_nn
		self.e_noise = e_noise
		self.e_clip = e_clip
		self.iters= iters
		if threads==0:  threads= multiprocessing.cpu_count()-1
		self.threads= threads
		if inv_link=="sigmoid":  self.inv_link= 1
		if inv_link=="identity":  self.inv_link= 0
		self.seed = seed
		self.verbose = verbose
		self.reset()

	def reset(self):
		init_nn= self.init_nn
		D= self.D
		D_nn = self.D_nn
		D_nn2 = self.D_nn2
		rand= np.random.Generator(randomgen.xoroshiro128.Xoroshiro128(seed= self.seed))
		self.w0 = (rand.random((D + 1) * D_nn) - 0.5) * init_nn
		self.w1 = (rand.random((D_nn + 1) * D_nn2) - 0.3) * init_nn
		self.w2 = (rand.rand(D_nn2 + 1) - 0.5) * init_nn
		self.z1= np.zeros((D_nn,), dtype=np.float64)
		self.z2= np.zeros((D_nn2,), dtype=np.float64)
		self.c0= np.zeros((D,), dtype=np.float64)
		self.c1= np.zeros((D_nn,), dtype=np.float64)
		self.c2= np.zeros((D_nn2,), dtype=np.float64)

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
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *w2= &self.w2[0], *z1= &self.z1[0], *z2= &self.z2[0]
		cdef double[:] pp= p
		cdef int lenn, D= self.D, D_nn= self.D_nn, D_nn2= self.D_nn2, row_count= X_indptr.shape[0]-1, row, ptr
		for row in range(row_count):
			ptr= X_indptr[row]
			lenn= X_indptr[row + 1] - ptr
			inds= <int*> X_indices.data + ptr
			vals= <double*> X_data.data + ptr
			pp[row]= inv_link_f(predict_single(inds, vals, lenn, D, D_nn, D_nn2, w0, w1, w2, z1, z2, threads), \
					self.inv_link)
		return p

	def partial_fit(self, X, y, int threads = 0, int seed = 0):
		return self.fit(X, y, threads=threads, seed=seed, reset=False)

	def fit(self, X, y, int threads= 0, int seed= 0, reset=True):
		if threads == 0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X = ssp.csr_matrix(X, dtype=np.float64)
		if reset or self.D==0:
			self.D= X.shape[1]
			self.reset()
		elif X.shape[1] != self.D:
			print("Dimension mismatch! self.D=", self.D, "X.shape[1]=", X.shape[1])
		if type(y) != np.array:  y = np.array(y, dtype=np.float64)
		return self.fit_f(X.data, X.indices, X.indptr, y, threads, seed)

	def fit_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr,
					np.ndarray[double, ndim=1, mode='c'] y, int threads, int seed):
		cdef double alpha= self.alpha, L2= self.L2, e_noise= self.e_noise, e, e_total= 0, e_clip= self.e_clip, abs_e
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *w2= &self.w2[0], *z1= &self.z1[0], *z2= &self.z2[0], \
					*c0= &self.c0[0], *c1= &self.c1[0], *c2= &self.c2[0]
		cdef double *ys= <double*> y.data
		cdef unsigned int lenn, D= self.D, D_nn= self.D_nn, D_nn2= self.D_nn2, ptr, row_count= X_indptr.shape[0]-1, \
							row, inv_link= self.inv_link, j=0, jj
		cdef int* inds, indptr
		cdef double* vals
		rand= np.random.Generator(randomgen.xoroshiro128.Xoroshiro128(seed= self.seed))
		for iter in range(self.iters):
			e_total= 0.0
			for row in range(row_count):
				ptr= X_indptr[row]
				lenn= X_indptr[row+1]-ptr
				inds= <int*> X_indices.data+ptr
				vals= <double*> X_data.data+ptr
				e= inv_link_f(predict_single(inds, vals, lenn, D, D_nn, D_nn2, w0, w1, w2, z1, z2, threads), \
							  self.inv_link) -ys[row]
				abs_e= fabs(e)
				e_total+= abs_e
				e += (rand.rand() - 0.5) * e_noise
				if abs_e> e_clip:
					if e>0:  e= e_clip
					else:  e= -e_clip
				update_single(inds, vals, lenn, D, D_nn, D_nn2, e, alpha, L2, w0, w1, w2, z1, z2, c0, c1, c2, threads)
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
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *w2= &self.w2[0], *z1= &self.z1[0], *z2= &self.z2[0]
		cdef unsigned int lenn, D= self.D, D_nn= self.D_nn, D_nn2= self.D_nn2, row_count= X_indptr.shape[0]-1, row, ptr
		if layer==1: p= np.zeros(((X_indptr.shape[0]-1),self.D_nn), dtype= np.float64)
		else:  p= np.zeros(((X_indptr.shape[0]-1),self.D_nn2), dtype= np.float64)
		cdef double[:,:] pp= p
		for row in range(row_count):
			ptr= X_indptr[row]
			lenn= X_indptr[row + 1] - ptr
			inds= <int*> X_indices.data + ptr
			vals= <double*> X_data.data + ptr
			if layer==1:  predict_single(inds, vals, lenn, D, D_nn, D_nn2, w0, w1, w2, &pp[row][0], z2, threads)
			else:  predict_single(inds, vals, lenn, D, D_nn, D_nn2, w0, w1, w2, z1, &pp[row][0], threads)
		return p

	def __getstate__(self):
		return (self.alpha,
				self.L2,
				self.e_noise,
				self.e_clip,
				self.init_nn,
				self.D,
				self.D_nn,
				self.D_nn2,
				self.iters,
				self.threads,
				np.asarray(self.w1),
				np.asarray(self.w2),
				np.asarray(self.z1),
				np.asarray(self.z2),
				np.asarray(self.c0),
				np.asarray(self.c1),
				np.asarray(self.c2),
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
		 self.D_nn2,
		 self.iters,
		 self.threads,
		 self.w1,
		 self.w2,
		 self.z1,
		 self.z2,
		 self.c0,
		 self.c1,
		 self.c2,
		 self.inv_link,
		 self.seed,
		 self.verbose) = params
