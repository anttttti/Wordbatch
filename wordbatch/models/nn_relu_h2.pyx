# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
import pickle as pkl
import gzip
cimport cython
from cpython cimport array
import scipy.sparse as ssp
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, log, fmax, fmin, sqrt, fabs
import multiprocessing

np.import_array()

cdef double inv_link_f(double e, int inv_link) nogil:
	if inv_link==1:  return 1.0 / (1.0 + exp(-fmax(fmin(e, 35.0), -35.0)))
	return e

cdef double predict_single(int* inds, double* vals, int lenn, unsigned int D, unsigned int D_nn, unsigned int D_nn2,
				double* w0, double* w1, double* w2, double* z1, double* z2, int threads) nogil:
	cdef unsigned int i, ii, j, k, DD_nn= D*D_nn, DD_nn2= D_nn*D_nn2
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

cdef void update_single(int* inds, double* vals, int lenn, unsigned int D, unsigned int D_nn, unsigned int D_nn2,
						double e, double alpha, double L2, double* w0, double* w1, double* w2, double* z1,
					   	double* z2, double* c0, double* c1, double* c2, int threads) nogil:
	cdef unsigned int i, ii, j, k, DD_nn= D*D_nn, DD_nn2= D_nn*D_nn2
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
	cdef double[:] w0
	cdef double[:] w1
	cdef double[:] w2
	cdef double[:] z1
	cdef double[:] z2
	cdef double[:] c0
	cdef double[:] c1
	cdef double[:] c2

	cdef unsigned int threads
	cdef unsigned int iters
	cdef unsigned int D
	cdef unsigned int D_nn
	cdef unsigned int D_nn2
	cdef double init_nn

	cdef double L2
	cdef double alpha
	cdef double e_noise
	cdef int inv_link
	cdef int seed

	def __init__(self,
				 double alpha=0.1,
				 double L2=0.001,
			   	 double e_noise=0.001,
				 unsigned int D=2**25,
				 unsigned int D_nn=40,
				 unsigned int D_nn2=40,
				 double init_nn=0.0001,
				 unsigned int iters=1,
				 inv_link= "sigmoid",
				 int threads= 0,
				 int seed= 0):

		self.alpha= alpha
		self.L2= L2
		self.e_noise= e_noise
		self.D= D
		self.D_nn= D_nn
		self.D_nn2= D_nn2
		self.iters= iters
		if threads==0:  threads= multiprocessing.cpu_count()-1
		self.threads= threads
		if inv_link=="sigmoid":  self.inv_link= 1
		if inv_link=="identity":  self.inv_link= 0

		self.seed = seed
		rand= np.random.RandomState(seed)

		self.w0 = (rand.rand((D + 1) * D_nn) - 0.5) * init_nn
		self.w1 = (rand.rand((D_nn + 1) * D_nn2) - 0.3) * init_nn
		self.w2 = (rand.rand(D_nn2 + 1) - 0.5) * init_nn


		self.z1= np.zeros((D_nn,), dtype=np.float64)
		self.z2= np.zeros((D_nn2,), dtype=np.float64)
		self.c0= np.zeros((D,), dtype=np.float64)
		self.c1= np.zeros((D_nn,), dtype=np.float64)
		self.c2= np.zeros((D_nn2,), dtype=np.float64)

	def predict(self, X, int threads= 0):
		if threads==0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
		return self.predict_f(X.data, X.indices, X.indptr, threads)

	def predict_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr, int threads):
		cdef double alpha= self.alpha, L2= self.L2
		p= np.zeros(X_indptr.shape[0]-1, dtype= np.float64)
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *w2= &self.w2[0], *z1= &self.z1[0], *z2= &self.z2[0]
		cdef double[:] pp= p
		cdef unsigned int lenn, D= self.D, D_nn= self.D_nn, D_nn2= self.D_nn2, row_count= X_indptr.shape[0]-1, row, ptr
		for row in range(row_count):
			ptr= X_indptr[row]
			lenn= X_indptr[row + 1] - ptr
			inds= <int*> X_indices.data + ptr
			vals= <double*> X_data.data + ptr
			pp[row]= inv_link_f(predict_single(inds, vals, lenn, D, D_nn, D_nn2, w0, w1, w2, z1, z2, threads), \
					self.inv_link)
		return p

	def fit(self, X, y, int threads= 0, int seed= 0, int verbose=0):
		if threads == 0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X = ssp.csr_matrix(X, dtype=np.float64)
		if type(y) != np.array:  y = np.array(y, dtype=np.float64)
		self.fit_f(X.data, X.indices, X.indptr, y, threads, seed, verbose)

	def fit_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr,
					np.ndarray[double, ndim=1, mode='c'] y, int threads, int seed, int verbose):
		cdef double alpha= self.alpha, L2= self.L2, e_noise= self.e_noise, e, e_total= 0
		cdef double *w0= &self.w0[0], *w1= &self.w1[0], *w2= &self.w2[0], *z1= &self.z1[0], *z2= &self.z2[0], \
					*c0= &self.c0[0], *c1= &self.c1[0], *c2= &self.c2[0]
		cdef double *ys= <double*> y.data
		cdef unsigned int lenn, D= self.D, D_nn= self.D_nn, D_nn2= self.D_nn2, ptr, row_count= X_indptr.shape[0]-1, \
				   			row, inv_link= self.inv_link, j=0, jj
		cdef int* inds, indptr
		cdef double* vals
		rand= np.random.RandomState(seed)

		for iter in range(self.iters):
			e_total= 0.0
			for row in range(row_count):
				ptr= X_indptr[row]
				lenn= X_indptr[row+1]-ptr
				inds= <int*> X_indices.data+ptr
				vals= <double*> X_data.data+ptr
				e= inv_link_f(predict_single(inds, vals, lenn, D, D_nn, D_nn2, w0, w1, w2, z1, z2, threads), \
							  self.inv_link) -ys[row]
				e_total+= fabs(e)
				e += (rand.rand() - 0.5) * e_noise
				update_single(inds, vals, lenn, D, D_nn, D_nn2, e, alpha, L2, w0, w1, w2, z1, z2, c0, c1, c2, threads)
			if verbose > 0:  print "Total e:", e_total

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

	#Pickling needs to be fixed.
	# def pickle_model(self, filename):
	# 	with gzip.open(filename, 'wb') as model_file:
	# 		pkl.dump(self.get_params(), model_file, protocol=2)
    #
	# def unpickle_model(self, filename):
	# 	self.set_params(pkl.load(gzip.open(filename, 'rb')))
    #
	# def __getstate__(self):
	# 	return (self.alpha, self.beta, self.L1, self.L2, self.D, self.iters,
	# 			np.asarray(self.w), np.asarray(self.z), np.asarray(self.n), self.inv_link)
    #
	# def __setstate__(self, params):
	# 	(self.alpha, self.beta, self.L1, self.L2, self.D, self.iters, self.w, self.z, self.n, self.inv_link)= params
