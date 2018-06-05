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

if sys.version_info.major == 3:
	import pickle as pkl
else:
	import cPickle as pkl

np.import_array()

cdef extern from "avx_ext.h":# nogil:
	void update_fm_ftrl_avx(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w,
							double* z, double* n, double alpha_fm, const double L2_fm, double* w_fm, double* z_fm,
							double* n_fm, int D_fm, int bias_term, int nThreads);
	double predict_fm_ftrl_avx(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha,
							   double beta, double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm,
							   double weight_fm, int D_fm, int bias_term, int nThreads);


cdef double inv_link_f(double e, int inv_link) nogil:
	if inv_link==1:  return 1.0 / (1.0 + exp(-fmax(fmin(e, 35.0), -35.0))) #Sigmoid + logloss
	return e

cdef double predict_single(int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
						   double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm,
						   int D_fm, bint bias_term, int threads) nogil:
	cdef int i, ii, k
	cdef double sign, zi, d, wi, wi2, wfmk, e= 0.0, e2= 0.0

	if bias_term:
		if z[0] != 0:
			wi = w[0] = -z[0] / ((beta + sqrt(n[0])) * ialpha)
			e += wi
		else:  w[0] = 0.0

	for ii in prange(lenn, nogil=True, num_threads= threads):
		i= inds[ii]
		zi= z[i]
		sign= -1.0 if zi < 0 else 1.0
		if sign * zi  > L1:
			w[ii+1]= wi= (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2)
			e+= wi * vals[ii]
		else:  w[ii+1] = 0.0

	wi2= 0.0
	for k in prange(D_fm, nogil=True, num_threads=threads):
		wfmk= 0.0
		for ii in range(lenn):
			d= z_fm[inds[ii] * D_fm + k] * vals[ii]
			wfmk= wfmk+d
			wi2+= d **2
		e2+= wfmk **2
		w_fm[k]= wfmk
	e2= (e2- wi2)* 0.5 *weight_fm
	return e+e2

cdef void update_single(int* inds, double* vals, int lenn, double e, double ialpha, double* w, double* z, double* n,
						double alpha_fm, double L2_fm, double* w_fm, double* z_fm, double* n_fm,
						int D_fm, bint bias_term, int threads) nogil:
	cdef int i, ii, k
	cdef double g, g2, ni, v, lr, e2= e**2, reg, L2_fme= L2_fm / e
	cdef double *z_fmi
	if bias_term: #Update bias with FTRL-proximal
		g2= e ** 2
		ni= n[0]
		z[0]+= e - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[0]
		n[0]+= g2

	for ii in prange(lenn, nogil=True, num_threads= threads):
	#for ii in range(lenn):
		i= inds[ii]
		v= vals[ii]
		#Update 1st order model with FTRL-proximal
		g= e * v
		g2= g * g
		ni= n[i]
		z[i]+= g - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[ii+1]
		n[i]+= g2

		#Update FM with adaptive regularized SGD
		z_fmi= z_fm+ i * D_fm
		lr= g* alpha_fm / (sqrt(n_fm[i])+1.0)
		reg= v - L2_fme
		for k in range(D_fm):  z_fmi[k]-= lr * (w_fm[k] - z_fmi[k] * reg)
		n_fm[i] += e2

cdef class FM_FTRL:
	cdef double[:] w
	cdef double[:] z
	cdef double[:] n
	cdef double[:] w_fm
	cdef double[:] z_fm
	cdef double[:] n_fm

	cdef unsigned int threads
	cdef unsigned int iters
	cdef unsigned int D
	cdef unsigned int D_fm
	cdef double L1
	cdef double L2
	cdef double alpha
	cdef double beta
	cdef double alpha_fm
	cdef double L2_fm
	cdef double weight_fm
	cdef double init_fm
	cdef double e_noise
	cdef double e_clip
	cdef int inv_link
	cdef bint bias_term
	cdef int use_avx
	cdef int seed
	cdef int verbose

	def __init__(self,
				 double alpha=0.02,
				 double beta=0.01, # ~ alpha/2
				 double L1=0.0001,
				 double L2=0.1,
				 unsigned int D=2**25,
				 double alpha_fm=0.03,
				 double L2_fm= 0.005,
				 double init_fm= 0.01,
				 unsigned int D_fm=20,
				 double weight_fm= 10.0,
				 double e_noise= 0.0001,
				 double e_clip= 1.0,
				 unsigned int iters=5,
				 inv_link= "identity",
				 bint bias_term=1,
				 int threads= 0,
				 int use_avx=1,
				 int seed= 0,
				 int verbose=1):

		self.alpha= alpha
		self.beta= beta
		self.L1= L1
		self.L2= L2
		self.D= D
		self.alpha_fm= alpha_fm
		self.L2_fm= L2_fm
		self.init_fm= init_fm
		self.D_fm= D_fm
		self.weight_fm= weight_fm
		self.e_noise= e_noise
		self.e_clip= e_clip
		self.iters= iters
		if threads==0:  threads= multiprocessing.cpu_count()-1
		self.threads= threads
		if inv_link=="sigmoid":  self.inv_link= 1
		if inv_link=="identity":  self.inv_link= 0
		self.bias_term= bias_term
		self.use_avx = use_avx
		self.seed = seed
		self.verbose= verbose
		self.reset()

	def reset(self):
		D= self.D
		D_fm= self.D_fm
		self.w = np.ones((D), dtype=np.float64)
		self.z = np.zeros((D), dtype=np.float64)
		self.n = np.zeros((D), dtype=np.float64)
		self.w_fm = np.zeros(D_fm, dtype=np.float64)
		rand= randomgen.xoroshiro128.Xoroshiro128(seed= self.seed).generator
		self.z_fm = (rand.random_sample(D * D_fm) - 0.5) * self.init_fm
		self.n_fm = np.zeros(D, dtype=np.float64)

	def predict(self, X, int threads= 0):
		if threads==0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X= ssp.csr_matrix(X, dtype=np.float64)
		return self.predict_f(np.ascontiguousarray(X.data), np.ascontiguousarray(X.indices),
							  np.ascontiguousarray(X.indptr), threads)

	def predict_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr, int threads):
		cdef double ialpha= 1.0/self.alpha, L1= self.L1, beta= self.beta, baL2= beta * ialpha + self.L2, \
					weight_fm= self.weight_fm
		cdef double *w= &self.w[0], *z= &self.z[0], *n= &self.n[0], *n_fm= &self.n_fm[0], \
					*z_fm= &self.z_fm[0], *w_fm= &self.w_fm[0]
		cdef unsigned int D_fm= self.D_fm, k
		p= np.zeros(X_indptr.shape[0]-1, dtype= np.float64)
		cdef double[:] pp= p
		cdef unsigned int lenn, row_count= X_indptr.shape[0]-1, row, ptr
		cdef bint bias_term= self.bias_term
		for row in range(row_count):
			ptr= X_indptr[row]
			lenn= X_indptr[row + 1] - ptr
			inds= <int*> X_indices.data + ptr
			vals= <double*> X_data.data + ptr

			if self.use_avx == 1:
				pp[row]= inv_link_f(predict_fm_ftrl_avx(inds, vals, lenn,
												   L1, baL2, ialpha, beta, w, z, n,
												   w_fm, z_fm, n_fm, weight_fm,
												   D_fm, bias_term, threads), self.inv_link)
			else:
				pp[row]= inv_link_f(predict_single(inds, vals, lenn,
												   L1, baL2, ialpha, beta, w, z, n,
												   w_fm, z_fm, n_fm, weight_fm,
												   D_fm, bias_term, threads), self.inv_link)
		return p


	def partial_fit(self, X, y, sample_weight= None, int threads = 0, int seed = 0):
		return self.fit(X, y, sample_weight= sample_weight, threads = threads, seed = seed, reset= False)

	def fit(self, X, y, sample_weight= None, int threads= 0, int seed= 0, reset= True):
		if reset:  self.reset()
		if threads == 0:  threads= self.threads
		if type(X) != ssp.csr.csr_matrix:  X = ssp.csr_matrix(X, dtype=np.float64)
		#if type(y) != np.array:  y = np.array(y, dtype=np.float64)
		y= np.ascontiguousarray(y, dtype=np.float64)
		if sample_weight is not None and type(sample_weight) != np.array:
			sample_weight= np.array(sample_weight, dtype=np.float64)
		return self.fit_f(X.data, X.indices, X.indptr, y, sample_weight, threads, seed)

	def fit_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr,
					np.ndarray[double, ndim=1, mode='c'] y,
					sample_weight,
					int threads, int seed):
		cdef double ialpha= 1.0/self.alpha, L1= self.L1, beta= self.beta, baL2= beta * ialpha + self.L2, \
					alpha_fm= self.alpha_fm, weight_fm= self.weight_fm, L2_fm= self.L2_fm, e, e_total= 0, zfmi, \
					e_noise= self.e_noise, e_clip= self.e_clip, abs_e
		cdef double *w= &self.w[0], *z= &self.z[0], *n= &self.n[0], *n_fm= &self.n_fm[0], \
					*z_fm= &self.z_fm[0], *w_fm= &self.w_fm[0], *ys= <double*> y.data
		cdef unsigned int D_fm= self.D_fm, lenn, ptr, row_count= X_indptr.shape[0]-1, row, inv_link= self.inv_link
		cdef bint bias_term= self.bias_term
		cdef int* inds, indptr
		cdef double* vals

		rand = randomgen.xoroshiro128.Xoroshiro128(seed=seed).generator
		for iter in range(self.iters):
			e_total= 0.0
			for row in range(row_count):
				ptr= X_indptr[row]
				lenn= X_indptr[row+1]-ptr
				inds= <int*> X_indices.data+ptr
				vals= <double*> X_data.data+ptr

				if self.use_avx == 1:
					e = inv_link_f(predict_fm_ftrl_avx(inds, vals, lenn,
													   L1, baL2, ialpha, beta, w, z, n,
													   w_fm, z_fm, n_fm, weight_fm,
													   D_fm, bias_term, threads), inv_link) - ys[row]
				else:
					e= inv_link_f(predict_single(inds, vals, lenn,
												L1, baL2, ialpha, beta, w, z, n,
												w_fm, z_fm, n_fm, weight_fm,
												D_fm, bias_term, threads), inv_link) -ys[row]

				abs_e= fabs(e)
				e_total+= abs_e
				e += (rand.rand() - 0.5) * e_noise
				if abs_e> e_clip:
					if e>0:  e= e_clip
					else:  e= -e_clip
				if sample_weight is not None:
					e*= sample_weight[row]

				if self.use_avx == 1:
					update_fm_ftrl_avx(inds, vals, lenn, e, ialpha, w, z, n, alpha_fm, L2_fm, w_fm, z_fm, n_fm, D_fm,
										 bias_term, threads)
				else:
					update_single(inds, vals, lenn, e, ialpha, w, z, n, alpha_fm, L2_fm, w_fm, z_fm, n_fm, D_fm,
								  bias_term, threads)

			if self.verbose>0:  print "Total e:", e_total
		return self

	def pickle_model(self, filename):
		with gzip.open(filename, 'wb') as model_file:
			pkl.dump(self.get_params(), model_file, protocol=2)

	def unpickle_model(self, filename):
		self.set_params(pkl.load(gzip.open(filename, 'rb')))

	def __getstate__(self):
		return (self.alpha,
				self.beta,
				self.L1,
				self.L2,
				self.alpha_fm,
				self.L2_fm,
				self.e_noise,
				self.e_clip,
				self.weight_fm,
				self.init_fm,
				self.D,
				self.D_fm,
				self.iters,
				np.asarray(self.w),
				np.asarray(self.z),
				np.asarray(self.n),
				np.asarray(self.w_fm),
				np.asarray(self.z_fm),
				np.asarray(self.n_fm),
				self.inv_link,
				self.seed,
				self.use_avx,
				self.bias_term,
				self.threads,
				self.verbose)

	def __setstate__(self, params):
		(self.alpha,
		 self.beta,
		 self.L1,
		 self.L2,
		 self.alpha_fm,
		 self.L2_fm,
		 self.e_noise,
		 self.e_clip,
		 self.weight_fm,
		 self.init_fm,
		 self.D,
		 self.D_fm,
		 self.iters,
		 self.w,
		 self.z,
		 self.n,
		 self.w_fm,
		 self.z_fm,
		 self.n_fm,
		 self.inv_link,
		 self.seed,
		 self.use_avx,
		 self.bias_term,
		 self.threads,
		 self.verbose)= params
