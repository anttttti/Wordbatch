#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import types
from contextlib import closing
import scipy.sparse as ssp
import random
import pandas as pd
import sys
from math import ceil
if sys.version_info.major == 3:
	import copyreg as copy_reg
else:
	import copy_reg

def _pickle_method(m):
	if sys.version_info.major == 3:
		if m.im_self is None:  return getattr, (m.im_self.__class__, m.im_func.__name__)
		else:  return getattr, (m.im_self, m.im_func.__name__)
	else:
		if m.__self__ is None:  return getattr, (m.__self__.__class__, m.__func__.__name__)
		else:  return getattr, (m.__self__, m.__func__.__name__)
copy_reg.pickle(types.MethodType, _pickle_method)

class Batcher(object):
	"""Scheduler to handle parallel jobs on minibatches

	Parameters
	----------
	procs: int
		Number of process(es)/thread(s) for executing task in parallel

	minibatch_size: int
		Expected size of each minibatch to individually perform task on.

	timeout: int | float
		Maximum time(seconds) to wait until the jobs finish or until a timeout occurs.

	spark_context: boolean
		Use SparkContext as backend. If set, input data is assumed to be RDD in format generated by lists2rddbatches().

	method: {'serial', 'multiprocessing', 'threading'}
		Backend method to apply task to the minibatches. Spark is special case, set by "spark_context"

			- 'serial' will apply task sequentially to minibatches without additional parallelism.

			- 'multiprocessing' will apply task to minibatches in parallel using multiple processes.

			- 'threading' will apply task to minibatches in parallel using multiple threads.

	verbose: int
		Verbosity level. Setting verbose > 0 will display additional information depending on the specific level set.
	"""
	def __init__(self, procs= 0, minibatch_size= 20000, timeout= 600, spark_context= None, method= "multiprocessing",
				 verbose= 1):
		if procs==0:  procs= multiprocessing.cpu_count()
		self.procs= procs
		self.verbose= verbose
		self.minibatch_size= minibatch_size
		self.timeout= timeout
		self.spark_context= spark_context
		self.method= method

	def lists2rddbatches(self, lists, minibatch_size=0):
		if minibatch_size==0:  minibatch_size= self.minibatch_size
		start= 0; len_data= len(lists[0]); batch_count= 0
		batches= []
		while start < len_data:
			batches.append([batch_count]+[column[start:start + minibatch_size] for column in lists])
			start+= minibatch_size
			batch_count+= 1
		return self.spark_context.parallelize(batches)

	def rddbatches2lists(self, rddbatches, sort= True):
		batches= rddbatches.collect()
		if sort:  batches= sorted(batches)
		texts= []
		labels= []
		for batch in batches:
			texts.append(batch[1])
			labels.append(batch[2])
		texts= self.merge_batches(texts)
		labels= self.merge_batches(labels)
		return texts, labels

	def split_batches(self, data, minibatch_size= None):
		"""Split data into minibatches with a specified size

		Parameters
		----------
		data: iterable and indexable
			List-like data to be split into batches. Includes spark_contextipy matrices and Pandas DataFrames.

		minibatch_size: int
			Expected sizes of minibatches split from the data.

		Returns
		-------
		data_split: list
			List of minibatches, each entry is a list-like object representing the data subset in a batch.
		"""
		if minibatch_size==None: minibatch_size= self.minibatch_size
		if isinstance(data, list) or isinstance(data, tuple):  len_data= len(data)
		else:  len_data= data.shape[0]
		if isinstance(data,pd.DataFrame):
			data_split = [data.iloc[x * minibatch_size:(x + 1) * minibatch_size] for x in
						  range(int(ceil(len_data / minibatch_size)))]
		else:
			data_split= [data[x* minibatch_size:min(len_data, (x+1)*minibatch_size)]
						 for x in range(int(ceil(len_data/minibatch_size)))]
		return data_split

	def merge_batches(self, data):
		"""Merge a list of data minibatches into one single instance representing the data

		Parameters
		----------
		data: list
			List of minibatches to merge

		Returns
		-------
		(anonymous): sparse matrix | pd.DataFrame | list
			Single complete list-like data merged from given batches
		"""
		if isinstance(data[0], ssp.csr_matrix):  return ssp.vstack(data)
		if isinstance(data[0], pd.DataFrame) or isinstance(data[0], pd.Series):  return pd.concat(data)
		return [item for sublist in data for item in sublist]

	def parallelize_batches(self, task, data, args, method=None, timeout=-1, input_split=False,
							merge_output= True, minibatch_size= None, procs=None):
		"""

		Parameters
		----------
		task: function
			Function to apply on each minibatch with other specified arguments

		data: list-like
			Samples to split into minibatches and apply the specified function on

		args: list
			Arguments to pass to the specified function following the mini-batch

		method: {'serial', 'multiprocessing', 'threading'}
			Method to apply task to the batches

			- 'serial' will apply task sequentially to the batches without additional parallelism.

			- 'multiprocessing' will apply task to batches in parallel using multiple processes.

			- 'threading' will apply task to batches in parallel using multiple threads.

		timeout: int | float
			Maximum time(seconds) to wait until the jobs finish or until a timeout occurs.

		input_split: boolean, default False
			If True, input data is already mapped into minibatches, otherwise data will be split on call.

		merge_output: boolean, default True
			If True, results from minibatches will be reduced into one single instance before return.

		minibatch_size: int
			Expected size of each mini-batch to individually perform task on. The actual sizes will be
			the same as the specified value except the last mini-batch, whose size might be exactly the same
			as this value or smaller.

		procs: int
			Number of process(es)/thread(s) to use to execute task in parallel.

		Returns
		-------
		results: list-like | list of list-like
			If merge_output is specified as True, this will be a list-like object representing
			the dataset, with each entry as a sample. Otherwise this will be a list of list-like
			objects, with each entry representing the results from a minibatch.
		"""
		if procs==None: procs= self.procs
		if method == None: method= self.method
		if self.verbose > 1:
			print("Parallel task:", task, " Method:", method, " Procs:", self.procs,
                  " input_split:", input_split, " merge_output:", merge_output)
		if method=="spark":
			#Assumes data is in column 1 of the RDD, as generated by lists2rddbatches
			rdd_col= 1
			if not (input_split):  data= self.lists2rddbatches(data)
			def apply_func(batch):  return batch[:rdd_col]+[task([batch[rdd_col]]+args)]+batch[rdd_col+1:]
			results= data.map(apply_func)
			if merge_output:  results= self.rddbatches2lists(results)
			return results

		if self.verbose> 10:
			print("len(data):", len(data), "len(args):", len(args), "[type(x) for x in data]:",
				  [type(x) for x in data], "[type(x) for x in args]:", [type(x) for x in args])

		if timeout==-1:  timeout= self.timeout
		attempt= 0
		if not(input_split):
			paral_params= [[data_batch]+ args for data_batch in self.split_batches(data, minibatch_size)]
		else:
			paral_params=  [[data_batch]+ args for data_batch in data]
		if method == "serial":
			results = [task(minibatch) for minibatch in paral_params]
		else:
			while (attempt != -1):
				try:
					if method=="multiprocessing":
						with closing(multiprocessing.Pool(max(1, procs), maxtasksperchild=2)) as pool:
							results= pool.map_async(task, paral_params)
							if timeout==0:
								pool.close()
								pool.join()
								results= results.get()
							else:
								results.wait(timeout=timeout)
								if results.ready():  results= results.get()
								else:  raise ValueError('Parallelization timeout')
					elif method=="threading":
						with closing(multiprocessing.dummy.Pool(max(1,procs))) as pool:
						   results= pool.map(task, paral_params)
						   pool.close()
						   pool.join()
					#elif method == "parallelpython":
					#    job_server= pp.Server()
					#    jobs= [job_server.submit(task, (x,), (), ()) for x in paral_params]
					#    results= [x() for x in jobs]
				except:
					print("Parallelization fail. Method:", method, "Task:", task)
					attempt+= 1
					if timeout!=0: timeout*= 2
					if attempt>=5:  return None
					print("Retrying, attempt:", attempt, "timeout limit:", timeout, "seconds")
					continue
				attempt= -1
		if merge_output:  return self.merge_batches(results)
		return results

	def shuffle_batch(self, texts, labels= None, seed= None):
		"""Shuffle a list of samples, as well as the labels if specified

		Parameters
		----------
		texts: list-like
			List of samples to shuffle

		labels: list-like (optional)
			List of labels to shuffle, should be correspondent to the samples given

		seed: int
			The seed of the pseudo random number generator to use for shuffling

		Returns
		-------
		texts: list
			List of shuffled samples (texts parameters)

		labels: list (optional)
			List of shuffled labels. This will only be returned when non-None
			labels is passed
		"""
		if seed!=None:  random.seed(seed)
		index_shuf= list(range(len(texts)))
		random.shuffle(index_shuf)
		texts= [texts[x] for x in index_shuf]
		if labels==None:  return texts
		labels= [labels[x] for x in index_shuf]
		return texts, labels

	def __getstate__(self):
		return dict((k, v) for (k, v) in self.__dict__.items())

	def __setstate__(self, params):
		for key in params:  setattr(self, key, params[key])
