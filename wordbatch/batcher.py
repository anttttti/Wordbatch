#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
from contextlib import closing
import scipy.sparse as ssp
import random
import pandas as pd
from math import ceil


class Batcher(object):
	"""Scheduler to handle parallel jobs on minibatches

	Parameters
	----------
	procs: int
		Number of process(es)/thread(s) for executing task in parallel. Used for multiprocessing, threading and Loky

	minibatch_size: int
		Expected size of each minibatch

	backend: {'serial', 'multiprocessing', 'threading', 'loky', 'spark', 'dask', 'ray'}
		Backend for computing the tasks

			- 'serial' sequential execution without a backend scheduler

			- 'multiprocessing' Python standard multiprocessing library

			- 'threading' Python standard threading library

			- 'loky' Loky fork of multiprocessing library

			- 'spark' PySpark local or distributed execution

			- 'dask' Dask Distributed local or distributed execution

			- 'ray' Ray local or distributed execution

	task_num_cpus: int
		Number of CPUs to reserve per minibatch task for Ray

	task_num_gpus: int
		Number of GPUs to reserve per minibatch task for Ray

	backend_handle: object
		Backend handle for sending tasks

	verbose: int
		Verbosity level. Setting verbose > 0 will display additional information depending on the specific level set.
	"""
	def __init__(self, procs= 0, minibatch_size= 20000, backend_handle= None, backend= "multiprocessing",
				 task_num_cpus= 1, task_num_gpus= 0, verbose= 0):
		if procs==0:  procs= multiprocessing.cpu_count()
		self.procs= procs
		self.verbose= verbose
		self.minibatch_size= minibatch_size
		self.backend_handle= backend_handle
		self.backend= backend
		self.task_num_cpus = task_num_cpus
		self.task_num_gpus = task_num_gpus

	def list2indexedrdd(self, lst, minibatch_size=0):
		if minibatch_size==0:  minibatch_size= self.minibatch_size
		start= 0; len_data= len(lst); batch_count= 0
		batches= []
		while start < len_data:
			batches.append([batch_count]+[lst[start:start + minibatch_size]])
			start+= minibatch_size
			batch_count+= 1
		return self.backend_handle.parallelize(batches)

	def indexedrdd2list(self, indexedrdd, sort= True):
		batches= indexedrdd.collect()
		if sort:  batches= sorted(batches)
		return [batch[1] for batch in batches]

	def split_batches(self, data, minibatch_size= None, backend= None):
		"""Split data into minibatches with a specified size

		Parameters
		----------
		data: iterable and indexable
			List-like data to be split into batches. Includes backend_handleipy matrices and Pandas DataFrames.

		minibatch_size: int
			Expected sizes of minibatches split from the data.

		backend: object
			Backend to use, instead of the Batcher backend attribute

		Returns
		-------
		data_split: list
			List of minibatches, each entry is a list-like object representing the data subset in a batch.
		"""
		if minibatch_size is None:  minibatch_size= self.minibatch_size
		if backend is None:  backend= self.backend
		if isinstance(data, list) or isinstance(data, tuple) or isinstance(data, dict):  len_data= len(data)
		else:  len_data= data.shape[0]
		if backend=="spark":  return self.list2indexedrdd(data, minibatch_size)
		if isinstance(data,pd.DataFrame):
			data= [data.iloc[x * minibatch_size:(x + 1) * minibatch_size] for x in
						  range(int(ceil(len_data / minibatch_size)))]
		elif isinstance(data, dict):
			data = [dict(list(data.items())[x * minibatch_size:min(len_data, (x + 1) * minibatch_size)])
			        for x in range(int(ceil(len_data / minibatch_size)))]
		else:
			data= [data[x* minibatch_size:min(len_data, (x+1)*minibatch_size)]
					 for x in range(int(ceil(len_data/minibatch_size)))]
		###if backend=="dask":  return self.backend_handle.scatter(data)
		return data

	def collect_batches(self, data, backend= None, sort= True):
		if backend is None:  backend= self.backend
		if backend == "spark":  data=  self.indexedrdd2list(data, sort)
		if backend == "dask":  data = self.backend_handle.gather(data)
		return data

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

	def process_batches(self, task, data, args, backend=None, backend_handle=None, input_split=False,
	                    merge_output= True, minibatch_size= None, procs=None, task_num_cpus= None,
						task_num_gpus= None, verbose= None):
		"""

		Parameters
		----------
		task: function
			Function to apply on each minibatch with other specified arguments

		data: list-like
			Samples to split into minibatches and apply the specified function on

		args: list
			Arguments to pass to the specified function following the mini-batch

		input_split: boolean, default False
			If True, input data is already mapped into minibatches, otherwise data will be split on call.

		merge_output: boolean, default True
			If True, results from minibatches will be reduced into one single instance before return.

		procs: int
			Number of process(es)/thread(s) for executing task in parallel. Used for multiprocessing, threading,
			Loky and Ray

		minibatch_size: int
			Expected size of each minibatch

		backend: {'serial', 'multiprocessing', 'threading', 'loky', 'spark', 'dask', 'ray'}
			Backend for computing the tasks

				- 'serial' sequential execution without a backend scheduler

				- 'multiprocessing' Python standard multiprocessing library
	
				- 'threading' Python standard threading library

				- 'loky' Loky fork of multiprocessing library

				- 'spark' PySpark local or distributed execution

				- 'dask' Dask Distributed local or distributed execution

				- 'ray' Ray local or distributed execution

		backend_handle: object
			Backend handle for sending tasks

		task_num_cpus: int
			Number of CPUs to reserve per minibatch task for Ray

		task_num_gpus: int
			Number of GPUs to reserve per minibatch task for Ray

		verbose: int
			Verbosity level. Setting verbose > 0 will display additional information depending on the specific level set.

		Returns
		-------
		results: list-like | list of list-like
			If merge_output is specified as True, this will be a list-like object representing
			the dataset, with each entry as a sample. Otherwise this will be a list of list-like
			objects, with each entry representing the results from a minibatch.
		"""
		if procs is None:  procs= self.procs
		if backend is None:  backend= self.backend
		if backend_handle is None:  backend_handle = self.backend_handle
		if task_num_cpus is None:  task_num_cpus = self.task_num_cpus
		if task_num_gpus is None:  task_num_gpus = self.task_num_gpus
		if verbose is None: verbose= self.verbose
		if verbose > 1:
			print("Task:", task, " backend:", backend, " backend_handle:", backend_handle, " procs:",
		      self.procs, " input_split:", input_split, " merge_output:", merge_output)

		if verbose> 10:
			print("len(data):", len(data), "len(args):", len(args), "[type(x) for x in data]:",
				  [type(x) for x in data], "[type(x) for x in args]:", [type(x) for x in args])

		if not(input_split):
			if backend=="spark":
				paral_params= self.split_batches(data, minibatch_size, backend="spark")
			else:
				paral_params= [[data_batch]+ args for data_batch in self.split_batches(data, minibatch_size)]
		else:
			if backend!="spark":  paral_params= [[data_batch]+ args for data_batch in data]
			else:  paral_params= data
		if verbose > 1:  print("Start task, len(paral_params)", len(paral_params))
		if backend == "serial":
			results = [task(minibatch) for minibatch in paral_params]
		else:
			if backend=="multiprocessing":
				with closing(multiprocessing.Pool(max(1, procs), maxtasksperchild=2)) as pool:
					results = pool.map_async(task, paral_params)
					pool.close()
					pool.join()
					results= results.get()
			elif backend=="threading":
				with closing(multiprocessing.dummy.Pool(max(1,procs))) as pool:
					results= pool.map(task, paral_params)
					pool.close()
					pool.join()
			if backend=="loky":
				from loky import get_reusable_executor
				pool= get_reusable_executor(max_workers=max(1, procs))
				results= list(pool.map(task, paral_params))
			elif backend == "dask":
				###if not (input_split):  data= self.scatter(data)
				results = [self.backend_handle.submit(task, params) for params in paral_params]
			elif backend == "spark":
				def apply_func_to_indexedrdd(batch):
					return [batch[0]] + [task([batch[1]] + args)]
				results = paral_params.map(apply_func_to_indexedrdd)
			elif backend == "ray":
				@self.backend_handle.remote(num_cpus=task_num_cpus, num_gpus=task_num_gpus)
				def f_ray(f, data):
					return f(data)
				results = [f_ray.remote(task, paral_params.pop(0)) for _ in range(min(len(paral_params), self.procs))]
				uncompleted = results
				while (len(paral_params) > 0):
					# More tasks than available processors. Queue the task calls
					done, remaining = self.backend_handle.wait(uncompleted, timeout=60, fetch_local=False)
					if len(done) == 0: continue
					done= done[0]
					uncompleted = [x for x in uncompleted if x != done]
					if len(remaining) > 0:
						new = f_ray.remote(task, paral_params.pop(0))
						uncompleted.append(new)
						results.append(new)
				results = [self.backend_handle.get(x) for x in results]
			#ppft currently not supported. Supporting arbitrary tasks requires modifications to passed arguments
			#elif backend == "ppft":
			#   jobs = [self.backend_handle.submit(task, (x,), (), ()) for x in paral_params]
			#	results = [x() for x in jobs]

		if merge_output:  return self.merge_batches(self.collect_batches(results, backend=backend))
		if verbose > 2:
			print("Task:", task, " backend:", backend, " backend_handle:", backend_handle, " completed")
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
		if seed != None:  random.seed(seed)
		index_shuf= list(range(len(texts)))
		random.shuffle(index_shuf)
		texts= [texts[x] for x in index_shuf]
		if labels == None:  return texts
		labels= [labels[x] for x in index_shuf]
		return texts, labels

	def __getstate__(self):
		return dict((k, v) for (k, v) in self.__dict__.items())

	def __setstate__(self, params):
		for key in params:  setattr(self, key, params[key])
