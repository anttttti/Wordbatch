#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import wordbatch.batcher

def batch_transform(args):
	f= args[1]
	f_args= args[2]
	f_kwargs= args[3]
	return f(args[0], *f_args, **f_kwargs)

class ApplyBatch(object):
	#Applies a function to the entire minibatch. Use this for example on Pandas dataframes, to avoid per-row overhead.
	#Function needs to be applicable to the array/list of values!
	#If not, modify/wrap the function to process a list, or use Apply
	def __init__(self, function, batcher=None, args=[], kwargs={}):
		if batcher is None:  self.batcher = wordbatch.batcher.Batcher()
		else:  self.batcher = batcher
		self.function= function
		self.args= [args]
		self.kwargs= [kwargs]

	def fit(self, data, input_split= False):
		return self

	def fit_transform(self, data, input_split=False, merge_output=True, minibatch_size=None, batcher=None):
		return self.transform(data, input_split, merge_output, minibatch_size, batcher)

	def transform(self, data, input_split=False, merge_output=True, minibatch_size=None, batcher=None):
		if batcher is None:  batcher = self.batcher
		return batcher.process_batches(batch_transform, data, [self.function] + self.args + self.kwargs,
		                                    input_split=input_split, merge_output=merge_output,
		                                    minibatch_size= minibatch_size)

# import wordbatch.batcher as batcher
# b= batcher.Batcher(minibatch_size=2)#, method="serial")
# import numpy as np
# a= ApplyBatch(np.power, b, [2],{})
# print(a.transform([1, 2, 3, 4]))