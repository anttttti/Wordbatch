#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import wordbatch.batcher

def batch_transform(args):
	return args[1].transform(args[0])

class BatchTransformer(object):
	def __init__(self, transformer, call_fit=True, batcher=None):
		if batcher is None:  self.batcher = wordbatch.batcher.Batcher()
		else:  self.batcher = batcher
		self.transformer= transformer
		self.call_fit= call_fit

	def fit(self, data, input_split=False, batcher=None):
		if self.call_fit:
			if input_split:  self.transformer.fit(self.batcher.merge_batches(self.batcher.collect_batches(data)))
			else:  self.transformer.fit(data)
		return self

	def fit_transform(self, data, input_split=False, merge_output=True, minibatch_size=None, batcher=None):
		if self.call_fit:  self.fit(data, input_split= input_split)
		return self.transform(data, input_split, merge_output, minibatch_size= minibatch_size, batcher=batcher)

	def transform(self, data, input_split=False, merge_output=True, minibatch_size=None, batcher=None):
		if batcher is None:  batcher = self.batcher
		return batcher.process_batches(batch_transform, data, [self.transformer],
		                               input_split=input_split, merge_output=merge_output,
		                               minibatch_size= minibatch_size)
