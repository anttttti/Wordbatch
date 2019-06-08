#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import re
import wordbatch.batcher
import wordbatch.pipelines
import wordbatch.transformers

def batch_transform(args):
	return args[1].batch_transform(args[0])

non_alphanums= re.compile(u'[^A-Za-z0-9]+')
def default_normalize_text(text):
	return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x)>1])

class TextNormalizer(object):
	def __init__(self, normalize_text= default_normalize_text, freeze= False, verbose= 1):
		self.verbose= verbose
		self.freeze= freeze
		self.non_alphanums = re.compile(u'[^A-Za-z0-9]+')
		self.normalize_text= normalize_text
		self.reset()

	def reset(self):
		return self

	def batch_transform(self, data):  return [self.normalize_text(text) for text in data]

	def transform(self, data, input_split=False, merge_output=True, batcher=None):
		if batcher is None:  batcher = wordbatch.batcher.Batcher()
		return batcher.process_batches(batch_transform, data, [self], input_split=input_split,
		                               merge_output=merge_output)

	def fit(self, data, y=None, input_split=False, merge_output=True, batcher=None):
		return self

	def fit_transform(self, data, y=None, input_split=False, merge_output=True, batcher=None):
		return self.transform(data, input_split, merge_output, batcher)