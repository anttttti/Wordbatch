#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import wordbatch.batcher
import wordbatch.pipelines
import wordbatch.transformers

class WordBatch(object):
	def __init__(self, normalize_text="new", dictionary="new", tokenizer=None, extractor=None,
				 freeze=False, fit_extractor= False, batcher=None, verbose=0):
		self.verbose= verbose
		if batcher is None:  self.batcher= wordbatch.batcher.Batcher(verbose=verbose)
		else:  self.batcher=  batcher

		if normalize_text is None:  self.normalize_text= None
		elif normalize_text== "new":  self.normalize_text= wordbatch.transformers.text_normalizer.TextNormalizer()
		elif callable(normalize_text):
			self.normalize_text= wordbatch.pipelines.apply.Apply(normalize_text, batcher=batcher)
		else:  self.normalize_text= normalize_text

		if tokenizer is None:  self.tokenizer= None
		else:  self.tokenizer= tokenizer

		if dictionary is None:  self.dictionary= None
		elif dictionary== "new":  self.dictionary= wordbatch.transformers.dictionary.Dictionary()
		else:  self.dictionary= dictionary

		if extractor is None:  self.extractor= None
		else:  self.extractor= wordbatch.pipelines.batch_transformer.BatchTransformer(extractor, batcher=batcher,
		                                                                              call_fit=fit_extractor)
		if self.extractor is not None:
			if hasattr(self.extractor.transformer, "dictionary"):
				self.extractor.transformer.dictionary = self.dictionary
		self.freeze= freeze

	def reset(self):
		if self.dictionary is not None:  self.dictionary.reset()
		return self

	def process(self, data, y=None, input_split=False, reset=True, update=True, minibatch_size=None, batcher= None):
		if batcher is None:  batcher = self.batcher
		if reset:  self.reset()
		if self.freeze:  update= False

		if self.normalize_text is not None:
			if self.verbose > 0:  print("Normalize text")
			data= self.normalize_text.transform(data, input_split=input_split, merge_output=False,
			                                    minibatch_size= minibatch_size, batcher= self.batcher)
			input_split= True
		if self.tokenizer is not None:
			if self.verbose > 0:  print("Tokenize text")
			if update:  data= self.tokenizer.fit_transform(data, y=y, input_split=input_split, merge_output=False,
			                                               reset=reset, minibatch_size= minibatch_size,
			                                               batcher=self.batcher)
			else: data= self.tokenizer.transform(data, y=y, input_split=input_split, merge_output=False,
			                                     minibatch_size= minibatch_size, batcher=self.batcher)
			input_split= True
		if self.dictionary is not None:
			if self.verbose > 0:  print("Use dictionary")
			if update:  data= self.dictionary.fit_transform(data, y=y, input_split=input_split, merge_output=False,
				                                     reset=reset, minibatch_size= minibatch_size, batcher=self.batcher)
			else: data= self.dictionary.transform(data, y=y, input_split=input_split, merge_output=False,
			                                      minibatch_size= minibatch_size, batcher=self.batcher)
		if self.verbose> 2: print("len(self.dictionary.dft):", len(self.dictionary.dft))
		return data

	def fit(self, data, y=None, input_split=False, reset=True, minibatch_size=None, batcher= None):
		if batcher is None:  batcher = self.batcher
		self.process(data, y, input_split, reset=reset, update= True, minibatch_size= minibatch_size, batcher= batcher)
		if self.extractor is not None:
			self.extractor.fit(data, input_split=input_split)
		return self

	def transform(self, data, y=None, cache_features=None, input_split=False, reset=False, update=False,
	              minibatch_size=None, batcher= None):
		if batcher is None:  batcher = self.batcher
		if cache_features is not None:
			if self.extractor is not None and os.path.exists(cache_features) and \
					hasattr(self.extractor.transformer, "load_features"):
				return self.extractor.transformer.load_features(cache_features)
		if not(input_split):  data= batcher.split_batches(data, minibatch_size= minibatch_size)
		data= self.process(data, y=y, input_split=True, reset=reset, update=update)
		if self.extractor is not None:
			if self.verbose > 0:  print("Extract features")
			if update: data= self.extractor.fit_transform(data, input_split=True, merge_output=True)
			else: data= self.extractor.transform(data, input_split=True, merge_output=True)
			if cache_features is not None and hasattr(self.extractor.transformer, "load_features"):
				self.extractor.transformer.save_features(cache_features, data)
			return data
		else:
			return batcher.merge_batches(data, batcher.backend)

	def partial_fit(self, data, y=None, input_split=False, minibatch_size=None, batcher= None):
		if batcher is None:  batcher = self.batcher
		return self.fit(data, y, input_split, reset=False, minibatch_size= minibatch_size, batcher=batcher)

	def fit_transform(self, data, y=None, cache_features=None, input_split=False, reset=True, minibatch_size=None,
	                  batcher=None):
		if batcher is None:  batcher = self.batcher
		return self.transform(data, y, cache_features, input_split, reset, update=True, minibatch_size= minibatch_size,
		                      batcher=batcher)

	def partial_fit_transform(self, data, y=None, cache_features=None, input_split=False, minibatch_size=None,
	                          batcher= None):
		if batcher is None:  batcher = self.batcher
		return self.transform(data, y, cache_features, input_split, reset=False, update=True,
		                      minibatch_size= minibatch_size, batcher=batcher)

	def __getstate__(self):
		return dict((k, v) for (k, v) in self.__dict__.items())

	def __setstate__(self, params):
		for key in params:  setattr(self, key, params[key])
