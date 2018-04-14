#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import re
import os
import wordbatch.batcher as batcher

non_alphanums= re.compile(u'[^A-Za-z0-9]+')
def default_normalize_text(text):
	return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x)>1])

class WordBatch(object):
	def __init__(self, normalize_text= default_normalize_text, max_words= 10000000, min_df= 0, max_df= 1.0,
				 spellcor_count=0, spellcor_dist=2, raw_min_df= -1, stemmer= None, extractor=None,
				 procs=0, minibatch_size= 20000, timeout= 600, spark_context= None, freeze= False,
				 method= "multiprocessing", verbose= 1):
		if procs==0:  procs= multiprocessing.cpu_count()
		self.verbose= verbose
		self.batcher= batcher.Batcher(procs=procs, minibatch_size=minibatch_size, timeout=timeout,
									  spark_context=spark_context, method=method, verbose=verbose)

		import wordbatch.transformers.apply as apply
		if normalize_text is None:  self.normalize_text= None
		else:  self.normalize_text= apply.Apply(self.batcher, normalize_text)

		import wordbatch.transformers.dictionary as dictionary
		self.dictionary= dictionary.Dictionary(self.batcher, min_df=min_df, max_df=max_df, max_words= max_words,
											   freeze= False, verbose=verbose)

		import wordbatch.transformers.tokenizer as tokenizer
		if spellcor_count>0 or stemmer!=None:
			self.tokenizer= tokenizer.Tokenizer(self.batcher, spellcor_count, spellcor_dist, raw_min_df, stemmer,
			                                    verbose= verbose)
		else: self.tokenizer= None
		self.set_extractor(extractor)
		self.freeze= freeze

	def reset(self):
		self.dictionary.reset()
		return self

	def set_extractor(self, extractor=None):
		if extractor is not None:
			if type(extractor) != tuple and type(extractor) != list:
				self.extractor = extractor(self.batcher, self.dictionary,  {})
			else:  self.extractor = extractor[0](self.batcher, self.dictionary, extractor[1])
		else: self.extractor = None

	def process(self, texts, input_split= False, reset= True, update= True):
		if reset:  self.reset()
		if self.freeze:  update= False

		if self.normalize_text is not None:
			if self.verbose > 0:  print("Normalize text")
			texts= self.normalize_text.transform(texts, input_split= input_split, merge_output= False)
			input_split= True

		if self.tokenizer is not None:
			if self.verbose > 0:  print("Tokenize text")
			if update:  texts= self.tokenizer.fit_transform(texts, input_split= input_split, merge_output= False,
															reset= reset)
			else: texts= self.tokenizer.transform(texts, input_split= input_split, merge_output= False)
			input_split= True

		if self.dictionary is not None:
			if update:
				texts= self.dictionary.fit_transform(texts, input_split=input_split, merge_output=False, reset=reset)
			if self.verbose> 2: print("len(self.dictionary.dft):", len(self.dictionary.dft))
		return texts

	def fit(self, texts, input_split= False, reset= True):
		self.process(texts, input_split, reset=reset, update= True)
		return self

	def transform(self, texts, extractor= None, cache_features= None, input_split= False, reset= False, update= False):
		if extractor== None:  extractor= self.extractor
		if cache_features != None and os.path.exists(cache_features):  return extractor.load_features(cache_features)
		if not(input_split):  texts= self.batcher.split_batches(texts)

		texts= self.process(texts, input_split=True, reset=reset, update= update)
		if extractor!= None:
			texts= extractor.transform(texts, input_split= True, merge_output= True)
			if cache_features!=None:  extractor.save_features(cache_features, texts)
			return texts
		else:
			return self.batcher.merge_batches(texts)

	def partial_fit(self, texts, input_split=False):
		return self.fit(texts, input_split, reset=False)

	def fit_transform(self, texts, extractor=None, cache_features=None, input_split=False, reset=True):
		return self.transform(texts, extractor, cache_features, input_split, reset, update=True)

	def partial_fit_transform(self, texts, extractor=None, cache_features=None, input_split=False):
		return self.transform(texts, extractor, cache_features, input_split, reset=False, update=True)

	def __getstate__(self):
		return dict((k, v) for (k, v) in self.__dict__.items())

	def __setstate__(self, params):
		for key in params:  setattr(self, key, params[key])
