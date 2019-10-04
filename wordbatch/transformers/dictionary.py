#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from collections import defaultdict
import operator

def batch_get_dfs(args):
	dft= defaultdict(int)
	for text in args[0]:
		if type(text)==str:
			for word in set(text.split(" ")):  dft[word]+= 1
		else:
			dft[str(text)]+= 1
	return [dict(dft), len(args[0])]

class Dictionary(object):
	def __init__(self, min_df=0, max_df=1.0, max_words= 10000000000000, freeze= False, encode=True, verbose=0):
		self.verbose = verbose
		self.freeze = freeze
		self.max_words = max_words
		self.min_df = min_df
		self.max_df = max_df
		self.encode= encode
		self.word2id= None
		self.reset()

	def reset(self):
		if self.encode:  self.word2id = {}
		self.dft = {}
		self.doc_count = 0
		return self

	def get_pruning_dft(self, dft):
		sorted_dft = sorted(list(dft.items()), key=operator.itemgetter(1,0), reverse=True)
		if type(self.min_df) == type(1):  min_df2 = self.min_df
		else:  min_df2 = self.doc_count * self.min_df
		if type(self.max_df) == type(1):   max_df2 = self.max_df
		else:  max_df2 = self.doc_count * self.max_df
		return sorted_dft, min_df2, max_df2

	def prune_dictionary(self, max_words=None, min_df=None, max_df=None, re_encode= False, prune_dfs= True,
						 set_max_words= True):
		#Prune dictionary. Optionally prune document frequency table as well
		if max_words is not None: self.max_words= max_words
		if min_df is not None: self.min_df= min_df
		if max_df is not None: self.max_df= max_df
		max_words= self.max_words
		word2id = self.word2id
		dft = self.dft
		sorted_dft, min_df2, max_df2 = self.get_pruning_dft(dft)
		c= 0
		#print(len(sorted_dft), len(self.word2id), len(self.raw_dft))
		for word, df in sorted_dft:
			if word2id is not None:
				if word not in word2id:
					if re_encode:  word2id[word]= -1
					else:  continue
			c+= 1
			if c > max_words or df < min_df2 or df > max_df2:
				if prune_dfs: dft.pop(word)
				if word2id is not None:  word2id.pop(word)
			elif re_encode:
				if word2id is not None:  word2id[word]= c
		if set_max_words and word2id is not None:  self.max_words= len(word2id)

	def fit(self, data, y=None, input_split= False, reset= False, minibatch_size=None, batcher= None):
		if reset:  self.reset()
		if self.word2id is None:
			self.word2id = {}
		word2id= self.word2id
		if batcher is None:  dfts, doc_counts= zip(*batch_get_dfs(data))
		else:
			# import wordbatch.pipelines
			# dfts, doc_counts = zip(*batcher.collect_batches(
			# 	wordbatch.pipelines.apply_batch.ApplyBatch(get_dfs, batcher=batcher).transform(
			# 		data, input_split=input_split, merge_output=False)
			# ))
			dfts, doc_counts= zip(*batcher.collect_batches(
				batcher.process_batches(batch_get_dfs, data, [], input_split= input_split, merge_output=False,
				                        minibatch_size=minibatch_size)))
		self.doc_count += sum(doc_counts)
		dft = defaultdict(int, self.dft)
		for dft2 in dfts:
			for k, v in dft2.items():  dft[k] += v
		if word2id is not None:
			#Add entries. Online pruning only used to prevent inclusion into dictionary
			sorted_dft, min_df2, max_df2 = self.get_pruning_dft(dft)
			for word, df in sorted_dft:
				if len(word2id)>= self.max_words: break
				if df<min_df2 or df>max_df2: continue
				if word in word2id:  continue
				word2id[word] = len(word2id)+1
				if self.verbose>2: print("Add word to dictionary:", word, dft[word], word2id[word])
		self.dft= dict(dft)
		return self

	def partial_fit(self, data, y=None, input_split=False, minibatch_size=None, batcher=None):
		return self.fit(data, y, input_split, reset=False, minibatch_size=minibatch_size, batcher=batcher)

	def fit_transform(self, data, y=None, input_split= False, merge_output= True, reset= True, minibatch_size=None,
	                  batcher= None):
		self.fit(data, y=y, input_split= input_split, reset=reset, minibatch_size= minibatch_size, batcher=batcher)
		return self.transform(data, y=y, input_split= input_split, merge_output= merge_output, batcher= None)

	def partial_fit_transform(self, data, y=None, input_split=False, minibatch_size=None,
	                          batcher=None):
		return self.transform(data, y, input_split, reset=False, update=True, batcher=batcher)

	def transform(self, data, y=None, input_split= False, merge_output= True, minibatch_size=None,
	              batcher= None):
		if input_split and merge_output and batcher is not None:  data= batcher.merge_batches(data)
		return data