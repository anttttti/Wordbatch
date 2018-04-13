#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
#from nltk.metrics import edit_distance
import Levenshtein #python-Levenshtein
from collections import defaultdict
from collections import Counter

WB_DOC_CNT= u'###DOC_CNT###' #Used for Spark document counting across RDFs

def batch_get_dfs(args):
	dft= Counter()
	for text in args[0]:
		for word in set(text.split(" ")):  dft[word]+= 1
	dft[WB_DOC_CNT]+= len(args[0]) #Avoid Spark collect() by counting here
	return dft

def correct_spelling(word, dft, spell_index, spellcor_count, spellcor_dist):
	#T. Bocek, E. Hunt, B. Stiller: Fast Similarity Search in Large Dictionaries, 2007
	if dft.get(word, 0)>spellcor_count or len(word)<3:  return word
	max_df= -100000000000000
	max_word= word
	spell_suggestions= get_deletions(word, spellcor_dist)
	candidates= {}
	for x in spell_suggestions:
		if x in spell_index:
			for y in spell_index[x]:  candidates[y]= 1
	#for word2 in list(candidates.keys()):
	for word2 in candidates:
		#score= edit_distance(word, word2, True)
		score= Levenshtein.distance(word, word2)
		if score>spellcor_dist:  continue
		#score = float(dft[word2]) / score
		score= dft[word2]
		#score = Levenshtein.jaro_winkler(word, word2)
		#score= dft[word2]*Levenshtein.jaro_winkler(word, word2)
		if score > max_df:
			max_df= score
			max_word= word2
	return max_word

def batch_correct_spellings(args):
	corrs= args[1]
	return [u" ".join([corrs.get(word, word) for word in text.split(" ")]) for text in args[0]]

def get_deletions(word, order):
	stack = {word: order}
	results = {}
	while len(stack) > 0:
		stack2 = {}
		for word2 in stack:
			order2 = stack[word2] - 1
			for x in range(len(word2)):
				if order2 != 0:  stack2[word2[:x] + word2[x + 1:]] = order2
				results[word2[:x] + word2[x + 1:]] = 1
		stack = stack2
	return list(results.keys())

class Tokenizer(object):
	def __init__(self, batcher, spellcor_count=0, spellcor_dist=2, raw_min_df= 0, stemmer= None, freeze= False,
	             verbose= 1):
		self.verbose= verbose
		self.freeze= freeze
		if spellcor_count == 0:
			spellcor_dist = 0
		elif spellcor_dist == 0:
			spellcor_count = 0
		self.spellcor_count = spellcor_count
		self.spellcor_dist = spellcor_dist
		self.stemmer = stemmer
		self.raw_min_df = raw_min_df
		self.batcher= batcher
		self.reset()

	def reset(self):
		self.dft = Counter()
		self.doc_count = 0
		return self

	def fit(self, data, input_split= False, reset= True):
		if reset:  self.reset()
		if self.freeze:  return self
		dft = self.dft
		dfts = self.batcher.parallelize_batches(batch_get_dfs, data, [], input_split=input_split, merge_output=False)
		if self.batcher.spark_context is not None:  dfts = [batch[1] for batch in dfts.collect()]
		self.doc_count += sum([dft2.pop(WB_DOC_CNT) for dft2 in dfts])
		for dft2 in dfts:  dft.update(dft2)
		return self

	def fit_transform(self, data, input_split= False, merge_output= True, reset= True):
		self.fit(data, input_split, reset)
		return self.transform(data, input_split, merge_output)

	def transform(self, data, input_split= False, merge_output= True):
		if self.verbose > 0:  print("Make word normalization dictionary")
		if self.spellcor_dist > 0:
			dft2 = {word: self.dft[word] for word in self.dft if self.dft[word] > self.spellcor_count}
			spell_index = defaultdict(list)
			for word in dft2:
				if len(word) > 15:  continue
				for word2 in get_deletions(word, self.spellcor_dist):
					spell_index[word2].append(word)
		if self.stemmer != None:
			if self.spellcor_count > 0:
				corrs = {word: self.stemmer.stem(correct_spelling(
					word, dft2, spell_index, self.spellcor_count, self.spellcor_dist)) for word in self.dft}
			else:  corrs = {word: self.stemmer.stem(word) for word in self.dft}
		else:
			corrs = {word: correct_spelling(
				word, dft2, spell_index, self.spellcor_count, self.spellcor_dist) for word in self.dft}
		corrs = {key: value for key, value in corrs.items() if key != value}
		if self.verbose > 0:  print("Make word normalizations")
		return self.batcher.parallelize_batches(batch_correct_spellings, data, [corrs],
										input_split=input_split, merge_output=merge_output)
# import wordbatch.batcher as batcher
# b= batcher.Batcher(method="serial")
# t= [[1, 2], [3, 4]]
# import numpy as np
# a= Apply(b, np.power, [2],{})
# print(a.transform(t))