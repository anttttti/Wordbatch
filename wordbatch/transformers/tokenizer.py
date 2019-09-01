#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
#from nltk.metrics import edit_distance
import Levenshtein #python-Levenshtein
from collections import defaultdict

def batch_get_dfs(args):
	dft= defaultdict(int)
	for text in args[0]:
		for word in set(text.split(" ")):  dft[word]+= 1
	return [dict(dft), len(args[0])]

def correct_spelling(word, dft, corrections_index, spellcor_count, spellcor_dist):
	#T. Bocek, E. Hunt, B. Stiller: Fast Similarity Search in Large Dictionaries, 2007
	if dft.get(word, 0)>spellcor_count or len(word)<3:  return word
	max_df= -100000000000000
	max_word= word
	spell_suggestions= get_deletions(word, spellcor_dist)
	candidates= {}
	for x in spell_suggestions:
		if x in corrections_index:
			for y in corrections_index[x]:  candidates[y]= 1
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

def make_corrections_index(dft, spellcor_count, spellcor_dist):
	dft2 = {w[0]: w[1] for w in dft.items() if w[1] > spellcor_count}
	corrections_index = defaultdict(list)
	for word in dft2:
		if len(word) > 15:  continue
		for word2 in get_deletions(word, spellcor_dist):
			corrections_index[word2].append(word)
	return corrections_index

class Tokenizer(object):
	def __init__(self, spellcor_count=0, spellcor_dist=2, stemmer= None, freeze= False, verbose= 0):
		self.verbose= verbose
		self.freeze= freeze
		if spellcor_count == 0:
			spellcor_dist = 0
		elif spellcor_dist == 0:
			spellcor_count = 0
		self.spellcor_count = spellcor_count
		self.spellcor_dist = spellcor_dist
		self.stemmer = stemmer
		self.reset()

	def reset(self):
		self.dft = {}
		self.doc_count = 0
		return self

	def fit(self, data, y= None, input_split= False, reset= True, minibatch_size=None, batcher= None):
		if reset:  self.reset()
		if self.freeze:  return self
		if batcher is None:  dfts, doc_counts= zip(*batch_get_dfs(data))
		else:
			dfts, doc_counts= zip(*batcher.collect_batches(
				batcher.process_batches(batch_get_dfs, data, [], input_split= input_split, merge_output=False,
				                        minibatch_size=minibatch_size)))
		self.doc_count += sum(doc_counts)
		dft = defaultdict(int, self.dft)
		for dft2 in dfts:
			for k, v in dft2.items():  dft[k] += v
		self.dft= dict(dft)
		return self

	def partial_fit(self, data, y=None, input_split=False, minibatch_size=None, batcher=None):
		return self.fit(data, y, input_split, reset=False, minibatch_size= minibatch_size, batcher=batcher)

	def fit_transform(self, data, y=None, input_split= False, merge_output= True, reset= True,
	                  minibatch_size=None, batcher= None):
		self.fit(data, y=y, input_split= input_split, reset=reset,  minibatch_size= minibatch_size, batcher=batcher)
		return self.transform(data, y=y, input_split=input_split, merge_output=merge_output,
		                      minibatch_size=minibatch_size, batcher=batcher)

	def partial_fit_transform(self, data, y=None, input_split=False, minibatch_size=None, batcher=None):
		return self.transform(data, y, input_split, reset=False, update=True, minibatch_size=minibatch_size,
		                      batcher=batcher)

	def transform(self, X, y=None, input_split= False, merge_output= True, minibatch_size=None, batcher= None):
		if self.verbose > 0:  print("Make word normalization dictionary")
		do_corrections= 1 if (self.spellcor_count > 0) and (self.spellcor_dist>0) else 0
		if not(do_corrections) and self.stemmer is None:  return X
		if do_corrections:
			corrections_index= make_corrections_index(self.dft, self.spellcor_count, self.spellcor_dist)
		if self.stemmer is not None:
			if do_corrections:
				corrs = {word: self.stemmer.stem(correct_spelling(
					word, self.dft, corrections_index, self.spellcor_count, self.spellcor_dist)) for word in self.dft}
			else:  corrs = {word: self.stemmer.stem(word) for word in self.dft}
		elif do_corrections:
			corrs = {word: correct_spelling(
				word, self.dft, corrections_index, self.spellcor_count, self.spellcor_dist) for word in self.dft}
		corrs = {key: value for key, value in corrs.items() if key != value}
		if self.verbose > 0:  print("Make word normalizations")
		if batcher is None:  return batch_correct_spellings(X)
		return batcher.process_batches(batch_correct_spellings, X, [corrs], input_split=input_split,
		                               merge_output=merge_output, minibatch_size= minibatch_size)
