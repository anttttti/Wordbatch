import re
from contextlib import contextmanager
import time
from wordbatch.pipelines import decorator_apply as apply
from wordbatch.batcher import Batcher
import warnings
import pandas as pd
from nltk.stem.porter import PorterStemmer
from numba import int64, float64
import os
import json

tripadvisor_dir= "../data/tripadvisor/json"

import ray
#ray start --head --node-ip-address 169.254.93.14
#ray.init(redis_address=scheduler_ip+":57113") #Change port accordingly
ray.init()

@contextmanager
def timer(name):
	t0 = time.time()
	yield
	print(name + " done in " + str(time.time() - t0) + "s")

if 1==1:
	texts= []
	for jsonfile in os.listdir(tripadvisor_dir):
		with open(tripadvisor_dir + "/" + jsonfile, 'r') as inputfile:
			for line in inputfile:
				try:
					line = json.loads(line.strip())
				except:
					continue
				for review in line["Reviews"]:
					texts.append(review["Content"])
# 	pd.to_pickle(texts, "tripadvisor_data.pkl")
# else:
# 	texts= pd.read_pickle("tripadvisor_data.pkl")

non_alphanums = re.compile('[\W+]')
stemmer= PorterStemmer()

def normalize_text(text):
	text= " ".join([word for word in non_alphanums.sub(" ",text.lower()).strip().split() if len(word)>1])
	return text

print(len(texts))
backends= [
	###['multiprocessing', ""], #doesn't serialize lambda functions
	['ray', ray]
	['loky', ""],
	['serial', ""],
]

#data_size= 200000
#data_size= 500000
data_size= 1280000

def test_backend(texts, backend):
	df = pd.DataFrame(texts, columns=['text'])
	df['text']= df['text'].fillna("")
	batcher = Batcher(minibatch_size=5000, backend=backend[0], backend_handle=backend[1])
	#batcher = Batcher(minibatch_size=data_size//8, backend=backend[0], backend_handle=backend[1])
	if backend[0]=="ray":
		backend[1].shutdown()
		backend[1].init()

	try:
		with timer("Text normalization: " + str(len(df)) + "," + backend[0]), warnings.catch_warnings():
			warnings.simplefilter("ignore")
			df['text_normalized'] = apply(normalize_text, batcher)(df['text'])
		with timer("Text normalization without Wordbatch: " + str(len(df)) + "," + backend[0]) \
				, warnings.catch_warnings():
			warnings.simplefilter("ignore")
			df['text_normalized'] = [normalize_text(x) for x in df['text']]
	except Exception as e:
		print("Failed text normalization: " +"," + str(len(df)) + "," + backend[0])
	# #"Exception:", e.split("\n")[0])

	try:
		def div(x, y):
			return 0 if y==0 else x / y
		df['len_text'] = df['text'].str.len().astype(int)
		df['len_text_normalized'] = df['text_normalized'].str.len().astype(int)
		# list(zip(df['text_normalized'], df['text_normalized']))
		# np.vstack([df['text_normalized'], df['text_normalized']]).T
		with timer("Text length ratio vectorized: " + str(len(df)) + "," + backend[0]), warnings.catch_warnings():
			df['len_ratio'] = apply(div, batcher, vectorize=[float64(int64, int64)])(
				df[['len_text', 'len_text_normalized']].values)
		with timer("Text length ratio without vectorization: " + str(len(df)) + "," + backend[0]), \
		     warnings.catch_warnings():
			df['len_ratio'] = apply(lambda x:div(*x), batcher)(df[['len_text', 'len_text_normalized']].values)
		with timer("Text length ratio without Wordbatch: " + str(len(df)) + "," + backend[0]), warnings.catch_warnings():
			df['len_ratio'] = [div(x, y) for x, y in zip(df['len_text'], df['len_text_normalized'])]
	except Exception as e:
		print("Failed text length ratios: " +"," + str(len(df)) + "," + backend[0])
	# #return

	try:
		with timer("Splitting first word: " + str(len(df)) + "," + backend[0]) \
				, warnings.catch_warnings():
			warnings.simplefilter("ignore")
			df['first_word'] = apply(lambda x: x.split(" ")[0], batcher)(df['text_normalized'])
		with timer("Splitting first word without Wordbatch: " + str(len(df)) + "," + backend[0]), \
			 warnings.catch_warnings():
			warnings.simplefilter("ignore")
			df['first_word'] = [x.split(" ")[0] for x in df['text_normalized']]
	except Exception as e:
		print("Failed splitting first word: " + str(len(df)) + "," + backend[0])
	# "Exception:", e.split("\n")[0])

	try:
		with timer("Stemming first word: " + str(len(df)) + "," + backend[0]), \
		   warnings.catch_warnings():
			warnings.simplefilter("ignore")
			df['first_word_stemmed'] = apply(stemmer.stem, batcher)(df['first_word'])
		with timer("Stemming first word without Wordbatch: " + str(len(df)) + "," + backend[0]), \
		   warnings.catch_warnings():
			warnings.simplefilter("ignore")
			df['first_word_stemmed'] = [stemmer.stem(x) for x in df['first_word']]
	except Exception as e:
		print("Failed stemming first word: " + str(len(df)) + "," + backend[0])
	# # "Exception:", e.split("\n")[0])
	#
	try:
		with timer("Stemming first word (cache=1000): " + str(len(df)) + "," + backend[0]), \
		   warnings.catch_warnings():
			warnings.simplefilter("ignore")
			df['first_word_stemmed'] = apply(stemmer.stem, batcher, cache=1000)(df['first_word'])

		with timer("Stemming first word (cache=1000) without Wordbatch: " + str(len(df)) + "," + backend[0]), \
		   warnings.catch_warnings():
			warnings.simplefilter("ignore")
			from functools import lru_cache
			cache_stem= lru_cache(maxsize=1000)(stemmer.stem)
			df['first_word_stemmed'] = [cache_stem(x) for x in df['first_word']]
	except Exception as e:
		print("Failed stemming first word: " + str(len(df)) + "," + backend[0])
	# "Exception:", e.split("\n")[0])

	try:
		batcher.minibatch_size = 200
		with timer("Groupby aggregation: " + str(len(df)) + "," + backend[0]), \
		   warnings.catch_warnings():
			warnings.simplefilter("ignore")
			group_ids, groups = zip(*df[['first_word_stemmed', 'text']].groupby('first_word_stemmed'))
			res = apply(lambda x: x['text'].str.len().agg('mean'), batcher)(groups)
			df['first_word_stemmed_mean_text_len'] = df['first_word_stemmed'].map(
				{x: y for x, y in zip(group_ids, res)})

		batcher.minibatch_size = 10
		df['first_word_stemmed_hashbin'] = [hash(x) % 500 for x in df['first_word_stemmed']]
		with timer("Groupby aggregation hashbin: " + str(len(df)) + "," + backend[0]), \
		     warnings.catch_warnings():
			warnings.simplefilter("ignore")
			group_ids, groups = zip(*df[['first_word_stemmed', 'text', 'first_word_stemmed_hashbin']]
			                        .groupby('first_word_stemmed_hashbin'))
			res = pd.concat(apply(lambda x: x.groupby('first_word_stemmed').apply(
				lambda z: z['text'].str.len().agg('mean')), batcher)(groups))
			df['first_word_stemmed_mean_text_len'] = df['first_word_stemmed'].map(res)

		with timer("Groupby aggregation without Wordbatch: " + str(len(df)) + "," + backend[0]), \
			 warnings.catch_warnings():
			warnings.simplefilter("ignore")
			group_ids, groups = zip(*df[['first_word_stemmed', 'text']].groupby('first_word_stemmed'))
			res = [x['text'].str.len().agg('mean') for x in groups]
			df['first_word_stemmed_mean_text_len'] = df['first_word_stemmed'].map(
				{x: y for x, y in zip(group_ids, res)})
		del (res, group_ids, groups, df, batcher)
	except Exception as e:
		print("Failed groupby aggregation: " + str(len(df)) + "," + backend[0])
	# "Exception:", e.split("\n")[0])

texts= texts[:data_size]
if __name__ == '__main__':
	for backend in backends:
		test_backend(texts, backend)
