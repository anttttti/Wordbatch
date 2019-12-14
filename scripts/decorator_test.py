import re
from contextlib import contextmanager
import time
from wordbatch.pipelines import decorator_apply as apply
from wordbatch.batcher import Batcher
import warnings
import pandas as pd
from nltk.stem.porter import PorterStemmer
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
	['serial', ""],
	###['multiprocessing', ""], #doesn't serialize lambda functions
	['loky', ""],
	['ray', ray]
]

options= [
	['none', {}],
	['cache', {"cache":1000}],
	['vectorize', {"vectorize":True}],
	['both', {"cache":1000, "vectorize":True}]
]

#data_sizes= [10000]
data_sizes= [1280000]

if __name__ == '__main__':
	for data_size in data_sizes:
		texts_chunk = texts[:data_size]
		df = pd.DataFrame(index=range(len(texts_chunk)))
		for backend in backends:
			batcher = Batcher(procs=16, minibatch_size=5000, backend=backend[0], backend_handle=backend[1])
			for option in options:
				try:
					with timer("Text normalization: " + option[0] + "," + str(len(texts_chunk)) + "," + backend[0])\
							, warnings.catch_warnings():
						warnings.simplefilter("ignore")
						df['text']= apply(normalize_text, batcher, **option[1])(texts_chunk)
				except Exception as e:
					print("Failed text normalization: "+ option[0] + "," + str(len(texts_chunk)) + "," + backend[0])
					      #"Exception:", e.split("\n")[0])

				try:
					with timer("Splitting first word: " + option[0] + "," + str(len(texts_chunk)) + "," + backend[0])\
							, warnings.catch_warnings():
						warnings.simplefilter("ignore")
						df['first_word']= apply(lambda x: x.split(" ")[0], batcher, **option[1])(df['text'])
				except Exception as e:
					print("Failed splitting first word: " + option[0] + "," + str(len(texts_chunk)) + "," + backend[0])
					      #"Exception:", e.split("\n")[0])

				try:
					with timer("Stemming first word: "+option[0]+","+str(len(texts_chunk))+","+backend[0]), \
					     warnings.catch_warnings():
						warnings.simplefilter("ignore")
						df['first_word_stemmed'] = apply(stemmer.stem, batcher, **option[1])(df['first_word'])
				except Exception as e:
					print("Failed stemming first word: " + option[0] + "," + str(len(texts_chunk)) + "," + backend[0])
					      #"Exception:", e.split("\n")[0])

				try:
					with timer("Groupby aggregation: "+option[0]+","+str(len(texts_chunk))+","+backend[0]), \
					     warnings.catch_warnings():
						warnings.simplefilter("ignore")
						group_ids, groups = zip(*df.groupby('first_word_stemmed'))
						res= apply(lambda x:x['text'].str.len().agg('mean'), batcher, **option[1])(groups)
						df['first_word_stemmed_mean_text_len']= df['first_word_stemmed'].map(
							{x:y for x, y in zip(group_ids, res)})
						# df['first_word_stemmed_mean_text_len'] = df['first_word_stemmed'].map(
						# 	{x: y for x, y in zip(list(zip(*df.groupby('first_word_stemmed')))[0],
						# 	                      apply(lambda x:x['text'].str.len().agg('mean'), batcher, **option[1])(
						# 		                      list(zip(*df.groupby('first_word_stemmed')))[1]))})
				except Exception as e:
					print("Failed groupby aggregation: " + option[0] + "," + str(len(texts_chunk)) + "," + backend[0])
					      #"Exception:", e.split("\n")[0])
