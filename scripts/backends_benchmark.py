import re
from contextlib import closing, contextmanager
import time
from wordbatch.pipelines import WordBatch, Apply, ApplyBatch
from wordbatch.extractors import WordHash, WordBag
from wordbatch.transformers import Tokenizer, Dictionary
from wordbatch.batcher import Batcher
import os
import json
from sklearn.feature_extraction.text import HashingVectorizer
import warnings
import pandas as pd

tripadvisor_dir= "../data/tripadvisor/json"

#Configure below to allow Dask / Spark
# scheduler_ip= "169.254.93.14"
# from dask.distributed import Client
# #dask-scheduler --host 169.254.93.14
# #dask-worker 169.254.93.14:8786 --nprocs 16
# dask_client = Client(scheduler_ip+":8786")
#
# from pyspark import SparkContext, SparkConf
# # conf= SparkConf().setAll([('spark.executor.memory', '4g'), ('spark.driver.memory', '30g'),
# # 						  ('spark.driver.maxResultSize', '10g')])
# import os
# os.environ['PYSPARK_PYTHON'] = '/home/USERNAME/anaconda3/envs/ENV_NAME/bin/python'
# conf= SparkConf().setAll([('spark.executor.memory', '4g'), ('spark.driver.memory', '30g'),
# 						  ('spark.driver.maxResultSize', '10g')]).setMaster("spark://169.254.93.14:7077")
# spark_context = SparkContext(conf=conf)

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
nums_re= re.compile("\W*[0-9]+\W*")
triples_re= re.compile(r"(\w)\1{2,}")
trash_re= [re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "), re.compile("[-']{2,}"),
		   re.compile(" '"),re.compile("  +")]
from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer()

def normalize_text(text):
	text= text.lower()
	text= nums_re.sub(" NUM ", text)
	text= " ".join([word for word in non_alphanums.sub(" ",text).strip().split() if len(word)>1])
	return text

print(len(texts))
backends= [
	['serial', ""],
	['multiprocessing', ""],
	['loky', ""],
	#['dask', dask_client], #Uncomment once configured
	#['spark', spark_context], #Uncomment once configured
	['ray', ray]
]

tasks= [
	"ApplyBatch",
	"WordBag",
]

data_sizes= [40000, 80000, 160000, 320000, 640000, 1280000]

for task in tasks:
	for data_size in data_sizes:
		texts_chunk = texts[:data_size]
		print("Task:", task, "Data size:", data_size)
		for backend in backends:
			batcher = Batcher(procs=16, minibatch_size=5000, backend=backend[0], backend_handle=backend[1])
			#try:
			with timer("Completed: ["+task+","+str(len(texts_chunk))+","+backend[0]+"]"), warnings.catch_warnings():
				warnings.simplefilter("ignore")
				if task=="ApplyBatch":
					hv = HashingVectorizer(decode_error='ignore', n_features=2 ** 25, preprocessor=normalize_text,
										   ngram_range=(1, 2), norm='l2')
					t= ApplyBatch(hv.transform, batcher=batcher).transform(texts_chunk)
					print(t.shape, t.data[:5])

				if task=="WordBag":
					wb = WordBatch(normalize_text=normalize_text,
					               dictionary=Dictionary(min_df=10, max_words=1000000, verbose=0),
					               tokenizer= Tokenizer(spellcor_count=2, spellcor_dist=2, raw_min_df= 2,
									 					stemmer= stemmer),
					               extractor=WordBag(hash_ngrams=0, norm= 'l2', tf= 'binary', idf= 50.0),
					               batcher= batcher,
					               verbose= 0)
					t = wb.fit_transform(texts_chunk)
					print(t.shape, t.data[:5])
			# except:
			# 	print("Failed ["+task+","+str(len(texts_chunk))+","+backend[0]+"]")
		print("")