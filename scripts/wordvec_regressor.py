from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import re
import json
import gzip
from wordbatch.pipelines import WordBatch
from wordbatch.models import FTRL
from wordbatch.extractors import WordVec, Hstack
from wordbatch.data_utils import shuffle
import threading
import sys
if sys.version_info.major == 3:
	import pickle as pkl
else:
	import cPickle as pkl

non_alphanums = re.compile('[\W+]')
nums_re= re.compile("\W*[0-9]+\W*")
trash_re= [re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "), re.compile("[-']{2,}"),
		   re.compile(" '"),re.compile("  +")]

def normalize_text(text):
	text= text.lower()
	text= nums_re.sub(" NUM ", text)
	text= " ".join([word for word in non_alphanums.sub(" ",text).strip().split() if len(word)>1])
	return text

class WordvecRegressor(object):
	def __init__(self, pickle_model="", datadir=None, batcher=None):
		self.wb= WordBatch(normalize_text, extractor=Hstack([
		    WordVec(wordvec_file="../../../data/word2vec/glove.twitter.27B.100d.txt.gz", normalize_text=normalize_text,
		            encoding="utf8"),
		    WordVec(wordvec_file="../../../data/word2vec/glove.6B.50d.txt.gz", normalize_text=normalize_text,
		            encoding="utf8")]))
		# from wordbatch.pipelines import FeatureUnion
		# from wordbatch.transformers import Dictionary, TextNormalizer
		# from sklearn.pipeline import Pipeline
		# tn= TextNormalizer(normalize_text=normalize_text)
		# dct= Dictionary()
		# vec1= WordVec(wordvec_file="../../../data/word2vec/glove.twitter.27B.100d.txt.gz",
		# 			  normalize_text=normalize_text, encoding="utf8", dictionary= dct)
		# vec2= WordVec(wordvec_file="../../../data/word2vec/glove.6B.50d.txt.gz",
		# 			  normalize_text=normalize_text, encoding="utf8", dictionary= dct)
		# self.wb = Pipeline(steps= [("tn", tn), ("dct", dct), ("vecs", FeatureUnion([("vec1", vec1), ("vec2", vec2)]))])
		self.batcher= batcher

		self.clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=100+50, iters=1, inv_link= "identity")

		if datadir==None:  (self.wb, self.clf)= pkl.load(gzip.open(pickle_model, 'rb'))
		else: self.train(datadir, pickle_model)

	def fit_batch(self, texts, labels, rcount):
		texts, labels = shuffle(texts, labels, seed=rcount)
		print("Transforming", rcount)
		#texts= self.wb.fit_transform(texts, tn__batcher=self.batcher, dct__reset= False, dct__batcher= self.batcher)
		texts = self.wb.fit_transform(texts)
		print("Training", rcount)
		self.clf.fit(texts, labels, reset= False)

	def train(self, datadir, pickle_model=""):
		texts= []
		labels= []
		training_data= os.listdir(datadir)
		rcount= 0
		batchsize= 80000

		p= None
		for jsonfile in training_data:
			with open(datadir + "/" + jsonfile, 'r') as inputfile:
				for line in inputfile:
					#if rcount > 1000000: break
					try: line= json.loads(line.strip())
					except:  continue
					for review in line["Reviews"]:
						rcount+= 1
						if rcount % 100000 == 0:  print(rcount)
						if rcount % 6 != 0: continue
						if "Overall" not in review["Ratings"]: continue
						texts.append(review["Content"])
						labels.append((float(review["Ratings"]["Overall"]) - 3) *0.5)
						if len(texts) % batchsize == 0:
							if p != None:  p.join()
							p= threading.Thread(target=self.fit_batch, args=(texts, labels, rcount))
							p.start()
							texts= []
							labels= []
		if p != None:  p.join()
		self.fit_batch(texts, labels, rcount)

		# if pickle_model!="":
		# 	with gzip.open(pickle_model, 'wb') as model_file:
		# 		backend = self.wb.batcher.backend
		# 		backend_handle = self.wb.batcher.backend_handle
		# 		self.wb.batcher.backend = "serial"
		# 		self.wb.batcher.backend_handle = None
		# 		pkl.dump((self.wb, self.clf), model_file, protocol=2)
		# 		self.wb.batcher.backend = backend
		# 		self.wb.batcher.backend_handle = backend_handle

	def predict(self, texts):
		vecs= self.wb.transform(texts)
		return self.clf.predict(vecs)