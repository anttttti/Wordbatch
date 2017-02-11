import os
import re
import json
import pickle as pkl
import gzip
import wordbatch
from wordbatch.models import FTRL
from wordbatch.extractors import WordBag
import threading

non_alphanums = re.compile(u'[\W+]')
nums_re= re.compile("\W*[0-9]+\W*")
triples_re= re.compile(ur"(\w)\1{2,}")
trash_re= [re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "), re.compile("[-']{2,}"),
           re.compile(" '"),re.compile("  +")]
from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer()
def normalize_text(text):
    text= text.lower()
    text= nums_re.sub(" NUM ", text)
    text= " ".join([word for word in non_alphanums.sub(" ",text).strip().split() if len(word)>1])
    return text

class WordbagRegressor(object):
    def __init__(self, pickle_model="", datadir=None):
        self.wordbatch = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams":3,
          "hash_ngrams_weights":[-1.0, -1.0, 1.0],"hash_size":2**23, "norm":'l2', "tf":'binary', "idf":50.0}))
        self.clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=1, inv_link="identity")
        if datadir==None:  (self.wordbatch, self.clf)= pkl.load(gzip.open(pickle_model, u'rb'))
        else: self.train(datadir, pickle_model)

    def fit_batch(self, texts, labels, rcount):
        texts, labels= self.wordbatch.shuffle_batch(texts, labels, rcount)
        print "Transforming", rcount
        texts= self.wordbatch.transform(texts)
        print "Training", rcount
        self.clf.fit(texts, labels)

    def train(self, datadir, pickle_model=""):
        texts= []
        labels= []
        training_data= os.listdir(datadir)
        rcount= 0
        batchsize= 100000

        p = None
        for jsonfile in training_data:
            with open(datadir + "/" + jsonfile, u'r') as inputfile:
                for line in inputfile:
                    #if rcount > 1000000: break
                    try: line = json.loads(line.strip())
                    except:  continue
                    for review in line["Reviews"]:
                        rcount+= 1
                        if rcount % 100000 == 0:  print rcount
                        if rcount % 7 != 0: continue
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

        self.wordbatch.dictionary_freeze= True

        if pickle_model!="":
            with gzip.open(pickle_model, u'wb') as model_file:
                pkl.dump((self.wordbatch, self.clf), model_file, protocol=2)

    def predict(self, texts):
        counts= self.wordbatch.transform(texts)
        return self.clf.predict(counts)

    def predict_parallel(self, texts):
        counts= self.wordbatch.transform(texts)
        return self.wordbatch.predict_parallel(counts, self.clf)