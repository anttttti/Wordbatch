import os
import re
import json
import pickle as pkl
import gzip
from sklearn.linear_model import *
import scipy as sp
import wordbatch

non_alphanums = re.compile(u'[\W]')
nums_re= re.compile("\W*[0-9]+\W*")
trash_re= [re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "), re.compile("[-']{2,}"),
           re.compile(" '"),re.compile("  +")]

def normalize_text(text):
    text= text.lower()
    text= nums_re.sub(" NUM ", text)
    text= " ".join([word for word in non_alphanums.sub(" ",text).split() if len(word)>1])
    return text

class WordvecRegressor(object):
    def __init__(self, pickle_model="", datadir=None):
        self.wordbatch= wordbatch.WordBatch(normalize_text, procs=8,
                                            extractors=[
                        ("wordvec", {"wordvec_file": "data/word2vec/glove.twitter.27B.100d.txt.gz",
                                    "normalize_text": normalize_text}),
                        ("wordvec", {"wordvec_file": "data/word2vec/glove.6B.50d.txt.gz",
                                    "normalize_text": normalize_text})
                     ])
        self.wordbatch.dictionary_freeze= True
        self.clf= SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                                           n_iter=1, shuffle=True, verbose=0, epsilon=0.1, random_state=0,
                                           learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False,
                                            average=False)

        if datadir==None:  (self.wordbatch, self.clf)= pkl.load(gzip.open(pickle_model, u'rb'))
        else: self.train(datadir, pickle_model)

    def train(self, datadir, pickle_model=""):
        texts= []
        labels= []
        training_data= os.listdir(datadir)
        rcount= 0
        batchsize= 100000

        for jsonfile in training_data:
            with open(datadir + "/" + jsonfile, u'r') as inputfile:
                for line in inputfile:
                    #if rcount > 1000000: break
                    try: line= json.loads(line.strip())
                    except:  continue
                    for review in line["Reviews"]:
                        rcount+= 1
                        if rcount % 10000 == 0:  print rcount
                        if rcount % 6 != 0: continue
                        if "Overall" not in review["Ratings"]: continue
                        texts.append(review["Content"])
                        labels.append((float(review["Ratings"]["Overall"]) - 3) *0.5)
                        if len(texts) % batchsize == 0:
                            vecs= self.wordbatch.transform(texts)
                            del (texts)
                            if len(vecs)>1:  vecs= sp.hstack(vecs)
                            self.clf.partial_fit(vecs, labels)
                            del(vecs)
                            del(labels)
                            texts= []
                            labels= []
        if len(texts)>0:
            vecs= self.wordbatch.transform(texts)
            del(texts)
            if len(vecs) > 1:  vecs= sp.hstack(vecs)
            self.clf.partial_fit(vecs, labels)
            del(vecs)
            del(labels)

        if pickle_model!="":
            with gzip.open(pickle_model, u'wb') as model_file:
                pkl.dump((self.wordbatch, self.clf), model_file, protocol=2)

    def predict(self, texts):
        vecs= self.wordbatch.transform(texts)
        if len(vecs) > 1:  vecs= sp.hstack(vecs)
        return self.clf.predict(vecs)