import os
import re
import json
import pickle as pkl
import gzip
from sklearn.linear_model import *
import scipy.sparse as ssp
import wordbatch

non_alphanums = re.compile(u'[\W]')
nums_re= re.compile("\W*[0-9]+\W*")
triples_re= re.compile(ur"(\w)\1{2,}")
trash_re= [re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "), re.compile("[-']{2,}"),
           re.compile(" '"),re.compile("  +")]
from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer()
def normalize_text(text):
    text= text.lower()
    text= nums_re.sub(" NUM ", text)
    text= " ".join([word for word in non_alphanums.sub(" ",text).split() if len(word)>1])
    return text

class WordhashRegressor(object):
    def __init__(self, pickle_model="", datadir=None):
        self.wordbatch = wordbatch.WordBatch(normalize_text,
                                             extractors=[("wordhash", {"decode_error":'ignore', "n_features":2 ** 25,
                                             "non_negative":False, "ngram_range":(1,2), "norm":'l2'})],
                                             procs=8)
        self.clf= Ridge(alpha=1.0, random_state=0)
        if datadir==None:  (self.wordbatch, self.clf)= pkl.load(gzip.open(pickle_model, u'rb'))
        else: self.train(datadir, pickle_model)

    def train(self, datadir, pickle_model=""):
        texts = []
        labels = []
        training_data = os.listdir(datadir)
        rcount = 0
        texts2 = []
        batchsize = 100000

        for jsonfile in training_data:
            with open(datadir + "/" + jsonfile, u'r') as inputfile:
                for line in inputfile:
                    # if rcount > 1000000: break
                    try:
                        line = json.loads(line.strip())
                    except:
                        continue
                    for review in line["Reviews"]:
                        rcount += 1
                        if rcount % 10000 == 0:  print rcount
                        if rcount % 9 != 0: continue
                        if "Overall" not in review["Ratings"]: continue
                        texts.append(review["Content"])
                        labels.append((float(review["Ratings"]["Overall"]) - 3) * 0.5)
                        if len(texts) % batchsize == 0:
                            texts2.append(self.wordbatch.transform(texts))
                            del (texts)
                            texts = []
        texts2.append(self.wordbatch.transform(texts))
        del (texts)
        if len(texts2) == 1:
            texts = texts2[0]
        else:
            texts = ssp.vstack(texts2)

        self.wordbatch.dictionary_freeze = True

        self.clf.fit(texts, labels)
        if pickle_model != "":
            with gzip.open(pickle_model, u'wb') as model_file:
                pkl.dump((self.wordbatch, self.clf), model_file, protocol=2)

    def predict(self, texts):
        counts= self.wordbatch.transform(texts)
        return self.clf.predict(counts)