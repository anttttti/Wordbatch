from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import re
import json
import gzip
from sklearn.linear_model import *
import scipy.sparse as ssp
import wordbatch
from wordbatch.extractors import WordHash
import threading
import sys
if sys.version_info.major == 3:
    import pickle as pkl
else:
    import cPickle as pkl

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

class BatchData(object):
    def __init__(self):
        self.texts= None

class WordhashRegressor(object):
    def __init__(self, pickle_model="", datadir=None):
        self.wb= wordbatch.WordBatch(normalize_text,
                                             extractor=(WordHash, {"decode_error":'ignore', "n_features":2 ** 25,
                                             "non_negative":False, "ngram_range":(1,2), "norm":'l2'}))
        self.clf= Ridge(alpha=1.0, random_state=0)
        if datadir==None:  (self.wb, self.clf)= pkl.load(gzip.open(pickle_model, 'rb'))
        else: self.train(datadir, pickle_model)

    def transform_batch(self, texts, batch_data):
        batch_data.texts= self.wb.fit_transform(texts, reset= False)

    def train(self, datadir, pickle_model=""):
        texts= []
        labels= []
        training_data= os.listdir(datadir)
        rcount= 0
        texts2= []
        batchsize= 100000

        batch_data = BatchData()
        p_input= None
        for jsonfile in training_data:
            with open(datadir + "/" + jsonfile, 'r') as inputfile:
                for line in inputfile:
                    # if rcount > 1000000: break
                    try:  line = json.loads(line.strip())
                    except:  continue
                    for review in line["Reviews"]:
                        rcount+= 1
                        if rcount % 100000 == 0:  print(rcount)
                        if rcount % 9 != 0: continue
                        if "Overall" not in review["Ratings"]: continue
                        texts.append(review["Content"])
                        labels.append((float(review["Ratings"]["Overall"]) - 3) * 0.5)
                        if len(texts) % batchsize == 0:
                            if p_input != None:
                                p_input.join()
                                texts2.append(batch_data.texts)
                            p_input = threading.Thread(target=self.transform_batch, args=(texts, batch_data))
                            p_input.start()
                            texts= []
        if p_input != None:
            p_input.join()
            texts2.append(batch_data.texts)
            texts2.append(self.wb.fit_transform(texts, reset= False))
        del (texts)
        if len(texts2) == 1:  texts= texts2[0]
        else:  texts= ssp.vstack(texts2)

        self.wb.dictionary_freeze = True

        self.clf.fit(texts, labels)
        if pickle_model != "":
            with gzip.open(pickle_model, 'wb') as model_file:
                pkl.dump((self.wb, self.clf), model_file, protocol=2)

    def predict(self, texts):
        counts= self.wb.transform(texts)
        return self.clf.predict(counts)