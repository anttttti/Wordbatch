import pickle as pkl
import gzip
import re
import os
import json
import scipy as sp
import numpy as np
from neon.backends import gen_backend
from neon.data import ArrayIterator
from neon.initializers import *
from neon.layers import *
from neon.models import Model
from neon.optimizers import *
from neon.transforms import *
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.callbacks.callbacks import Callbacks
import wordbatch
import random
from threading import Thread

non_alphas = re.compile('[^A-Za-z\'-]+')
trash_re= [re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "),
           re.compile("[-']{2,}"),re.compile(" '"),re.compile("  +")]

def normalize_text(text):
    text= text.lower()
    for x in trash_re:
        while x.search(text) != None:  text = x.sub(" ", text)
    return non_alphas.sub(' ', text).strip()

class WordseqRegressor():
    def __init__(self, pickle_model="", datadir=None):
        self.maxlen = 100
        self.n_words = 100000
        parser = NeonArgparser(__doc__)
        self.args = parser.parse_args()
        self.args.batch_size = self.batch_size = 2048 #
        self.args.deterministic= None
        self.args.rng_seed= 0
        print extract_valid_args(self.args, gen_backend)
        self.be = gen_backend(**extract_valid_args(self.args, gen_backend))

        embedding_dim = 100
        init_emb = Uniform(-0.1 / embedding_dim, 0.1 / embedding_dim)
        init_glorot = GlorotUniform()
        self.layers = [
            LookupTable(vocab_size=self.n_words, embedding_dim=embedding_dim, init=init_emb, pad_idx=0, update=True,
                        name="LookupTable"),
            Dropout(keep=0.5),
            BiLSTM(100, init=init_glorot, activation=Tanh(), gate_activation=Logistic(), reset_cells=True,
                    split_inputs=False, name="BiLSTM"),
            RecurrentMean(),
            Affine(1, init_glorot, bias=init_glorot, activation=Identity(), name="Affine")
        ]

        self.wordbatch= wordbatch.WordBatch(normalize_text, n_words=self.n_words,
                                             extractors=[(wordbatch.WordSeq, {"seq_maxlen": self.maxlen})])

        if datadir == None:
            self.model= Model(self.layers)
            self.model.load_params(pickle_model)
            self.wordbatch= pkl.load(gzip.open(pickle_model + ".wb", 'rb'))
        else: self.train(datadir, pickle_model)

    def remove_unks(self, x):
        return [[self.n_words if w >= self.n_words else w for w in sen] for sen in x]

    def format_texts(self, texts):  return self.remove_unks(self.wordbatch.transform(texts))

    class ThreadWithReturnValue(Thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
            Thread.__init__(self, group, target, name, args, kwargs, Verbose)
            self._return = None

        def run(self):
            if self._Thread__target is not None:
                self._return = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)

        def join(self):
            Thread.join(self)
            return self._return

    def train(self, datadir, pickle_model=""):
        texts= []
        labels= []
        training_data = os.listdir(datadir)
        rcount= 0
        texts2= []
        batchsize= 100000

        t= None
        for jsonfile in training_data:
            with open(datadir + "/" + jsonfile, u'r') as inputfile:
                for line in inputfile:
                    #if rcount > 1000000: break
                    try: line= json.loads(line.strip())
                    except:  continue
                    for review in line["Reviews"]:
                        rcount+= 1
                        if rcount % 100000 == 0:  print rcount
                        if rcount % 8 != 0: continue
                        if "Overall" not in review["Ratings"]: continue
                        texts.append(review["Content"])
                        labels.append((float(review["Ratings"]["Overall"]) - 3) *0.5)
                        if len(texts) % batchsize == 0:
                            if t != None:  texts2.append(t.join())
                            t= self.ThreadWithReturnValue(target= self.wordbatch.transform, args= (texts,))
                            t.start()
                            texts= []
        texts2.append(t.join())
        texts2.append(self.wordbatch.transform(texts))
        del(texts)
        texts= sp.vstack(texts2)

        self.wordbatch.dictionary_freeze= True

        train = [np.asarray(texts, dtype='int32'), np.asanyarray(labels, dtype='float32')]
        train[1].shape = (train[1].shape[0], 1)

        num_epochs= 10
        cost= GeneralizedCost(costfunc=SumSquared())
        self.model= Model(layers=self.layers)
        optimizer= Adam(learning_rate=0.01)

        index_shuf= list(range(len(train[0])))
        random.shuffle(index_shuf)
        train[0]= np.asarray([train[0][x] for x in index_shuf], dtype='int32')
        train[1]= np.asarray([train[1][x] for x in index_shuf], dtype='float32')
        train_iter = ArrayIterator(train[0], train[1], nclass=1, make_onehot=False)
        self.model.fit(train_iter, optimizer=optimizer, num_epochs=num_epochs, cost=cost,
                       callbacks=Callbacks(self.model, **self.args.callback_args))

        if pickle_model != "":
            self.model.save_params(pickle_model)
            with gzip.open(pickle_model + ".wb", 'wb') as model_file:  pkl.dump(self.wordbatch, model_file, protocol=2)

    def predict_batch(self, texts):
        input= np.array(self.format_texts(texts))
        output= np.zeros((texts.shape[0], 1))
        test= ArrayIterator(input, output, nclass=1, make_onehot=False)
        results= [row[0] for row in self.model.get_outputs(test)]
        return results
