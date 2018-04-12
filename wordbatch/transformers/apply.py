#!python
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

def batch_apply(args):
    f= args[1]
    f_args= args[2]
    f_kwargs= args[3]
    return [f(row, *f_args, **f_kwargs) for row in args[0]]

class Apply(object):
    def __init__(self, batcher, function, args=[], kwargs={}):
        self.batcher= batcher
        self.function= function
        self.args= [args]
        self.kwargs= [kwargs]

    def fit(self, data, input_split= False):
        pass

    def fit_transform(self, data, input_split= False, merge_output= True):
        self.transform(data, input_split, merge_output)

    def transform(self, data, input_split= False, merge_output= True):
        return self.batcher.parallelize_batches(batch_apply, data, [self.function]+self.args+self.kwargs,
                                              input_split=input_split, merge_output=merge_output)
# import wordbatch.batcher as batcher
# b= batcher.Batcher(method="serial")
# t= [[1, 2], [3, 4]]
# import numpy as np
# a= Apply(b, np.power, [2],{})
# print(a.transform(t))