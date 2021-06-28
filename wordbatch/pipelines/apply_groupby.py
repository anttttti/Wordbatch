#!python
import pandas as pd
from wordbatch.pipelines import Apply
import wordbatch.batcher

def decorator_apply_groupby(func, group, batcher=None, rows_per_bin=200, cache=None, vectorize=None):
	def wrapper_func(*args, **kwargs):
		return ApplyGroupBy(func, args=args[1:], kwargs=kwargs, group=group, rows_per_bin=rows_per_bin,
							batcher=batcher, cache=cache, vectorize=vectorize).transform(args[0])
	return wrapper_func

class ApplyGroupBy(object):
	def __init__(self, function, group, batcher=None, rows_per_bin= 200, cache=None, vectorize=None, args=[],
				 kwargs={}):
		if batcher is None:   self.batcher= wordbatch.batcher.Batcher()
		else:  self.batcher= batcher
		self.function= function
		self.group= group
		self.rows_per_bin = rows_per_bin
		self.cache= cache
		self.vectorize= vectorize
		self.args= [args]
		self.kwargs= [kwargs]

	def fit(self, data, input_split= False):
		return self

	def fit_transform(self, data, input_split= False, merge_output= True):
		return self.transform(data, input_split, merge_output)

	def transform(self, data, input_split= False, merge_output= True):
		bin_ids = data[self.group].unique()
		group_bins= {x:1 for x in bin_ids} if len(bin_ids) <= self.rows_per_bin else \
			{x[0]: x[1] for x in zip(bin_ids, pd.qcut(bin_ids, len(bin_ids) // self.rows_per_bin))}
		group_bin_col = data[self.group].map(group_bins)
		bin_ids, groups = zip(*data.groupby(group_bin_col, as_index=False))
		t= [x for x in Apply(self.function, self.batcher, *self.args, *self.kwargs, self.cache,
							 self.vectorize).transform(groups, input_split, merge_output)
			if len(x) > 0]
		try:
			t= pd.concat(t, sort=False) # t is Series or DataFrame
		except:
			t= [item for sublist in t for item in sublist] # t is some iterable
		return t