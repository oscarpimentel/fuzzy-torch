from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn.functional as F
from .datasets import TensorDict
from flamingchoripan.strings import xstr
import pandas as pd

###################################################################################################################################################

class MetricResult():
	def __init__(self, batch_metric,
		reduction_mode='mean',
		):
		assert len(batch_metric.shape)==1
		self.original_len = len(batch_metric)
		self.reduction_mode = reduction_mode
		if self.reduction_mode=='mean':
			self.batch_metric = batch_metric.mean()[None]

	def get_metric(self,
		numpy=True,
		):
		return self.batch_metric.item() if numpy else self.batch_metric

	def __len__(self):
		return self.original_len

	def __repr__(self):
		return f'{xstr(self.get_metric())}'

	def __add__(self, other):
		if other==0 or other is None:
			return self
		elif self==0 or self is None:
			return other
		else:
			m = MetricResult(self.batch_metric+other.batch_metric)
			return m

	def __radd__(self, other):
		return self+other

	def __truediv__(self, other):
		self.batch_metric = self.batch_metric/other
		return self

	def get_info_df(self):
		df = pd.DataFrame([self.get_metric()], columns=['__metric__'])
		return df

###################################################################################################################################################

class FTMetric():
	def __init__(self, name, **kwargs):
		self.name = name
		for key in kwargs.keys():
			setattr(self, key, kwargs[key])

class DummyAccuracy(FTMetric):
	def __init__(self, name,
		**kwargs):
		self.name = name

	def __call__(self, tensor_dict):
		assert isinstance(tensor_dict, TensorDict)
		y_pred = tensor_dict['output']['y']
		y_target = tensor_dict['target']['y']
		m = torch.ones((len(y_pred)))/y_pred.shape[-1]*100
		return MetricResult(m)

class OnehotAccuracy(FTMetric):
	def __init__(self, name,
		target_is_onehot:bool=True,
		**kwargs):
		self.name = name
		self.target_is_onehot = target_is_onehot

	def __call__(self, tensor_dict):
		assert isinstance(tensor_dict, TensorDict)
		y_pred = tensor_dict['output']['y']
		y_target = tensor_dict['target']['y']
		
		if self.target_is_onehot:
			assert y_pred.size==y_target.size
			y_target = y_target.argmax(dim=-1)
		
		y_pred = y_pred.argmax(dim=-1)
		assert y_pred.shape==y_target.shape
		accuracies = (y_pred==y_target).float()
		return MetricResult(accuracies*100)