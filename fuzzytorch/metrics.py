from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn.functional as F
from fuzzytools.strings import xstr
import pandas as pd

###################################################################################################################################################

class MetricResult():
	def __init__(self, _batch_metric,
		reduction_mode='mean',
		):
		assert len(_batch_metric.shape)==1
		self._batch_metric = _batch_metric.detach()
		self.len_ = len(self._batch_metric)
		self.reduction_mode = reduction_mode
		if self.reduction_mode=='mean':
			self.batch_metric = self._batch_metric.mean()[None] # (b)
		elif self.reduction_mode=='sum':
			self.batch_metric = self._batch_metric.sum()[None] # (b)

	def to(self, device):
		pass
		
	def get_metric(self,
		get_tensor=False,
		):
		assert len(self.batch_metric.shape)==1
		assert len(self.batch_metric)==1
		if not get_tensor:
			return self.batch_metric.detach().item()
			#return self.batch_metric.data[0]
		else:
			self.batch_metric

	def __len__(self):
		return self.len_

	def __repr__(self):
		return f'{xstr(self.get_metric())}'

	def __add__(self, other):
		if other==0 or other is None:
			return self
		elif self==0 or self is None:
			return other
		else:
			assert self.reduction_mode==other.reduction_mode
			metric = MetricResult(self.batch_metric+other.batch_metric, # (b)+(b)
				self.reduction_mode,
				)
			return metric

	def __radd__(self, other):
		return self+other

	def __truediv__(self, other):
		self.batch_metric = self.batch_metric/other
		return self

	def get_info(self):
		d = {
			'_metric':self.get_metric(),
			}
		return d

###################################################################################################################################################

def get_labels_accuracy(y_pred, y_target, labels):
	labels_accuracies = []
	for k in range(labels):
		valid_idxs = torch.where(y_target==k)
		y_pred_k = y_pred[valid_idxs]
		y_target_k = y_target[valid_idxs]
		
		accuracies = (y_target_k==y_pred_k).float()*100
		if len(valid_idxs[0])>0:
			labels_accuracies.append(torch.mean(accuracies)[None])
	return torch.cat(labels_accuracies)

###################################################################################################################################################

class FTMetric(): # used for heritage
	def __init__(self, name, **kwargs):
		self.name = name
		for key in kwargs.keys():
			setattr(self, key, kwargs[key])

class DummyAccuracy(FTMetric):
	def __init__(self, name, **kwargs):
		self.name = name

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['_epoch']
		y_target = tdict['target']['y']
		y_pred = tdict['model']['y']

		m = torch.ones((len(y_pred)))/y_pred.shape[-1]*100
		return MetricResult(m)

class Accuracy(FTMetric):
	def __init__(self, name,
		target_is_onehot:bool=False,
		balanced=False,
		**kwargs):
		self.name = name
		self.target_is_onehot = target_is_onehot
		self.balanced = balanced

	def __call__(self, tdict, **kwargs):
		epoch = kwargs['_epoch']
		y_target = tdict['target']['y']
		y_pred = tdict['model']['y']
		labels = y_pred.shape[-1]

		assert y_target.dtype==torch.long

		if self.target_is_onehot:
			assert y_pred.shape==y_target.shape
			y_target = y_target.argmax(dim=-1)
		
		y_pred = y_pred.argmax(dim=-1)
		assert y_pred.shape==y_target.shape
		assert len(y_pred.shape)==1

		accuracies = get_labels_accuracy(y_pred, y_target, labels) if self.balanced else torch.mean((y_pred==y_target).float()*100)[None] # (b) > (1)
		#print(accuracies.shape)
		return MetricResult(accuracies)