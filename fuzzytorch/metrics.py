from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn.functional as F
from fuzzytools.strings import xstr
import pandas as pd
from . import losses as losses

###################################################################################################################################################

class BatchMetric():
	def __init__(self, batch_metric, batch_weights):
		losses._check(batch_metric, batch_weights)
		self.batch_metric = batch_metric.detach() # (b)
		self.batch_weights = batch_weights # (b)
		self.reset()

	def reset(self):
		pass

	def get_metric_item(self,
		get_tensor=False,
		):
		batch_weights = 1/len(self) if self.batch_weights is None else self.batch_weights
		metric_item = torch.sum(self.batch_metric*batch_weights) # (b)>()
		if get_tensor:
			return metric_item # ()
		else:
			return metric_item.detach().item() # ()

	def __len__(self):
		return len(self.batch_metric)

	def __repr__(self):
		return f'{xstr(self.get_metric_item())}'

	def __add__(self, other):
		if other==0 or other is None:
			return self
		elif self==0 or self is None:
			return other
		else:
			new_batch_metric = torch.cat([self.batch_metric, other.batch_metric], dim=0) # (b1+b2)
			new_batch_weights = None if (self.batch_weights is None or other.batch_weights is None) else torch.cat([self.batch_weights, other.batch_weights], dim=0) # (b1+b2)
			new_metric = BatchMetric(new_batch_metric, new_batch_weights)
			return new_metric

	def __radd__(self, other):
		return self+other

	def get_info(self):
		d = {
			'_metric':self.get_metric_item(),
			}
		return d

###################################################################################################################################################

class FTMetric():
	def __init__(self, name, weight_key,
		**kwargs):
		self.name = name
		self.weight_key = weight_key

	def _get_weights(self, tdict,
		**kwargs):
		if self.weight_key is None:
			return None
		else:
			batch_weights =  tdict[self.weight_key] # (b)
			return batch_weights

	# def compute_metric(self):

	def __call__(self, tdict,
		**kwargs):
		batch_weights = self._get_weights(tdict, **kwargs)
		_metric = self.compute_metric(tdict, **kwargs) # (b)
		metric_obj = BatchMetric(_metric, batch_weights)
		return metric_obj

###################################################################################################################################################

class LossWrapper(FTMetric):
	def __init__(self, loss_obj):
		super().__init__(loss_obj.name, loss_obj.weight_key)
		self.loss_obj = loss_obj

	def compute_metric(self, tdict,
		**kwargs):
		loss_dict = self.loss_obj.compute_loss(tdict, **kwargs) # (b)
		if isinstance(loss_dict, dict):
			_loss = loss_dict['_loss'] # (b)
			return _loss

		elif isinstance(loss_dict, torch.Tensor):
			_loss = loss_dict # (b)
			return _loss

		else:
			raise Exception(f'invalid type')

###################################################################################################################################################

# def get_labels_accuracy(y_pred, y_target, labels):
# 	labels_accuracies = []
# 	for k in range(labels):
# 		valid_idxs = torch.where(y_target==k)
# 		y_pred_k = y_pred[valid_idxs]
# 		y_target_k = y_target[valid_idxs]
		
# 		accuracies = (y_target_k==y_pred_k).float()*100
# 		if len(valid_idxs[0])>0:
# 			labels_accuracies.append(torch.mean(accuracies)[None])
# 	return torch.cat(labels_accuracies)

# ###################################################################################################################################################

# class DummyAccuracy(FTMetric):
# 	def __init__(self, name, **kwargs):
# 		self.name = name

# 	def __call__(self, tdict, **kwargs):
# 		epoch = kwargs['_epoch']
# 		y_target = tdict['target']['y']
# 		y_pred = tdict['model']['y']

# 		m = torch.ones((len(y_pred)))/y_pred.shape[-1]*100
# 		return MetricResult(m)

# class Accuracy(FTMetric):
# 	def __init__(self, name,
# 		target_is_onehot:bool=False,
# 		balanced=False,
# 		**kwargs):
# 		self.name = name
# 		self.target_is_onehot = target_is_onehot
# 		self.balanced = balanced

# 	def __call__(self, tdict, **kwargs):
# 		epoch = kwargs['_epoch']
# 		y_target = tdict['target']['y']
# 		y_pred = tdict['model']['y']
# 		labels = y_pred.shape[-1]

# 		assert y_target.dtype==torch.long

# 		if self.target_is_onehot:
# 			assert y_pred.shape==y_target.shape
# 			y_target = y_target.argmax(dim=-1)
		
# 		y_pred = y_pred.argmax(dim=-1)
# 		assert y_pred.shape==y_target.shape
# 		assert len(y_pred.shape)==1

# 		accuracies = get_labels_accuracy(y_pred, y_target, labels) if self.balanced else torch.mean((y_pred==y_target).float()*100)[None] # (b) > (1)
# 		#print(accuracies.shape)
# 		return MetricResult(accuracies)