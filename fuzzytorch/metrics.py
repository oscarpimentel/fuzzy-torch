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
		self.batch_metric = batch_metric.detach() # (n)
		self.batch_weights = batch_weights # (n)
		self.reset()

	def reset(self):
		pass

	def get_metric_item(self,
		get_tensor=False,
		):
		batch_weights = 1/len(self) if self.batch_weights is None else self.batch_weights # (n)
		metric_item = torch.sum(self.batch_metric*batch_weights) # (n)>()
		if get_tensor:
			return metric_item # ()
		else:
			return metric_item.detach().item() # ()

	def __len__(self):
		return len(self.batch_metric)

	def __repr__(self):
		return f'{xstr(self.get_metric_item())}'

	def __add__(self, other):
		# concatenate
		if self is None or self==0:
			return other

		elif other is None or other==0:
			return self

		elif type(self)==BatchMetric and type(other)==BatchMetric:
			new_batch_metric = torch.cat([self.batch_metric, other.batch_metric], dim=0) # (n1+n2)
			new_batch_weights = None if (self.batch_weights is None or other.batch_weights is None) else torch.cat([self.batch_weights, other.batch_weights], dim=0) # (n1+n2)
			new_metric = BatchMetric(new_batch_metric, new_batch_weights)
			return new_metric
		
		else:
			raise Exception(f'{type(self)}; {type(other)}')

	def __radd__(self, other):
		return self+other

	def get_info(self):
		d = {
			'_len':len(self),
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
			batch_weights =  tdict[self.weight_key] # (n)
			# print(f'batch_weights={batch_weights}')
			return batch_weights

	# def compute_metric(self):

	def __call__(self, tdict,
		**kwargs):
		batch_weights = self._get_weights(tdict, **kwargs)
		_metric = self.compute_metric(tdict, **kwargs) # (n)
		metric_obj = BatchMetric(_metric, batch_weights)
		return metric_obj

###################################################################################################################################################

class LossWrapper(FTMetric):
	def __init__(self, loss_obj):
		super().__init__(loss_obj.name, loss_obj.weight_key)
		self.loss_obj = loss_obj

	def compute_metric(self, tdict,
		**kwargs):
		loss_dict = self.loss_obj.compute_loss(tdict, **kwargs) # (n)
		if type(loss_dict)==dict:
			_loss = loss_dict['_loss'] # (n)
			return _loss

		elif type(loss_dict)==torch.Tensor:
			_loss = loss_dict # (n)
			return _loss

		else:
			raise Exception(f'type={type(loss_dict)}')