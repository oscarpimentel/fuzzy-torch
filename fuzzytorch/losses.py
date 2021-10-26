from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn.functional as F
from fuzzytools.strings import xstr
from . import exceptions as ex
import numpy as np
import pandas as pd

###################################################################################################################################################

def _check_batch(batch_loss):
	assert len(batch_loss.shape)==1 # (b)
	assert len(batch_loss)>0

def _check(batch_loss, batch_weights):
	_check_batch(batch_loss)
	assert batch_weights is None or batch_loss.shape==batch_weights.shape

###################################################################################################################################################

class BatchLoss():
	def __init__(self, batch_loss, batch_weights):
		_check(batch_loss, batch_weights)
		self.batch_loss = batch_loss # (b)
		self.batch_weights = batch_weights # (b)
		self.reset()

	def reset(self):
		self.batch_sublosses = {}

	def __len__(self):
		return len(self.batch_loss)

	def add_subloss(self, batch_subloss_name, batch_subloss):
		_check_batch(batch_subloss)
		self.batch_sublosses[batch_subloss_name] = batch_subloss # (b)

	def get_loss_item(self,
		get_tensor=False,
		):
		batch_weights = 1/len(self) if self.batch_weights is None else self.batch_weights
		loss_item = torch.sum(self.batch_loss*batch_weights) # (b)>()
		if torch.any(torch.isnan(loss_item)) or torch.any(~torch.isfinite(loss_item)):
			raise ex.NanLossError()
		if get_tensor:
			return loss_item # ()
		else:
			return loss_item.detach().item() # ()

	def backward(self):
		loss_item = self.get_loss_item(get_tensor=True)
		loss_item.backward() # gradient calculation

	def get_subloss_item(self, batch_subloss_name,
		get_tensor=False,
		):
		batch_weights = 1/len(self) if self.batch_weights is None else self.batch_weights
		loss_item = torch.sum(self.batch_sublosses[batch_subloss_name]*batch_weights) # (b)>()
		if get_tensor:
			return loss_item # ()
		else:
			return loss_item.detach().item() # ()

	def get_sublosses_names(self):
		return list(self.batch_sublosses.keys())

	def __repr__(self):
		lv = f'{xstr(self.get_loss_item())}'
		batch_sublosses = list(self.batch_sublosses.keys())
		if len(batch_sublosses)==0: 
			return f'{lv}'
		else:
			txt = '|'.join([f'{batch_subloss}={xstr(self.get_subloss_item(batch_subloss))}' for batch_subloss in batch_sublosses])
			return f'{lv} ({txt})'

	def __add__(self, other):
		if self is None or self==0:
			return other

		if other is None or other==0:
			return self

		if type(self)==BatchLoss and type(other)==BatchLoss:
			new_batch_loss = torch.cat([self.batch_loss, other.batch_loss], dim=0) # (b1+b2)
			new_batch_weights = None if (self.batch_weights is None or other.batch_weights is None) else torch.cat([self.batch_weights, other.batch_weights], dim=0) # (b1+b2)
			new_loss = BatchLoss(new_batch_loss, new_batch_weights)
			for subloss_name in self.get_sublosses_names():
				new_batch_subloss = torch.cat([self.batch_sublosses[subloss_name], other.batch_sublosses[subloss_name]], dim=0) # (b1+b2)
				new_loss.add_subloss(subloss_name, new_batch_subloss)
			return new_loss

		assert 0
		
	def __radd__(self, other):
		return self+other

	def get_info(self):
		d = {
			'_len':len(self),
			'_loss':self.get_loss_item(),
			}
		for subloss_name in self.get_sublosses_names():
			d[subloss_name] = self.get_subloss_item(subloss_name)
		return d

###################################################################################################################################################

class FTLoss():
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

	# def compute_loss(self):

	def __call__(self, tdict,
		**kwargs):
		batch_weights = self._get_weights(tdict, **kwargs)
		loss_dict = self.compute_loss(tdict, **kwargs)
		if isinstance(loss_dict, dict):
			assert '_loss' in loss_dict.keys()
			_loss = loss_dict['_loss']
			loss_obj = BatchLoss(_loss, batch_weights)
			for key in loss_dict.keys():
				sub_loss = loss_dict[key] # (b)
				if key=='_loss':
					continue
				loss_obj.add_subloss(key, sub_loss)
			return loss_obj

		elif isinstance(loss_dict, torch.Tensor):
			_loss = loss_dict # (b)
			loss_obj = BatchLoss(_loss, batch_weights)
			return loss_obj

		else:
			raise Exception(f'invalid type')

###################################################################################################################################################

# class XEntropy(FTLoss):
# 	def __init__(self, name,
# 			model_output_is_with_softmax:bool=False,
# 			target_is_onehot:bool=False,
# 			**kwargs):
# 		self.name = name
# 		self.model_output_is_with_softmax = model_output_is_with_softmax
# 		self.target_is_onehot = target_is_onehot

# 	def __call__(self, tdict, **kwargs):
# 		epoch = kwargs['_epoch']
# 		y_pred = tdict['model']['y']
# 		y_target = tdict['target']['y']
		
# 		batch_loss = batch_xentropy(y_pred, y_target, self.model_output_is_with_softmax, self.target_is_onehot) # (b,c) > (b)
# 		loss_res = LossResult(batch_loss)
# 		loss_res.add_subloss('loss*2', batch_loss*2)
# 		loss_res.add_subloss('loss*3', batch_loss*3)
# 		return loss_res