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

# def batch_xentropy_manual(y_pred, y_target,
# 	class_weight=None,
# 	):
# 	assert y_pred.size()==y_target.size()
# 	batch_loss = -torch.sum(y_target.float() * torch.log(y_pred+EPS), dim=-1) # (b,...,c) > (b,...)
# 	return batch_loss # (b,...)

# def batch_xentropy(y_pred, y_target,
# 	model_output_is_with_softmax:bool=False,
# 	target_is_onehot:bool=False,
# 	class_weight=None,
# 	):
# 	# F.cross_entropy already uses softmax as preprocessing internally
# 	# F.cross_entropy uses target as labels, not onehot
# 	classes = y_pred.size()[-1]
# 	no_classes_shape = y_pred.size()[:-1]
# 	if target_is_onehot: # [[01],[10],[01],[01],[10]]
# 		if model_output_is_with_softmax:
# 			batch_loss = batch_xentropy_manual(y_pred, y_target) # (b,...,c) > (b,...) # ugly case
# 		else:
# 			assert y_pred.shape==y_target.shape
# 			y_pred = y_pred.view(-1, classes)
# 			y_target = y_target.view(-1, classes).argmax(dim=-1)
# 			batch_loss = F.cross_entropy(y_pred, y_target, reduction='none') # (b,...,c) > (b)
# 			batch_loss = batch_loss.view(*no_classes_shape)

# 	else: # [0,1,3,4,2,0,1,1]
# 		if model_output_is_with_softmax:
# 			raise Exception('not implemented')
# 		else:
# 			assert y_pred.shape[0]==y_target.shape[0]
# 			assert len(y_pred.shape)==2
# 			assert len(y_target.shape)==1
# 			batch_loss = F.cross_entropy(y_pred.view(-1, classes), y_target.view(-1), reduction='none', weight=class_weight) # (b,...,c) > (b)
# 			batch_loss = batch_loss.view(*no_classes_shape)

# 	return batch_loss # (b,...)

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
		if other==0 or other is None:
			return self
		elif self==0 or self is None:
			return other
		else:
			new_batch_loss = torch.cat([self.batch_loss, other.batch_loss], dim=0) # (b1+b2)
			new_batch_weights = None if (self.batch_weights is None or other.batch_weights is None) else torch.cat([self.batch_weights, other.batch_weights], dim=0) # (b1+b2)
			new_loss = BatchLoss(new_batch_loss, new_batch_weights)
			for subloss_name in self.get_sublosses_names():
				new_batch_subloss = torch.cat([self.batch_sublosses[subloss_name], other.batch_sublosses[subloss_name]], dim=0) # (b1+b2)
				new_loss.add_subloss(subloss_name, new_batch_subloss)
			return new_loss

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