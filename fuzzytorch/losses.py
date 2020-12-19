from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn.functional as F
from flamingchoripan.strings import xstr
from . import exceptions as ex
import numpy as np
import pandas as pd

###################################################################################################################################################

def batch_xentropy_manual(y_pred, y_target,
	class_weight=None,
	):
	assert y_pred.size()==y_target.size()
	batch_loss = -torch.sum(y_target.float() * torch.log(y_pred+EPS), dim=-1) # (b,...,c) > (b,...)
	return batch_loss # (b,...)

def batch_xentropy(y_pred, y_target,
	model_output_is_with_softmax:bool=False,
	target_is_onehot:bool=False,
	class_weight=None,
	):
	# F.cross_entropy already uses softmax as preprocessing internally
	# F.cross_entropy uses target as labels, not onehot
	classes = y_pred.size()[-1]
	no_classes_shape = y_pred.size()[:-1]
	if target_is_onehot: # [[01],[10],[01],[01],[10]]
		if model_output_is_with_softmax:
			batch_loss = batch_xentropy_manual(y_pred, y_target) # (b,...,c) > (b,...) # ugly case
		else:
			assert y_pred.shape==y_target.shape
			y_pred = y_pred.view(-1, classes)
			y_target = y_target.view(-1, classes).argmax(dim=-1)
			batch_loss = F.cross_entropy(y_pred, y_target, reduction='none') # (b,...,c) > (b)
			batch_loss = batch_loss.view(*no_classes_shape)

	else: # [0,1,3,4,2,0,1,1]
		if model_output_is_with_softmax:
			raise Exception('not implemented')
		else:
			assert y_pred.shape[0]==y_target.shape[0]
			assert len(y_pred.shape)==2
			assert len(y_target.shape)==1
			batch_loss = F.cross_entropy(y_pred.view(-1, classes), y_target.view(-1), reduction='none', weight=class_weight) # (b,...,c) > (b)
			batch_loss = batch_loss.view(*no_classes_shape)

	return batch_loss # (b,...)

###################################################################################################################################################

class LossResult():
	def __init__(self, batch_loss_,
		reduction_mode='mean',
		):
		assert len(batch_loss_.shape)==1
		self.batch_loss_ = batch_loss_
		self.len_ = len(self.batch_loss_)
		self.reduction_mode = reduction_mode
		self.batch_sublosses = {}
		if self.reduction_mode=='mean':
			self.batch_loss = self.batch_loss_.mean()[None]

	def add_subloss(self, name, batch_subloss):
		assert len(batch_subloss.shape)==1
		self.batch_sublosses[name] = batch_subloss.mean()[None]

	def get_loss(self,
		numpy=True,
		):
		numpy_loss = self.batch_loss.item()
		if np.any(np.isnan(numpy_loss)) or np.any(numpy_loss==np.infty) or np.any(numpy_loss==-np.infty):
			raise ex.NanLossException()
		return numpy_loss if numpy else self.batch_loss

	def get_subloss(self, name,
		numpy=True,
		):
		return self.batch_sublosses[name].item() if numpy else self.batch_sublosses[name]

	def get_sublosses_names(self):
		return list(self.batch_sublosses.keys())

	def __len__(self):
		return self.len_

	def __repr__(self):
		lv = f'{xstr(self.get_loss())}'
		batch_sublosses = list(self.batch_sublosses.keys())
		if len(batch_sublosses)==0: 
			return f'{lv}'
		else:
			txt = '|'.join([f'{batch_subloss}={xstr(self.get_subloss(batch_subloss))}' for batch_subloss in batch_sublosses])
			return f'{lv} ({txt})'

	def __add__(self, other):
		if other==0 or other is None:
			return self
		elif self==0 or self is None:
			return other
		else:
			loss = LossResult(self.batch_loss+other.batch_loss)
			for sl_name in self.get_sublosses_names():
				loss.add_subloss(sl_name, self.batch_sublosses[sl_name]+other.batch_sublosses[sl_name])
			return loss

	def __radd__(self, other):
		return self+other

	def __truediv__(self, other):
		self.batch_loss = self.batch_loss/other
		for sl_name in self.get_sublosses_names():
			self.batch_sublosses[sl_name] = self.batch_sublosses[sl_name]/other
		return self

	def get_info_df(self):
		sublosses_names = self.get_sublosses_names()
		values = [len(self), self.get_loss()]+[self.get_subloss(name) for name in sublosses_names]
		df = pd.DataFrame([values], columns=['__len__', '__loss__']+sublosses_names)
		return df

###################################################################################################################################################

class FTLoss(): # used for heritage
	def __init__(self, name, **kwargs):
		self.name = name
		for key in kwargs.keys():
			setattr(self, key, kwargs[key])

class XEntropy(FTLoss):
	def __init__(self, name,
			model_output_is_with_softmax:bool=False,
			target_is_onehot:bool=False,
			**kwargs):
		self.name = name
		self.model_output_is_with_softmax = model_output_is_with_softmax
		self.target_is_onehot = target_is_onehot

	def __call__(self, tdict, **kwargs):
		y_pred = tdict['model']['y']
		y_target = tdict['target']['y']
		
		batch_loss = batch_xentropy(y_pred, y_target, self.model_output_is_with_softmax, self.target_is_onehot) # (b,c) > (b)
		loss_res = LossResult(batch_loss)
		loss_res.add_subloss('loss*2', batch_loss*2)
		loss_res.add_subloss('loss*3', batch_loss*3)
		return loss_res