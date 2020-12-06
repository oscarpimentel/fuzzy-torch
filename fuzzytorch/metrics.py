from __future__ import print_function
from __future__ import division
from . import C_

import torch

###################################################################################################################################################

def onehot_accuracy(y_pred, y_target,
	target_is_onehot:bool=True,
	pred_dict_key:str=None,
	target_dict_key:str=None,
	**kwargs):
	y_pred = (y_pred[pred_dict_key] if not pred_dict_key is None else y_pred)
	y_target = (y_target[target_dict_key] if not target_dict_key is None else y_target)
	
	if target_is_onehot:
		assert y_pred.size==y_target.size
		y_target = y_target.argmax(dim=-1)
	
	y_pred = y_pred.argmax(dim=-1)
	assert y_pred.shape==y_target.shape
	accuracies = (y_pred==y_target).float()
	accuracy = torch.mean(accuracies)
	return accuracy

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

	def __call__(self):
		return torch.ones((1))*0.666