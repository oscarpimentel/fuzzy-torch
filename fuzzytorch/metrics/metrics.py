from __future__ import print_function
from __future__ import division
from . import C_

import torch

class NewMetricCrit:
	def __init__(self, name, fun, kwargs={}):
		self.name = name
		self.fun = fun
		self.kwargs = kwargs

############# METRICS ZOO

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

def dummy_accuracy(y_pred, y_target,
	**kwargs):
	return torch.ones((1))*0.666