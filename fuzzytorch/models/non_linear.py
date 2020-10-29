from __future__ import print_function
from __future__ import division
from . import C_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################################
### Functions

def f_relu(x, dim:int):
	return F.relu(x)

def f_elu(x, dim:int):
	return F.elu(x)

def f_linear(x, dim:int):
	return x

def f_tanh(x, dim:int):
	return torch.tanh(x)

def f_sigmoid(x, dim:int):
	return torch.sigmoid(x)

def get_activation(activation:str):
	if activation=='relu':
		return f_relu
	elif activation=='elu':
		return f_elu
	elif activation=='tanh':
		return f_tanh
	elif activation=='sigmoid':
		return f_sigmoid
	elif activation=='linear':
		return f_linear
	elif activation=='softmax':
		return F.softmax
	else:
		raise Exception(f'the activation function {activation} doesnt exists')

def get_xavier_gain(activation:str,
	**kwargs):
	if activation=='relu':
		return math.sqrt(2)
	elif activation=='elu':
		return math.sqrt(2)
	elif activation=='leaky-relu':
		alpha = kwargs.get('alpha',1)
		return math.sqrt(2/(1+alpha**2))
	elif activation=='tanh':
		return 5/3
	elif activation=='sigmoid':
		return 1 # ???
	elif activation=='linear':
		return 1
	elif activation=='softmax':
		return 1
	else:
		raise Exception(f'the activation function {activation} doesnt exists')