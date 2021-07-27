from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

###################################################################################################################################################

TORCH_ACT_DICT = {
	'linear':'linear',
	'conv':'Conv',
	'sigmoid':'Sigmoid',
	'tanh':'Tanh',
	'relu':'ReLU',
	'RelU':'Leaky Relu',
	'selu':'SELU',
}

def f_linear(x, dim:int):
	return x

def f_sigmoid(x, dim:int):
	return torch.sigmoid(x)

def f_tanh(x, dim:int):
	return torch.tanh(x)

def f_relu(x, dim:int):
	return F.relu(x)

def f_elu(x, dim:int):
	return F.elu(x)

def get_activation(activation:str):
	if activation=='linear':
		return f_linear
	if activation=='sigmoid':
		return f_sigmoid
	if activation=='tanh':
		return f_tanh
	if activation=='relu':
		return f_relu
	if activation=='elu':
		return f_elu
	if activation=='softmax':
		return F.softmax
	raise Exception(f'no valid activation={activation}')

def get_xavier_gain(activation_name, param=None):
	#assert activation in TORCH_ACT_DICT.keys()
	#torch_activation = TORCH_ACT_DICT[activation]
	gain = torch.nn.init.calculate_gain(activation_name, param=param)
	return gain