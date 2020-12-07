from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn

###################################################################################################################################################

def count_parameters(module):
	return sum(p.numel() for p in module.parameters() if p.requires_grad)

def count_memory_mb(module,
	bits=32,
	):
	assert bits*count_parameters(module)

###################################################################################################################################################

def get_cnn_output_dims(w:int, kernel_size:int, padding:int, stride:int,
	cnn_stacks:int=1,
	pool_kernel_size:int=1,
	dilatation:int=1,
	):
	out = w
	for k in range(cnn_stacks): 
		out = ((out+2*padding-dilatation*(kernel_size-1)-1) / stride) + 1
	return  int(out/pool_kernel_size)

def get_padding(padding_mode:str, kernel_size:int):
	if padding_mode=='same':
		return int(kernel_size/2)
	else:
		raise Exception(f'not supported padding_mode:{padding_mode}')

###################################################################################################################################################

class TinyModels(nn.Module):
	'''
	Class used mostly for decorators
	'''
	def _silence(fun):
		def new_forward(self, *args, **kwargs):
			with HiddenPrints():
				ret = fun(self, *args, **kwargs) # speacially useful for self.forward
			return ret
		return new_forward

