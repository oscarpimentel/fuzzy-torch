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

