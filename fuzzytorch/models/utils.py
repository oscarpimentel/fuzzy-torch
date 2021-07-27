from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn

###################################################################################################################################################

def count_parameters(module):
	return sum(p.numel() for p in module.parameters() if p.requires_grad)

def count_memory_mb(module,
	bits=32,
	):
	assert 0, 'not implemented'
	assert bits*count_parameters(module)

