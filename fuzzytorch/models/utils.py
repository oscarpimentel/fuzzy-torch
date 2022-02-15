from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn

USES_DETACH = False # False True (optional)

###################################################################################################################################################

def get_nof_parameters(module):
	nof_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
	return nof_parameters

def get_onehot_clone(onehot,
	uses_detach=USES_DETACH,
	):
	assert onehot.dtype==torch.bool
	assert len(onehot.shape)==2
	
	if uses_detach:
		return onehot.clone().detach()
	else:
		return onehot.clone()