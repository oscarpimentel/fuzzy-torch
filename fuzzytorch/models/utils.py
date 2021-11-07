from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn

###################################################################################################################################################

def get_nof_parameters(module):
	return sum(p.numel() for p in module.parameters() if p.requires_grad)

