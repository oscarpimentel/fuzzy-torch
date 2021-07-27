from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn

###################################################################################################################################################

def softclamp_lrelu(x, a, b,
	negative_slope=0.001,
	):
	assert a<b
	#z = torch.clamp(x, a, s)
	z = F.leaky_relu(x-a, negative_slope=negative_slope)+a
	z = -(F.leaky_relu(-z+b, negative_slope=negative_slope)-b)
	return z

def softclamp_elu(x, a, b,
	alpha=0.1,
	):
	assert a<b
	#z = torch.clamp(x, a, s)
	z = F.elu(x-a)+a
	z = -(F.elu(-z+b)-b)
	return z

def softclamp(x, a, b):
	return softclamp_lrelu(x, a, b)

def cyclic_mod(x, a, b):
	assert b>a
	return (x-a)%(b-a)+a
