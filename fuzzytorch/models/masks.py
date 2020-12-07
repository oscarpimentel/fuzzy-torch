from __future__ import print_function
from __future__ import division
from . import C_

import torch
import numpy as np

###################################################################################################################################################

def get_last_element(x, onehot):
	'''
	x = (b,t,f)
	onehot = (b,t)
	'''
	assert len(x.shape)==3
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape

	b,t,f = x.size()
	indexs = torch.sum(onehot[...,None], dim=1)-1 # (b,t,1) > (b,1) # -1 because index is always 1 unit less than length
	indexs = torch.clamp(indexs, 0, None) # forced -1 -> 0 to avoid errors of empty sequences!!
	last_x = torch.gather(x, 1, indexs[:,:,None].expand(-1,-1,f)) # index (b,t,f) > (b,1,f)
	last_x = last_x[:,0,:]
	return last_x

def get_max_element(x, onehot):
	'''
	x = (b,t,f)
	onehot = (b,t)
	'''
	assert len(x.shape)==3
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape

	b,t,f = x.size()
	new_onehot = onehot.clone()
	new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
	x = x.masked_fill(~new_onehot[...,None], -1/C_.EPS) # clean using onehot
	x,_ = torch.max(x, dim=1)
	return x

###################################################################################################################################################

def generate_tensor_mask(start_index, curve_lengths, max_curve_lengths, device):
	return generate_tensor_mask_start_index_len(start_index, curve_lengths, max_curve_lengths, device)
	#return generate_tensor_mask_start_index_len(curve_lengths, max_curve_lengths, device)

def generate_tensor_mask_length(curve_lengths, max_curve_lengths,
	device=None,
	):
	'''
	5: 1 1 1 1 1 0 0 0...
	'''
	batch_size = len(curve_lengths)
	mask = torch.arange(max_curve_lengths).expand(batch_size, max_curve_lengths).to(curve_lengths.device if device is None else device)
	mask = (mask < curve_lengths[...,None])
	return mask.bool()


def generate_tensor_mask_start_index_length(start_index, curve_lengths, max_curve_lengths, device):
	'''
	2,5: 0 0 1 1 1 0 0 0...
	'''
	batch_size = len(curve_lengths)
	mask = torch.arange(max_curve_lengths).expand(batch_size, max_curve_lengths).to(device)
	mask = (mask < curve_lengths[...,None])&(mask > (start_index-1)[...,None])
	return mask.bool()

def generate_tensor_mask_laststep(curve_lengths, max_curve_lengths, device):
	'''
	5: 0 0 0 0 1 0 0 0...
	'''
	batch_size = len(curve_lengths)
	mask = torch.arange(max_curve_lengths).expand(batch_size, max_curve_lengths).to(device)
	mask = (mask == (curve_lengths-1)[...,None])
	return mask.bool()