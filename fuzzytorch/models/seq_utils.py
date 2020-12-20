from __future__ import print_function
from __future__ import division
from . import C_

import torch
import numpy as np

###################################################################################################################################################

def mapping(source, indexs, output,
	dim=1,
	):
	fixed_indexs = indexs.clone()
	IMD = source.shape[1]
	fixed_indexs[fixed_indexs==IMD] = output.shape[dim]-1
	fixed_indexs = fixed_indexs.unsqueeze(-1).expand(-1,-1,source.shape[-1])
	#print(output.device, fixed_indexs.device, source.device)
	output.scatter_(dim, fixed_indexs, source)

###################################################################################################################################################

def seq_clean(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	assert onehot.dtype==torch.bool
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape
	assert len(x.shape)==3

	x = x.masked_fill(~onehot[...,None], 0) # clean using onehot
	return x

def seq_mean(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	assert onehot.dtype==torch.bool
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape
	assert len(x.shape)==3

	x = seq_clean(x, onehot)
	x = x.sum(dim=1)/(onehot.sum(dim=1)[...,None]+C_.EPS) # (b,t,f) > (b,f)
	return x

def seq_last_element(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	assert onehot.dtype==torch.bool
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape
	assert len(x.shape)==3

	b,t,f = x.size()
	indexs = torch.sum(onehot[...,None], dim=1)-1 # (b,t,1) > (b,1) # -1 because index is always 1 unit less than length
	indexs = torch.clamp(indexs, 0, None) # forced -1 -> 0 to avoid errors of empty sequences!!
	last_x = torch.gather(x, 1, indexs[:,:,None].expand(-1,-1,f)) # index (b,t,f) > (b,1,f)
	last_x = last_x[:,0,:]
	return last_x

def seq_max_element(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	assert onehot.dtype==torch.bool
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape
	assert len(x.shape)==3

	b,t,f = x.size()
	new_onehot = onehot.clone()
	new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
	x = x.masked_fill(~new_onehot[...,None], -1/C_.EPS) # clean using onehot
	x,_ = torch.max(x, dim=1)
	return x

def get_seq_onehot_mask(seqlengths, max_seqlength,
	device=None,
	):
	assert len(seqlengths.shape)==1
	assert seqlengths.dtype==torch.long

	batch_size = len(seqlengths)
	mask = torch.arange(max_seqlength).expand(batch_size, max_seqlength).to(seqlengths.device if device is None else device)
	mask = (mask < seqlengths[...,None])
	return mask.bool() # (b,t,f)

def serial_to_parallel(x, onehot,
	fill_value=0,
	):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	assert onehot.dtype==torch.bool
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape
	assert len(x.shape)==3

	IMD = onehot.shape[1]
	s2p_mapping_indexs = (torch.cumsum(onehot, 1)-1).masked_fill(~onehot, IMD)
	#print('s2p_mapping_indexs', s2p_mapping_indexs.shape, s2p_mapping_indexs)
	new_shape = (x.shape[0], x.shape[1]+1, x.shape[2])
	new_x = torch.full(new_shape, fill_value, device=x.device, dtype=x.dtype)
	mapping(x, s2p_mapping_indexs, new_x)
	return new_x[:,:-1,:]