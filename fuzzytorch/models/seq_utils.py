from __future__ import print_function
from __future__ import division
from . import C_

import torch
import numpy as np

###################################################################################################################################################

def check_(x, onehot):
	assert onehot.dtype==torch.bool
	assert len(onehot.shape)==2
	assert x.shape[:-1]==onehot.shape
	assert len(x.shape)==3

###################################################################################################################################################

def seq_clean(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	x = x.masked_fill(~onehot[...,None], 0) # clean using onehot
	return x

def seq_avg_pooling(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	x = seq_clean(x, onehot) # important
	new_onehot = onehot.clone()
	new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
	x = x.sum(dim=1)/(new_onehot.sum(dim=1)[...,None]) # (b,t,f) > (b,f)
	return x

def seq_sum_pooling(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	x = seq_clean(x, onehot) # important
	new_onehot = onehot.clone()
	new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
	x = x.sum(dim=1) # (b,t,f) > (b,f)
	return x

def seq_last_element(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	b,t,f = x.size()
	indexs = torch.sum(onehot[...,None], dim=1)-1 # (b,t,1) > (b,1) # -1 because index is always 1 unit less than length
	indexs = torch.clamp(indexs, 0, None) # forced -1 -> 0 to avoid errors of empty sequences!!
	last_x = torch.gather(x, 1, indexs[:,:,None].expand(-1,-1,f)) # index (b,t,f) > (b,1,f)
	last_x = last_x[:,0,:]
	return last_x

def seq_min_pooling(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	b,t,f = x.size()
	new_onehot = onehot.clone()
	new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
	infty = 1/C_.EPS
	x = x.masked_fill(~new_onehot[...,None], +infty) # clean using onehot
	x,_ = torch.min(x, dim=1)
	return x

def seq_max_pooling(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	b,t,f = x.size()
	new_onehot = onehot.clone()
	new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
	infty = 1/C_.EPS
	x = x.masked_fill(~new_onehot[...,None], -infty) # clean using onehot
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
	return mask.bool() # (b,t)

###################################################################################################################################################

def seq_min_max_norm(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	min_ = seq_min_pooling(x, onehot)[:,None,:] # (b,f) > (b,1,f)
	max_ = seq_max_pooling(x, onehot)[:,None,:] # (b,f) > (b,1,f)
	#print(min_, max_)
	diff_ = max_-min_
	return (x-min_)/(diff_+C_.EPS)

def seq_avg_norm(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)
	assert torch.all(x>=0)

	avg_ = seq_avg_pooling(x, onehot)[:,None,:] # (b,f) > (b,1,f)
	return x/(avg_+C_.EPS)

def seq_sum_norm(x, onehot):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)
	assert torch.all(x>=0)

	sum_ = seq_sum_pooling(x, onehot)[:,None,:] # (b,f) > (b,1,f)
	return x/(sum_+C_.EPS)

###################################################################################################################################################

def seq_index_mapping_(source, idxs, output,
	dim=1,
	):
	assert source.dtype==output.dtype
	assert source.shape[0]==output.shape[0]
	assert source.shape[1]==output.shape[1]-1
	assert source.shape[2]==output.shape[2]
	assert len(source.shape)==3
	assert len(idxs.shape)==2
	assert idxs.shape==source.shape[:-1]

	fixed_indexs = idxs.clone()
	IMD = source.shape[1]
	fixed_indexs[fixed_indexs==IMD] = output.shape[dim]-1
	fixed_indexs = fixed_indexs.unsqueeze(-1).expand(-1,-1,source.shape[-1])
	#print(output.device, fixed_indexs.device, source.device)
	output.scatter_(dim, fixed_indexs, source)

def serial_to_parallel(x, onehot,
	fill_value=0,
	):
	'''
	x (b,t,f)
	onehot (b,t)
	'''
	check_(x, onehot)

	IMD = onehot.shape[1]
	s2p_mapping_indexs = (torch.cumsum(onehot, 1)-1).masked_fill(~onehot, IMD)
	#print('s2p_mapping_indexs', s2p_mapping_indexs.shape, s2p_mapping_indexs)
	new_shape = (x.shape[0], x.shape[1]+1, x.shape[2])
	new_x = torch.full(new_shape, fill_value, device=x.device, dtype=x.dtype)
	seq_index_mapping_(x, s2p_mapping_indexs, new_x)
	return new_x[:,:-1,:]

def parallel_to_serial(list_x, s_onehot,
	fill_value=0,
	):
	'''
	list_x list[(b,t,f)]
	onehot (b,t,d)
	'''
	assert isinstance(list_x, list)
	assert len(list_x)>0
	assert s_onehot.dtype==torch.bool
	assert len(s_onehot.shape)==3
	for x in list_x:
		assert x.shape[:-1]==s_onehot.shape[:-1]
		assert len(x.shape)==3

	modes = s_onehot.shape[-1]
	x_ = list_x[0]
	new_shape = (x_.shape[0], x_.shape[1]+1, x_.shape[2])
	x_s = torch.full(new_shape, fill_value, device=x.device, dtype=x.dtype)
	for i in range(modes):
		x = list_x[i]
		onehot = s_onehot[...,i]

		IMD = onehot.shape[1]
		s2p_mapping_indexs = (torch.cumsum(onehot, dim=1)-1).masked_fill(~onehot, IMD)
		source = torch.cumsum(torch.ones_like(s2p_mapping_indexs, device=x.device, dtype=x.dtype)[...,None], dim=1)-1
		
		p2s_mapping_indexs = torch.full((x_.shape[0], x_.shape[1]+1, 1), IMD, device=x.device, dtype=x.dtype)
		#print(source.shape)
		#print(p2s_mapping_indexs.shape)
		seq_index_mapping_(source, s2p_mapping_indexs, p2s_mapping_indexs)
		#print(p2s_mapping_indexs.shape)
		#idxs = torch.nonzero(onehot)
		#print(idxs.shape, idxs)
		#assert 0
		#print(x_s.shape, onehot.shape, x.shape)
		seq_index_mapping_(x, p2s_mapping_indexs[:,:-1,0].long(), x_s)
		#p2s_mapping_indexs = torch.where(onehot)
		#print(p2s_mapping_indexs)
		#assert 0
		#print('p2s_mapping_indexs', p2s_mapping_indexs.shape, p2s_mapping_indexs)
		#index_mapping(x, p2s_mapping_indexs, x_s)
	return x_s[:,:-1,:]

def get_random_onehot(x, modes):
	'''
	x (b,t,f)
	'''
	assert len(x.shape)==3
	assert modes>=2

	shape = list(x.shape)[:-1]+[modes]
	r = np.random.uniform(0, modes, size=shape)
	r_max = r.max(axis=-1)[...,None]
	onehot = torch.as_tensor(r>=r_max).bool()
	return onehot


def get_seq_clipped_shape(x, new_len):
	'''
	Used in dataset creation
	x (t,f)
	'''
	assert len(x.shape)==2
	if new_len is None:
		return x
	assert new_len>0

	t,f = x.size()
	if new_len<=t:
		return x[:new_len]
	else:
		new_x = torch.zeros(size=(new_len,f), device=x.device, dtype=x.dtype)
		new_x[:t] = x
		return new_x