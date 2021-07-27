from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn

###################################################################################################################################################

def get_correct_cnn_kwargs(len_input_space_shape, cnn_kwargs):
	new_cnn_kwargs = {}
	for key in cnn_kwargs.keys():
		x = cnn_kwargs[key]
		if isinstance(x, int):
			x = [x]*len_input_space_shape
		assert len(x)==len_input_space_shape
		new_cnn_kwargs[key] = x
	return new_cnn_kwargs

def get_spatial_field(cnn_kwargs, k):
	out = (cnn_kwargs['kernel_size'][k]-1)*(cnn_kwargs['dilation'][k]-1)+cnn_kwargs['kernel_size'][k]
	return int(out)

def get_pad_value(padding_mode, is_cnn, cnn_kwargs, k):
	if padding_mode==None:
		return [0,0]
	elif padding_mode=='same':
		p = get_spatial_field(cnn_kwargs, k)//2
		return [p,p] if is_cnn else [0,0]
	elif padding_mode=='causal':
		#print(cnn_kwargs['stride'])
		assert cnn_kwargs['stride'][k]==1
		p = get_spatial_field(cnn_kwargs, k)
		return [p-1,0]
	else:
		raise Exception(f'not supported padding_mode: {padding_mode}')

def get_output_space(input_space, cnn_kwargs, cnn_padding, k):
	input_space_k = input_space[k]
	if input_space_k is None:
		return None
	out = ((input_space_k+2*cnn_padding[k]//2-cnn_kwargs['dilation'][k]*(cnn_kwargs['kernel_size'][k]-1)-1) / cnn_kwargs['stride'][k]) + 1
	return int(out)




