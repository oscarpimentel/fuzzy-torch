from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn

###################################################################################################################################################

def get_cnn_output_dims(w:int, kernel_size:int, padding:int, stride:int,
	pool_kernel_size:int=1,
	dilatation:int=1,
	cnn_stacks:int=1,
	):
	out = w
	for k in range(cnn_stacks): 
		out = ((out+2*padding-dilatation*(kernel_size-1)-1) / stride) + 1
	return  int(out/pool_kernel_size)

def get_padding(padding_mode:str, kernel_size:int):
	if padding_mode=='same':
		return int(kernel_size/2)
	else:
		raise Exception(f'not supported padding_mode: {padding_mode}')
