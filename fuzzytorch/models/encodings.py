from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F 
from .basics import Linear
from . import utils as utils
import flamingchoripan.strings as strings

###################################################################################################################################################

class FILM(nn.Module):
	def __init__(self, mod_input_dims:int, mod_output_dims:int,
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		**kwargs):
		super().__init__()

		### CHECKS
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1

		self.mod_input_dims = mod_input_dims
		self.mod_output_dims = mod_output_dims
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.bias = bias

		### MODULES
		linear_kwargs = {
			'activation':'linear',
			'split_out':2,
			'bias':self.bias,
		}
		self.is_dummy = self.mod_input_dims==0
		if not self.is_dummy:
			self.mod_f = Linear(self.mod_input_dims, self.mod_output_dims, **linear_kwargs)
			self.in_dropout_f = nn.Dropout(self.in_dropout)
			self.out_dropout_f = nn.Dropout(self.out_dropout)

	def forward(self, x, mod, **kwargs):
		# x (b,t,fx)
		# mod (b,t,fm)
		assert x.shape[-1]==self.mod_output_dims

		if not self.is_dummy:
			x = self.in_dropout_f(x)
			gamma, beta = self.mod_f(mod)
			x = x*gamma+beta
			x = self.out_dropout_f(x)
		return x

	def __len__(self):
		return utils.count_parameters(self)
		
	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'mod_input_dims':self.mod_input_dims,
		'mod_output_dims':self.mod_output_dims,
		'in_dropout':self.in_dropout,
		'out_dropout':self.out_dropout,
		'bias':self.bias,
		}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'FILM({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt