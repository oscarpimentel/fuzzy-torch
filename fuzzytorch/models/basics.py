from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import non_linear
from . import utils
from torch.nn.init import xavier_uniform_, constant_, eye_
from flamingchoripan import strings as strings

###################################################################################################################################################

class Linear(nn.Module):
	def __init__(self, input_dims:int, output_dims:int,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		split_out=1,
		**kwargs):
		super().__init__()
		### CHECKS
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1
		assert split_out>=0

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.activation = activation
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.bias = bias
		self.uses_custom_non_linear_init = uses_custom_non_linear_init
		self.split_out = split_out

		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.linear = nn.Linear(self.input_dims, self.output_dims*self.split_out, bias=self.bias)
		self.activation_f = non_linear.get_activation(self.activation)
		self.reset()

	def reset(self):
		if self.uses_custom_non_linear_init:
			xavier_uniform_(self.linear.weight, gain=non_linear.get_xavier_gain(self.activation))
			if self.bias is not None:
				constant_(self.linear.bias, 0.0)

	def get_output_dims(self):
		return self.output_dims
		
	def forward(self, x):
		'''
		x: (b,...,t)
		'''
		assert self.input_dims==x.shape[-1]

		x = self.in_dropout_f(x)
		x = self.linear(x)
		x = self.activation_f(x, dim=-1)
		x = self.out_dropout_f(x)

		#### split if required
		if self.split_out>1:
			assert x.shape[-1]%self.split_out==0
			return torch.chunk(x, self.split_out, dim=-1)
		return x

	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'input_dims':self.input_dims,
		'output_dims':self.output_dims,
		'activation':self.activation,
		'in_dropout':self.in_dropout,
		'out_dropout':self.out_dropout,
		'bias':self.bias,
		'split_out':self.split_out
		}, ', ', '=')
		return txt

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		txt = f'Linear({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

class MLP(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		activation='relu',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,

		dropout=0.0,
		last_activation=None,
		**kwargs):
		super().__init__()
		### CHECKS
		assert isinstance(embd_dims_list, list) and len(embd_dims_list)>0
		assert dropout>=0 and dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.embd_dims_list = embd_dims_list.copy()
		self.dropout = dropout
		self.last_activation = last_activation

		self.embd_dims_list.insert(0, self.input_dims) # first
		self.embd_dims_list.append(self.output_dims) # last

		activations = [activation]*(len(self.embd_dims_list)-1) # create activations along
		if not self.last_activation is None:
			activations[-1] = self.last_activation

		self.fcs = nn.ModuleList()
		for k in range(len(self.embd_dims_list)-1):
			input_dims_ = self.embd_dims_list[k]
			output_dims_ = self.embd_dims_list[k+1]
			fc_kwargs = {
				'activation':activations[k],
				'in_dropout':in_dropout if k==0 else self.dropout,
				'out_dropout':out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
				'bias':bias,
			}
			self.fcs.append(Linear(input_dims_, output_dims_, **fc_kwargs))

		self.reset()

	def reset(self):
		for fc in self.fcs:
			fc.reset()

	def get_output_dims(self):
		return self.output_dims

	def forward(self, x):
		'''
		x: (b,...,t)
		'''
		assert self.input_dims==x.shape[-1]

		for fc in self.fcs:
			x = fc(x)
		return x

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		resume = ''
		for k,fc in enumerate(self.fcs):
			resume += f'  ({k}) - {str(fc)}\n'

		txt = f'MLP(\n{resume})({len(self):,}[p])'
		return txt

class PMLP(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		**kwargs):
		super().__init__()
		self.mlp1 = MLP(input_dims, output_dims, embd_dims_list, **kwargs)
		self.mlp2 = MLP(input_dims, output_dims, embd_dims_list, **kwargs)
		self.reset()

	def reset(self):
		self.mlp1.reset()
		self.mlp2.reset()

	def get_output_dims(self):
		return self.mlp1.get_output_dims()

	def forward(self, x1, x2):
		return self.mlp1(x1), self.mlp2(x2)