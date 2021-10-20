from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import non_linear
from . import utils
from fuzzytools import strings as strings

DEFAULT_NON_LINEAR_ACTIVATION = _C.DEFAULT_NON_LINEAR_ACTIVATION
NORM_MODE = 'pre_norm' # none pre_norm post_norm

###################################################################################################################################################

class DummyModule(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		pass

	def reset(self):
		pass

	def reset_parameters(self):
		pass

###################################################################################################################################################

class ResidualBlockHandler(nn.Module):
	def __init__(self, f,
		norm=None,
		norm_mode=NORM_MODE,
		activation='linear',
		residual_dropout=0.0,
		ignore_f=False, # used for ablations
		**kwargs):
		super().__init__()
		### CHECKS
		assert residual_dropout>=0 and residual_dropout<=1
		assert norm_mode in ['none', 'pre_norm', 'post_norm']

		self.f = f
		self.norm = DummyModule() if norm is None else norm
		self.norm_mode = norm_mode
		self.activation = activation
		self.residual_dropout = residual_dropout
		self.ignore_f = ignore_f
		self.reset()

	def reset(self):
		self.activation_f = non_linear.get_activation(self.activation)
		self.residual_dropout_f = nn.Dropout(self.residual_dropout)
		if hasattr(self.f, 'reset'):
			self.f.reset()
		if hasattr(self.norm, 'reset'):
			self.norm.reset()
		self.reset_parameters()

	def reset_parameters(self):
		if hasattr(self.f, 'reset_parameters'):
			self.f.reset_parameters()
		if hasattr(self.norm, 'reset_parameters'):
			self.norm.reset_parameters()
	
	def norm_x(self, x,
		norm_args=[],
		norm_kwargs={},
		):
		new_x = self.norm(x, *norm_args, **norm_kwargs)
		assert new_x.shape==x.shape
		return new_x

	def forward(self, x,
		f_args=[],
		f_kwargs={},
		norm_args=[],
		norm_kwargs={},
		f_returns_tuple=False,
		):
		'''
		auxiliar class for residual connection
		'''
		if self.norm_mode=='pre_norm':
			norm_x = self.norm_x(x, norm_args=norm_args, norm_kwargs=norm_kwargs)
			fx_args = self.f(norm_x, *f_args, **f_kwargs)
		else:
			fx_args = self.f(x, *f_args, **f_kwargs)

		fx = fx_args[0] if f_returns_tuple else fx_args
		if self.ignore_f:
			new_x = x
		else:
			new_x = x+self.residual_dropout_f(fx) # x=x+f(x)

		if self.norm_mode=='post_norm':
			new_x = self.norm_x(new_x, norm_args=norm_args, norm_kwargs=norm_kwargs)
			
		new_x = self.activation_f(new_x, dim=-1)
		assert x.shape==new_x.shape
		if f_returns_tuple:
			return tuple([new_x])+fx_args[1:]
		else:
			return new_x

###################################################################################################################################################

class Linear(nn.Module):
	def __init__(self, input_dims:int, output_dims:int,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		split_out=1,
		bias_value=None,
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
		self.split_out = split_out
		self.bias_value = bias_value
		self.reset()

	def reset(self):
		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.linear = nn.Linear(self.input_dims, self.output_dims*self.split_out, bias=self.bias)
		self.activation_f = non_linear.get_activation(self.activation)
		self.reset_parameters()

	def reset_parameters(self):
		self.linear.reset_parameters()
		# torch.nn.init.xavier_uniform_(self.linear.weight, gain=non_linear.get_xavier_gain(self.activation)) # ugly bug???
		if not self.bias is None and not self.bias_value is None:
			torch.nn.init.constant_(self.linear.bias, self.bias_value)

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
		'split_out':self.split_out,
		'bias_value':self.bias_value,
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
		activation=DEFAULT_NON_LINEAR_ACTIVATION,
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		dropout=0.0,
		last_activation='linear',
		**kwargs):
		super().__init__()
		### CHECKS
		assert isinstance(embd_dims_list, list) and len(embd_dims_list)>=0
		assert dropout>=0 and dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.embd_dims_list = [self.input_dims]+embd_dims_list+[self.output_dims]
		self.activation = activation
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.bias = bias
		self.dropout = dropout
		self.last_activation = last_activation
		self.reset()

	def reset(self):
		activations = [self.activation]*(len(self.embd_dims_list)-1) # create activations along
		if not self.last_activation is None:
			activations[-1] = self.last_activation

		self.fcs = nn.ModuleList()
		for k in range(0, len(self.embd_dims_list)-1):
			_input_dims = self.embd_dims_list[k]
			_output_dims = self.embd_dims_list[k+1]
			self.fcs.append(Linear(_input_dims, _output_dims,
				activation=activations[k],
				in_dropout=self.in_dropout if k==0 else self.dropout,
				out_dropout=self.out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
				bias=self.bias,
				))
		self.reset_parameters()

	def reset_parameters(self):
		for fc in self.fcs:
			fc.reset_parameters()

	def get_embd_dims_list(self):
		return self.embd_dims_list
		
	def get_output_dims(self):
		return self.output_dims

	def forward(self, x):
		'''
		x: (b,...,f)
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