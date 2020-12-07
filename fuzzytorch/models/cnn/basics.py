from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F
#from .dummies import DummyModule
from .. import non_linear
from .. import utils
from . import utils as cnn_utils
from torch.nn.init import xavier_uniform_, constant_, eye_
from flamingchoripan import strings as strings

###################################################################################################################################################

class CausalConv1DLinear(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, kernel_size:int,
		**kwargs):
		super().__init__()

		### ATTRIBUTES
		setattr(self, 'input_dims', input_dims)
		setattr(self, 'output_dims', output_dims)
		setattr(self, 'kernel_size', kernel_size)
		setattr(self, 'cnn_stacks', 1)
		setattr(self, 'activation', 'linear')
		setattr(self, 'last_activation', None)
		setattr(self, 'in_dropout', 0.0)
		setattr(self, 'dropout', 0.0)
		setattr(self, 'out_dropout', 0.0)
		setattr(self, 'bias', True)
		setattr(self, 'uses_custom_non_linear_init', False)
		setattr(self, 'split_out', 1)
		setattr(self, 'clean_sequences', True)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### CHECKS
		assert self.cnn_stacks>=1
		assert self.in_dropout>=0 and self.in_dropout<=1
		assert self.dropout>=0 and self.dropout<=1
		assert self.out_dropout>=0 and self.out_dropout<=1
		assert self.kernel_size>=1
		assert self.split_out>=1

		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.causal_padding = nn.ConstantPad1d([self.kernel_size-1,0], 0)
		cnn_kwargs = {
			'kernel_size':self.kernel_size,
			'bias':self.bias,
		}
		self.cnn1d_stack = nn.ModuleList()
		for k in range(self.cnn_stacks):
			cnn1d = nn.Conv1d(self.input_dims if k==0 else self.output_dims, self.output_dims, **cnn_kwargs)
			self.cnn1d_stack.append(cnn1d)

		self.activations = [self.activation]*self.cnn_stacks
		if not self.last_activation is None:
			self.activations[-1] = self.last_activation

		self.activations = [non_linear.get_activation(a) for a in self.activations]
		self._reset_parameters()
		assert 0, 'revisar cosas como el dropout o hacer split de clases'

	def _reset_parameters(self):
		if self.uses_custom_non_linear_init:
			for cnn1d in self.cnn1d_stack:
				torch.nn.init.xavier_uniform_(cnn1d.weight, gain=non_linear.get_xavier_gain(self.activation))
				if self.bias is not None:
					constant_(self.cnn1d.bias, 0.0)

	def get_output_dims(self):
		return self.output_dims

	def clean(self, x, onehot):
		if self.clean_sequences and not onehot is None:
			onehot = onehot.permute(0,2,1) # (b,t,1) > (b,1,t)
			x = x.masked_fill(~onehot, 0) # clean using onehot
		return x

	def forward(self, x, onehot=None):
		'''
		x: (b,t,f)
		onehot: (b,t,1)
		'''
		b,f,t = x.size()
		#print(x.shape)
		x = x.permute(0,2,1) # (b,t,f) > (b,f,t)
		for k,cnn1d in enumerate(self.cnn1d_stack):
			x = self.in_dropout_f(x)
			x = self.clean(x, onehot)
			#print('x1',x[0,0,:])
			x = self.causal_padding(x) if self.kernel_size>1 else x
			#print('x2',x[0,0,:])
			x = cnn1d(x)
			x = self.clean(x, onehot)
			x = self.activations[k](x, dim=-1)

		x = self.out_dropout_f(x)
		x = x.permute(0,2,1) # (b,f,t) > (b,t,f)
		#### split if required
		if self.split_out>1:
			assert f%self.split_out==0
			return torch.chunk(x, self.split_out, dim=1)
		return x

	def extra_repr(self):
		txt = f'input_dims={self.input_dims}, output_dims={self.output_dims}, kernel_size={self.kernel_size}'
		txt += f', in_dropout={self.in_dropout}, activation={self.activation}, bias={self.bias}'
		txt += f', cnn_stacks={self.cnn_stacks}' if self.cnn_stacks>1 else ''
		txt += f', out_drop={self.out_dropout}' if self.out_dropout>0 else ''
		txt += f', split_out={self.split_out}' if self.split_out>1 else ''
		return txt

	def __repr__(self):
		txt = f'CausalConv1DLinear({self.extra_repr()})'
		txt += f'({count_parameters(self):,}[p])'
		return txt

###################################################################################################################################################

class Conv2DLinear(nn.Module):
	def __init__(self, input_dims:int, input_space:list, output_dims:int,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		split_out=1,

		stride=1,
		kernel_size=3,
		padding_mode=None,
		padding=0,
		pool_kernel_size=1,
		**kwargs):
		super().__init__()
		### CHECKS
		assert len(input_space)==2
		kernel_size = [kernel_size]*2 if isinstance(kernel_size, int) else kernel_size
		stride = [stride]*2 if isinstance(stride, int) else stride
		padding = [padding]*2 if isinstance(padding, int) else padding
		pool_kernel_size = [pool_kernel_size]*2 if isinstance(pool_kernel_size, int) else pool_kernel_size
		assert len(kernel_size)==2
		assert len(stride)==2
		assert len(padding)==2
		assert len(pool_kernel_size)==2
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1
		assert split_out>=1

		self.input_dims = input_dims
		self.input_space = input_space
		self.output_dims = output_dims
		self.kernel_size = kernel_size
		self.activation = activation
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.bias = bias
		self.uses_custom_non_linear_init = uses_custom_non_linear_init
		self.split_out = split_out
		self.stride = stride
		self.padding_mode = padding_mode
		self.padding = padding
		self.pool_kernel_size = pool_kernel_size

		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.padding = self.padding if self.padding_mode is None else [cnn_utils.get_padding(self.padding_mode, k) for k in self.kernel_size]
		cnn_kwargs = {
			'kernel_size':self.kernel_size,
			'stride':self.stride,
			'padding':self.padding,
			'bias':self.bias
		}
		self.cnn = nn.Conv2d(self.input_dims, self.output_dims, **cnn_kwargs)
		self.activation_f = non_linear.get_activation(self.activation)
		self.pool = nn.MaxPool2d(self.pool_kernel_size)
		self.reset()

	def reset(self):
		if self.uses_custom_non_linear_init:
			torch.nn.init.xavier_uniform_(cnn2d.weight, gain=non_linear.get_xavier_gain(self.activation))
			if self.bias is not None:
				constant_(self.cnn2d.bias, 0.0)

	def get_output_space(self):
		return [cnn_utils.get_cnn_output_dims(self.input_space[k], self.kernel_size[k], self.padding[k], self.stride[k], pool_kernel_size=self.pool_kernel_size[k]) for k in range(len(self.input_space))]
			
	def get_output_dims(self):
		return self.output_dims

	def forward(self, x):
		'''
		x: (b,f,w,h)
		'''
		b,f,w,h = x.size()
		assert all([w==self.input_space[0], h==self.input_space[1], f==self.input_dims])

		x = self.in_dropout_f(x)
		x = self.cnn(x)
		x = self.pool(x)
		x = self.activation_f(x, dim=-1)
		x = self.out_dropout_f(x)
		#assert all([w==self.input_space[0], h==self.input_space[1], f==self.input_dims])

		#### split if required
		if self.split_out>1:
			assert f%self.split_out==0
			return torch.chunk(x, self.split_out, dim=1)
		return x

	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'input_dims':self.input_dims,
		'input_space':self.input_space,
		'output_dims':self.output_dims,
		'output_space':self.get_output_space(),

		'kernel_size':self.kernel_size,
		'stride':self.stride,
		'padding_mode':self.padding_mode,
		'padding':self.padding,
		'pool_kernel_size':self.pool_kernel_size,

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
		txt = f'Conv2DLinear({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

class MLConv2D(nn.Module):
	def __init__(self, input_dims:int, input_space:list, output_dims:int, embd_dims_list:list,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,

		stride=1,
		kernel_size=3,
		padding_mode=None,
		padding=0,
		pool_kernel_size=1,

		dropout=0.0,
		last_activation=None,
		**kwargs):
		super().__init__()
		### CHECKS
		assert isinstance(embd_dims_list, list) and len(embd_dims_list)>0
		assert dropout>=0 and dropout<=1

		self.input_dims = input_dims
		self.input_space = input_space
		self.output_dims = output_dims
		self.embd_dims_list = embd_dims_list.copy()
		self.dropout = dropout
		self.last_activation = last_activation

		self.embd_dims_list.insert(0, self.input_dims) # first
		self.embd_dims_list.append(self.output_dims) # last

		activations = [activation]*(len(self.embd_dims_list)-1) # create activations along
		if not self.last_activation is None:
			activations[-1] = self.last_activation

		input_space_ = self.input_space
		self.cnns = nn.ModuleList()
		for k in range(len(self.embd_dims_list)-1):
			input_dims_ = self.embd_dims_list[k]
			output_dims_ = self.embd_dims_list[k+1]
			cnn_kwargs = {
				'activation':activations[k],
				'in_dropout':in_dropout if k==0 else self.dropout,
				'out_dropout':out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
				'bias':bias,

				'stride':stride,
				'kernel_size':kernel_size,
				'padding_mode':padding_mode,
				'padding':padding,
				'pool_kernel_size':pool_kernel_size,
			}
			cnn = Conv2DLinear(input_dims_, input_space_, output_dims_, **cnn_kwargs)
			input_space_ = cnn.get_output_space()
			self.cnns.append(cnn)

		self.reset()

	def reset(self):
		for cnn in self.cnns:
			cnn.reset()

	def get_output_dims(self):
		return self.output_dims

	def get_output_space(self):
		return self.cnns[-1].get_output_space()

	def forward(self, x):
		'''
		x: (b,f,w,h)
		'''
		b,f,w,h = x.size()
		assert all([w==self.input_space[0], h==self.input_space[1], f==self.input_dims])

		for cnn in self.cnns:
			x = cnn(x)
		return x

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		resume = ''
		for k,cnn in enumerate(self.cnns):
			resume += f'({k}) - {str(cnn)}\n'

		txt = f'MLConv2D(\n{resume})({len(self):,}[p])'
		return txt
