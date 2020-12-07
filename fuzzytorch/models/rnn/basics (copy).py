from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F
#from .dummies import DummyModule
from . import non_linear
from . import utils
from torch.nn.init import xavier_uniform_, constant_, eye_

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

class Conv2DLinear(nn.Module):
	def __init__(self, input_dims:int, input2d_dims:list, output_dims:int, kernel_size:list, **kwargs):
		super().__init__()

		### ATTRIBUTES
		setattr(self, 'input_dims', input_dims)
		setattr(self, 'input2d_dims', input2d_dims)
		setattr(self, 'output_dims', output_dims)
		setattr(self, 'kernel_size', kernel_size)
		setattr(self, 'cnn_stacks', 1)
		setattr(self, 'activation', 'linear')
		setattr(self, 'in_dropout', 0.0)
		setattr(self, 'dropout', 0.0)
		setattr(self, 'out_dropout', 0.0)
		setattr(self, 'bias', True)
		setattr(self, 'uses_custom_non_linear_init', False)
		setattr(self, 'split_out', 1)
		setattr(self, 'stride', 1)
		setattr(self, 'padding_mode', None)
		setattr(self, 'padding', 0)
		setattr(self, 'pool_kernel_size', 1)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### CHECKS
		assert self.cnn_stacks>=1
		assert len(self.input2d_dims)==2
		self.kernel_size = [self.kernel_size]*2 if isinstance(self.kernel_size, int) else self.kernel_size
		self.stride = [self.stride]*2 if isinstance(self.stride, int) else self.stride
		self.padding = [self.padding]*2 if isinstance(self.padding, int) else self.padding
		self.pool_kernel_size = [self.pool_kernel_size]*2 if isinstance(self.pool_kernel_size, int) else self.pool_kernel_size
		assert len(self.kernel_size)==2
		assert len(self.stride)==2
		assert len(self.padding)==2
		assert len(self.pool_kernel_size)==2
		assert self.in_dropout>=0 and self.in_dropout<=1
		assert self.dropout>=0 and self.dropout<=1
		assert self.out_dropout>=0 and self.out_dropout<=1
		assert self.split_out>=1

		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.padding = self.padding if self.padding_mode is None else [get_padding(self.padding_mode, k) for k in self.kernel_size]
		cnn_kwargs = {
			'kernel_size':self.kernel_size,
			'stride':self.stride,
			'padding':self.padding,
			'bias':self.bias
		}
		self.cnn2d_stack = nn.ModuleList()
		for k in range(self.cnn_stacks):
			self.cnn2d_stack.append(nn.Conv2d(self.input_dims if k==0 else self.output_dims, self.output_dims, **cnn_kwargs))

		self.activation_f = non_linear.get_activation(self.activation)
		self.pool = nn.MaxPool2d(self.pool_kernel_size)
		_,self.output2d_dims = self.get_output_dims()
		self._reset_parameters()
		assert 0, 'revisar cosas como el dropout o hacer split de clases'

	def _reset_parameters(self):
		if self.uses_custom_non_linear_init:
			for cnn2d in self.cnn2d_stack:
				torch.nn.init.xavier_uniform_(cnn2d.weight, gain=non_linear.get_xavier_gain(self.activation))
				if self.bias is not None:
					constant_(self.cnn2d.bias, 0.0)

	def get_output_dims(self, return_output2d_dims=True):
		if return_output2d_dims:
			output2d_dims = [get_cnn_output_dims(self.input2d_dims[k], self.kernel_size[k], self.padding[k], self.stride[k], self.cnn_stacks, self.pool_kernel_size[k]) for k in range(len(self.kernel_size))]
			return self.output_dims, output2d_dims
		return self.output_dims

	def forward(self, x):
		'''
		x: (b,f,w,h)
		'''
		b,f,w,h = x.size()
		for cnn2d in self.cnn2d_stack:
			x = self.in_dropout_f(x)
			x = cnn2d(x)
			x = self.activation_f(x, dim=-1)

		x = self.pool(x)
		x = self.out_dropout_f(x)

		#### split if required
		if self.split_out>1:
			assert f%self.split_out == 0
			return torch.chunk(x, self.split_out, dim=1)
		return x

	def extra_repr(self):
		txt = f'input_dims={self.input_dims}, output_dims={self.output_dims}, kernel_size={self.kernel_size}'
		txt += f', input2d_dims={self.input2d_dims}, output2d_dims={self.output2d_dims}'
		txt += f', in_dropout={self.in_dropout}, activation={self.activation}, bias={self.bias}'
		txt += f', padding={self.padding}'
		txt += f', stride={self.stride}'
		txt += f', pool_kernel_size={self.pool_kernel_size}'
		txt += f', cnn_stacks={self.cnn_stacks}' if self.cnn_stacks>1 else ''
		txt += f', out_drop={self.out_dropout}' if self.out_dropout>0 else ''
		txt += f', out_drop={self.out_dropout}' if self.out_dropout>0 else ''
		txt += f', split_out={self.split_out}' if self.split_out>1 else ''
		return txt

	def __repr__(self):
		txt = f'Conv2DLinear({self.extra_repr()})'
		txt += f'({count_parameters(self):,}[p])'
		return txt

#####################################################################
### SOME MODELS

class Linear(nn.Module):
	def __init__(self, input_dims:int, output_dims:int,
		**kwargs):
		super().__init__()

		### ATTRIBUTES
		setattr(self, 'input_dims', input_dims)
		setattr(self, 'output_dims', output_dims)
		setattr(self, 'activation', 'linear')
		setattr(self, 'in_dropout', 0.0)
		setattr(self, 'out_dropout', 0.0)
		setattr(self, 'bias', True)
		setattr(self, 'uses_custom_non_linear_init', False)
		setattr(self, 'split_out', 1)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### CHECKS
		assert self.in_dropout>=0 and self.in_dropout<=1
		assert self.out_dropout>=0 and self.out_dropout<=1
		assert self.split_out>=0

		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.linear = nn.Linear(self.input_dims, self.output_dims*self.split_out, bias=self.bias)
		self.activation_f = non_linear.get_activation(self.activation)
		self._reset_parameters()

	def _reset_parameters(self):
		if self.uses_custom_non_linear_init:
			xavier_uniform_(self.linear.weight, gain=non_linear.get_xavier_gain(self.activation))
			if self.bias is not None:
				constant_(self.linear.bias, 0.0)

	def get_output_dims(self):
		return self.output_dims
		
	def forward(self, x):
		'''
		x: (b,f,t)
		'''
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
		txt = f'input_dims={self.input_dims}, output_dims={self.output_dims}'
		txt += f', activation={self.activation}, bias={self.bias}'
		txt += f', in_dropout={self.in_dropout}, out_drop={self.out_dropout}'
		txt += f', split_out={self.split_out}' if self.split_out>1 else ''
		return txt

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		txt = f'Linear({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

class MLP(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		**kwargs):
		super().__init__()

		### ATTRIBUTES
		setattr(self, 'input_dims', input_dims)
		setattr(self, 'output_dims', output_dims)
		setattr(self, 'embd_dims_list', embd_dims_list.copy())
		setattr(self, 'activation', 'relu')
		setattr(self, 'in_dropout', 0.0)
		setattr(self, 'dropout', 0.0)
		setattr(self, 'out_dropout', 0.0)
		setattr(self, 'last_activation', None)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### CHECKS
		assert isinstance(embd_dims_list, list)
		assert self.in_dropout>=0 and self.in_dropout<=1
		assert self.dropout>=0 and self.dropout<=1
		assert self.out_dropout>=0 and self.out_dropout<=1

		self.embd_dims_list.insert(0, self.input_dims)
		self.embd_dims_list.append(self.output_dims)

		self.activations = [self.activation]*(len(self.embd_dims_list)-1)
		if not self.last_activation is None:
			self.activations[-1] = self.last_activation

		#self.in_dropout_f = nn.Dropout(self.in_dropout)
		#self.out_dropout_f = nn.Dropout(self.out_dropout)

		self.fcs = nn.ModuleList()
		for k in range(len(self.embd_dims_list)-1):
			in_units = self.embd_dims_list[k]
			out_units = self.embd_dims_list[k+1]
			self.fcs.append(Linear(in_units, out_units,
				activation=self.activations[k],
				in_dropout=self.in_dropout if k==0 else self.dropout,
				out_dropout=self.out_dropout if k==(len(self.embd_dims_list)-1)-1 else 0.0),
			)

	def _reset_parameters(self):
		for fc in self.fcs:
			fc._reset_parameters()

	def get_output_dims(self):
		return self.output_dims

	def forward(self, x):
		for fc in self.fcs:
			x = fc(x)
		return x

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		resume = ''
		for k,fc in enumerate(self.fcs):
			resume += f'({k}) - {str(fc)}\n'

		txt = f'MLP(\n{resume})({len(self):,}[p])'
		return txt

class PMLP(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		**kwargs):
		super().__init__()
		self.mlp1 = MLP(input_dims, output_dims, embd_dims_list, **kwargs)
		self.mlp2 = MLP(input_dims, output_dims, embd_dims_list, **kwargs)

	def _reset_parameters(self):
		self.mlp1._reset_parameters()
		self.mlp2._reset_parameters()

	def get_output_dims(self):
		return self.mlp1.get_output_dims()

	def forward(self, x1, x2):
		return self.mlp1(x1), self.mlp2(x2)


#####################################################################
### SOME MODELS

#class RNN(nn.Module):

class LSTM(nn.Module):
	def __init__(self, input_dims, output_dims, max_curve_length,
		**kwargs):
		super().__init__()

		### ATTRIBUTES
		setattr(self, 'input_dims', input_dims)
		setattr(self, 'output_dims', output_dims)
		setattr(self, 'max_curve_length', max_curve_length)
		setattr(self, 'in_dropout', 0.0)
		setattr(self, 'out_dropout', 0.0)
		setattr(self, 'bidirectional', False)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### CHECKS
		assert self.in_dropout>=0 and self.in_dropout<=1
		assert self.out_dropout>=0 and self.out_dropout<=1

		### MODULES
		rnn_kwargs = {
			'num_layers':1,
			'bias':True,
			'batch_first':True,
			'dropout':0.0,
			'bidirectional':self.bidirectional
		}
		self.rnn = torch.nn.LSTM(self.input_dims, self.output_dims, **rnn_kwargs)
		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)

	def get_output_dims(self):
		return self.output_dims

	def extra_repr(self):
		txt = f'input_dims={self.input_dims}, output_dims={self.output_dims}, max_curve_length={self.max_curve_length}'
		txt += f', in_dropout={self.in_dropout}, out_dropout={self.out_dropout}'
		txt += f', bidirectional={self.bidirectional}' if self.bidirectional else ''
		return txt

	def __repr__(self):
		txt = f'LSTM({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt


	def forward(self, x_packed, **kwargs):
		x_packed = nn.utils.rnn.PackedSequence(self.in_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		x_packed, hidden = self.rnn(x_packed)
		x_packed = nn.utils.rnn.PackedSequence(self.out_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		return x_packed

class GRU(nn.Module):
	def __init__(self, input_dims, output_dims, max_curve_length,
		**kwargs):
		super().__init__()

		### ATTRIBUTES
		setattr(self, 'input_dims', input_dims)
		setattr(self, 'output_dims', output_dims)
		setattr(self, 'max_curve_length', max_curve_length)
		setattr(self, 'in_dropout', 0.0)
		setattr(self, 'out_dropout', 0.0)
		setattr(self, 'bidirectional', False)
		for name, val in kwargs.items():
			setattr(self, name, val)

		### CHECKS
		assert self.in_dropout>=0 and self.in_dropout<=1
		assert self.out_dropout>=0 and self.out_dropout<=1

		### MODULES
		rnn_kwargs = {
			'num_layers':1,
			'bias':True,
			'batch_first':True,
			'dropout':0.0,
			'bidirectional':self.bidirectional
		}
		self.rnn = torch.nn.GRU(self.input_dims, self.output_dims, **rnn_kwargs)
		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)

	def get_output_dims(self):
		return self.output_dims

	def extra_repr(self):
		txt = f'input_dims={self.input_dims}, output_dims={self.output_dims}, max_curve_length={self.max_curve_length}'
		txt += f', in_dropout={self.in_dropout}, out_dropout={self.out_dropout}'
		txt += f', bidirectional={self.bidirectional}' if self.bidirectional else ''
		return txt

	def __repr__(self):
		txt = f'GRU({self.extra_repr()})'
		txt += f'({count_parameters(self):,}[p])'
		return txt


	def forward(self, x_packed, **kwargs):
		x_packed = nn.utils.rnn.PackedSequence(self.in_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		x_packed, hidden = self.rnn(x_packed)
		x_packed = nn.utils.rnn.PackedSequence(self.out_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		return x_packed