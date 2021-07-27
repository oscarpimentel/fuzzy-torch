from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import utils
from fuzzytools import strings as strings

###################################################################################################################################################

class LSTM(nn.Module):
	def __init__(self, input_dims, output_dims,
		max_curve_length=None,
		in_dropout=0.,
		out_dropout=0.,
		bias=True,
		bidirectional=False,
		**kwargs):
		super().__init__()

		### CHECKS
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.max_curve_length = max_curve_length
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.bias = bias
		self.bidirectional = bidirectional

		### MODULES
		rnn_kwargs = {
			'num_layers':1,
			'bias':self.bias,
			'batch_first':True,
			'dropout':0.0,
			'bidirectional':self.bidirectional
		}
		self.rnn = torch.nn.LSTM(self.input_dims, self.output_dims, **rnn_kwargs)
		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)

	def reset(self):
		pass
		
	def get_output_dims(self):
		return self.output_dims

	def __len__(self):
		return utils.count_parameters(self)
		
	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'input_dims':self.input_dims,
		'output_dims':self.output_dims,
		'max_curve_length':self.max_curve_length,

		'in_dropout':self.in_dropout,
		'out_dropout':self.out_dropout,
		'bias':self.bias,
		'bidirectional':self.bidirectional,
		}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'LSTM({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

	def forward(self, x_packed,
		hc0=None,
		**kwargs):
		x_packed = nn.utils.rnn.PackedSequence(self.in_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		x_packed, hidden = self.rnn(x_packed, hc0)
		x_packed = nn.utils.rnn.PackedSequence(self.out_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		return x_packed

###################################################################################################################################################

class GRU(nn.Module):
	def __init__(self, input_dims, output_dims,
		max_curve_length=None,
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		bidirectional=False,
		**kwargs):
		super().__init__()

		### CHECKS
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.max_curve_length = max_curve_length
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.bias = bias
		self.bidirectional = bidirectional

		### MODULES
		rnn_kwargs = {
			'num_layers':1,
			'bias':self.bias,
			'batch_first':True,
			'dropout':0.0,
			'bidirectional':self.bidirectional
		}
		self.rnn = torch.nn.GRU(self.input_dims, self.output_dims, **rnn_kwargs)
		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)

	def reset(self):
		pass

	def get_output_dims(self):
		return self.output_dims

	def __len__(self):
		return utils.count_parameters(self)
		
	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'input_dims':self.input_dims,
		'output_dims':self.output_dims,
		'max_curve_length':self.max_curve_length,

		'in_dropout':self.in_dropout,
		'out_dropout':self.out_dropout,
		'bias':self.bias,
		'bidirectional':self.bidirectional,
		}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'GRU({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

	def forward(self, x_packed,
		h0=None,
		**kwargs):
		x_packed = nn.utils.rnn.PackedSequence(self.in_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		x_packed, hidden = self.rnn(x_packed, h0)
		x_packed = nn.utils.rnn.PackedSequence(self.out_dropout_f(x_packed.data), x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices)
		return x_packed

	# def forward(self, x,
	# 	h0=None,
	# 	**kwargs):
	# 	x, hidden = self.rnn(x, h0)
	# 	return x

###################################################################################################################################################

class MLRNN(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		max_curve_length=None,
		in_dropout=0.0,
		dropout=0.0,
		out_dropout=0.0,
		bias=True,
		bidirectional=False,
		**kwargs):
		super().__init__()

		### CHECKS
		assert isinstance(embd_dims_list, list) and len(embd_dims_list)>=0
		assert isinstance(bidirectional, bool)
		assert in_dropout>=0 and in_dropout<=1
		assert dropout>=0 and dropout<=1
		assert out_dropout>=0 and out_dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims//(1+int(bidirectional))
		self.embd_dims_list = [self.input_dims]+embd_dims_list+[self.output_dims]
		self.max_curve_length = max_curve_length
		self.in_dropout = in_dropout
		self.dropout = dropout
		self.out_dropout = out_dropout
		self.bias = bias
		self.bidirectional = bidirectional

		### MODULES
		self.rnns = nn.ModuleList()
		for k in range(len(self.embd_dims_list)-1):
			input_dims_ = self.embd_dims_list[k]
			output_dims_ = self.embd_dims_list[k+1]
			rnn_kwargs = {
				'max_curve_length':self.max_curve_length,
				'in_dropout':self.in_dropout if k==0 else self.dropout,
				'out_dropout':self.out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
				'bias':self.bias,
				'bidirectional':self.bidirectional,
			}
			rnn = self.rnn_class(input_dims_, output_dims_, **rnn_kwargs)
			self.rnns.append(rnn)

		self.reset()

	def reset(self):
		for rnn in self.rnns:
			rnn.reset()

	def get_embd_dims_list(self):
		return self.embd_dims_list

	def get_output_dims(self):
		return self.output_dims

	def forward(self, x, onehot,
		**kwargs):
		'''
		Parameters
		----------
		x (b,t,f): input tensor.
		onehot (b,t)

		Return
		----------
		x_out: (b,t,h): output tensor.
		'''
		assert onehot.dtype==torch.bool
		assert len(onehot.shape)==2
		assert x.shape[:-1]==onehot.shape
		assert len(x.shape)==3

		self.max_curve_length = x.shape[1]
		extra_info = {}
		lengths = torch.clamp(onehot.sum(dim=-1), 1, None) # forced to avoid errors of empty bands sequences
		cpu_lengths = lengths.detach().to('cpu') # lengths needs to be in cpu, is there a fix to this slow operation?
		for k,rnn in enumerate(self.rnns):
			x_packed = nn.utils.rnn.pack_padded_sequence(x, cpu_lengths, batch_first=True, enforce_sorted=False)
			x_packed = rnn(x_packed, **kwargs)
			x,_ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, padding_value=0, total_length=self.max_curve_length) # argument is Sequence
		return x, extra_info

	# def forward(self, x, onehot,
	# 	**kwargs):
	# 	'''
	# 	Parameters
	# 	----------
	# 	x (b,t,f): input tensor.
	# 	onehot (b,t)

	# 	Return
	# 	----------
	# 	x_out: (b,t,h): output tensor.
	# 	'''
	# 	assert onehot.dtype==torch.bool
	# 	assert len(onehot.shape)==2
	# 	assert x.shape[:-1]==onehot.shape
	# 	assert len(x.shape)==3

	# 	extra_info = {}
	# 	for k,rnn in enumerate(self.rnns):
	# 		x = rnn(x, **kwargs)
	# 	return x, extra_info

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		resume = ''
		for k,rnn in enumerate(self.rnns):
			resume += f'  ({k}) - {str(rnn)}\n'
		txt = f'MLRNN(\n{resume})({len(self):,}[p])'
		return txt

###################################################################################################################################################

class MLGRU(MLRNN):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		max_curve_length=None,
		in_dropout=0.0,
		dropout=0.0,
		out_dropout=0.0,
		bias=True,
		bidirectional=False,
		**kwargs):
		self.class_name = 'GRU'
		self.rnn_class = GRU
		super().__init__(input_dims, output_dims, embd_dims_list,
			max_curve_length,
			in_dropout,
			dropout,
			out_dropout,
			bias,
			bidirectional,
			)

###################################################################################################################################################

class MLLSTM(MLRNN):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		max_curve_length=None,
		in_dropout=0.0,
		dropout=0.0,
		out_dropout=0.0,
		bias=True,
		bidirectional=False,
		**kwargs):
		self.class_name = 'LSTM'
		self.rnn_class = LSTM
		super().__init__(input_dims, output_dims, embd_dims_list,
			max_curve_length,
			in_dropout,
			dropout,
			out_dropout,
			bias,
			bidirectional,
			)