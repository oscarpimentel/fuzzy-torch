from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import non_linear
from .. import utils
from . import utils as attn_utils
from ..basics import MLP
from torch.nn.init import xavier_uniform_, constant_, eye_
from fuzzytools import strings as strings
from fuzzytools import lists as lists
from .pytorch_multihead_clone import MultiheadAttention
#from torch.nn import MultiheadAttention
from .batch_norms import LayerNorm, MaskedBatchNorm1d
from ..others import FILM, TemporalEncoding
from .. import seq_utils as seq_utils
import numpy as np

###################################################################################################################################################

class SelfAttn(nn.Module):
	def __init__(self, input_dims:int, output_dims:int,
		max_curve_length=None,
		num_heads=2,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		attn_dropout=0.0,
		mlp_dropout=0.0,
		residual_dropout=0.0,
		bias=True,
		uses_length_wise_batchnorm=1,
		**kwargs):
		super().__init__()

		### CHECKS
		assert input_dims%num_heads==0
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1
		assert attn_dropout>=0 and attn_dropout<=1
		assert mlp_dropout>=0 and mlp_dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.max_curve_length = max_curve_length
		self.num_heads = num_heads
		self.activation = activation
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.attn_dropout = attn_dropout
		self.mlp_dropout = mlp_dropout
		self.residual_dropout = residual_dropout
		self.bias = bias
		self.uses_length_wise_batchnorm = uses_length_wise_batchnorm

		### ATTN
		attn_kwargs = {
			'dropout':self.attn_dropout,
			'bias':self.bias,
			'add_bias_kv':False,
			'add_zero_attn':False,
			'kdim':None,
			'vdim':None,
		}
		self.mh_attn = MultiheadAttention(self.input_dims, self.num_heads, **attn_kwargs)
		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.res1_dropout_f = nn.Dropout(self.residual_dropout)
		self.res2_dropout_f = nn.Dropout(self.residual_dropout)

		### MLP
		mlp_kwargs = {
			'activation':'relu', # transformer
			'in_dropout':0.,
			'out_dropout':0.,
			'bias':self.bias,
			'dropout':self.mlp_dropout,
			'last_activation':'linear', # transformer
		}
		self.mlp = MLP(self.input_dims, self.output_dims, [self.input_dims*2]*1, **mlp_kwargs)
		
		### BATCH NORM
		# MaskedBatchNorm1d is buggy?
		self.attn_bn = MaskedBatchNorm1d(self.input_dims) if self.uses_length_wise_batchnorm else LayerNorm(self.input_dims)
		self.mlp_bn = MaskedBatchNorm1d(self.input_dims) if self.uses_length_wise_batchnorm else LayerNorm(self.input_dims)

		self.activation_f = non_linear.get_activation(self.activation)
		self.reset()

	def reset(self):
		pass

	def register_src_mask(self, max_curve_length, device):
		max_curve_length_changed = not max_curve_length==self.max_curve_length
		if max_curve_length_changed:
			self.max_curve_length = max_curve_length
			#self.register_buffer('src_mask', attn_utils.generate_square_subsequent_mask(self.max_curve_length).to(device))
			self.src_mask = attn_utils.generate_square_subsequent_mask(self.max_curve_length).to(device) # slow to use .to() but it's not always
			#print(self.src_mask.device)

	def get_output_dims(self):
		return self.output_dims

	def __len__(self):
		return utils.count_parameters(self)
		
	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'input_dims':self.input_dims,
		'output_dims':self.output_dims,
		'max_curve_length':self.max_curve_length,
		'num_heads':self.num_heads,
		'activation':self.activation,
		'in_dropout':self.in_dropout,
		'out_dropout':self.out_dropout,
		'attn_dropout':self.attn_dropout,
		'mlp_dropout':self.mlp_dropout,
		'bias':self.bias,
		}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'SelfAttn({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

	def forward(self, x, onehot,
		mul_attn_mask=None,
		**kwargs):
		'''
		Parameters
		----------
		x (b,t,in): input tensor.
		onehot (b,t)

		Return
		----------
		x: (b,t,out): output tensor.
		scores: (b,h,t,qt)
		'''
		self.register_src_mask(x.shape[1], x.device)

		new_onehot = onehot.clone()
		new_onehot[:,0] = True # forced to avoid errors of empty bands sequences

		attn_kwargs = {
			'key_padding_mask':~new_onehot,
			'attn_mask':self.src_mask,
			'mul_attn_mask':mul_attn_mask,
			'need_weights':True,
		}
		x = self.in_dropout_f(x)
		queries = x.permute(1,0,2)
		keys = x.permute(1,0,2)
		values = x.permute(1,0,2)
		contexts, scores = self.mh_attn(queries, keys, values, **attn_kwargs)
		#assert 0
		
		#scores = scores.cpu()
		#print(scores.device)
		#assert torch.all(scores.sum(dim=-1)>=0.99999)
		x = contexts+self.res1_dropout_f(values) # res
		x = x.permute(1,0,2)
		x = self.attn_bn(x, onehot)

		x = self.mlp(x)+self.res2_dropout_f(x) # res
		x = self.mlp_bn(x, onehot)
		x = self.activation_f(x, dim=-1)
		x = self.out_dropout_f(x)
		#print(scores.shape)
		return x, scores

class MLSelfAttn(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
		max_curve_length=None,
		num_heads=2,
		activation=_C.DEFAULT_ACTIVATION,
		last_activation=_C.DEFAULT_LAST_ACTIVATION,
		in_dropout=0.0,
		dropout=0.0,
		out_dropout=0.0,
		attn_dropout=0.0,
		bias=True,
		uses_length_wise_batchnorm=1,
		**kwargs):
		super().__init__()

		### CHECKS
		assert in_dropout>=0 and in_dropout<=1
		assert dropout>=0 and dropout<=1
		assert out_dropout>=0 and out_dropout<=1
		assert attn_dropout>=0 and attn_dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.embd_dims_list = [self.input_dims]+embd_dims_list+[self.output_dims]
		self.max_curve_length = max_curve_length
		self.num_heads = num_heads
		self.activation = activation
		self.last_activation = last_activation
		self.in_dropout = in_dropout
		self.dropout = dropout
		self.out_dropout = out_dropout
		self.attn_dropout = attn_dropout
		self.bias = bias
		self.uses_length_wise_batchnorm = uses_length_wise_batchnorm

		activations = [activation]*(len(self.embd_dims_list)-1) # create activations along
		if not self.last_activation is None:
			activations[-1] = self.last_activation

		### MODULES
		self.self_attns = nn.ModuleList()
		for k in range(len(self.embd_dims_list)-1):
			input_dims_ = self.embd_dims_list[k]
			output_dims_ = self.embd_dims_list[k+1]
			attn_kwargs = {
				'max_curve_length':self.max_curve_length,
				'num_heads':self.num_heads,
				'activation':activations[k],
				'in_dropout':self.in_dropout if k==0 else self.dropout,
				'out_dropout':self.out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
				'attn_dropout':self.attn_dropout,
				'bias':self.bias,
				'uses_length_wise_batchnorm':self.uses_length_wise_batchnorm,
			}
			self_attn = SelfAttn(input_dims_, output_dims_, **attn_kwargs)
			self.self_attns += [self_attn]

		self.reset()

	def reset(self):
		for self_attn in self.self_attns:
			self_attn.reset()

	def get_embd_dims_list(self):
		return self.embd_dims_list

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		resume = ''
		for k,self_attn in enumerate(self.self_attns):
			resume += f'  ({k}) - {str(self_attn)}\n'
		txt = f'MLSelfAttn(\n{resume})({len(self):,}[p])'
		return txt

	def forward(self, x, onehot, **kwargs):
		'''
		Parameters
		----------
		x (b,t,in): input tensor.
		onehot (b,t)

		Return
		----------
		x: (b,t,out): output tensor.
		layers_scores: (b,layers,h,t,qt)
		'''
		assert onehot.dtype==torch.bool
		assert len(onehot.shape)==2
		assert x.shape[:-1]==onehot.shape
		assert len(x.shape)==3

		layers_scores = []
		for k,self_attn in enumerate(self.self_attns):
			x, layer_scores = self_attn(x, onehot, **kwargs)
			layers_scores += [layer_scores[:,None,...]]

		layers_scores = torch.cat(layers_scores, dim=1)
		return x, layers_scores

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		resume = ''
		for k,self_attn in enumerate(self.self_attns):
			resume += f'  ({k}) - {str(self_attn)}\n'
		txt = f'MLSelfAttn(\n{resume})({len(self):,}[p])'
		return txt

###################################################################################################################################################

class TimeErrorSelfAttn(SelfAttn):
	def __init__(self, input_dims:int, output_dims:int,
		max_curve_length=None,
		num_heads=2,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		attn_dropout=0.0,
		mlp_dropout=0.0,
		residual_dropout=0.0,
		bias=True,
		uses_length_wise_batchnorm=1,
		**kwargs):
		super().__init__(
			input_dims, output_dims,
			max_curve_length,
			num_heads,
			activation,
			in_dropout,
			out_dropout,
			attn_dropout,
			mlp_dropout,
			residual_dropout,
			bias,
			uses_length_wise_batchnorm,
			**kwargs
			)
		self.error_a = torch.nn.Parameter(torch.tensor([1.]*self.num_heads), requires_grad=True)
		self.error_b = torch.nn.Parameter(torch.tensor([0.]*self.num_heads), requires_grad=True)
		self.min_error = np.infty
		self.max_error = -np.infty
		#self.min_error = 0
		#self.max_error = 0.05

	def forward(self, x, onehot, error,
		**kwargs):
		'''
		#print(self.src_mask.shape, self.src_mask)
		assert torch.all(error>=0)
		error = error[...,None]
		#print(error.shape, error)

		if self.training:
			min_error = torch.min(seq_utils.seq_min_pooling(error.detach(), onehot.detach())).item()
			self.min_error = min_error if min_error<self.min_error else self.min_error
			max_error = torch.max(seq_utils.seq_max_pooling(error.detach(), onehot).detach()).detach().item()
			self.max_error = max_error if max_error>self.max_error else self.max_error
			pass
		else:
			#print(self.error_a, self.error_b)
			pass

		error = error.permute(0,2,1)[:,None,...] # (b,t,1) > (b,1,1,t)
		error_mask = error.repeat(1, self.num_heads, x.shape[1], 1) # (b,h,t,t)
		#print(error_mask.shape, error_mask)
		#pos_error_a = torch.log(torch.exp(self.error_a)+_C.EPS)
		pos_error_a = torch.sqrt(self.error_a**2+_C.EPS)
		#pos_error_a = torch.clamp(self.error_a, _C.EPS, None)
		error_b = self.error_b
		#error_mask = 2*(error_mask-self.min_error)/(self.max_error-self.min_error)-1 # norm [-1,1]
		extra_info = {
			#'min_error':self.min_error,
			#'max_error':self.max_error,
			#'pos_error_a':pos_error_a,
			#'error_b':error_b,
		}
		'''

		mul_attn_mask = None # dummy hehe
		#mul_attn_mask = 1-torch.sigmoid(pos_error_a[None,:,None,None]*error_mask+self.error_b[None,:,None,None])
		#mul_attn_mask = 1-torch.sigmoid(pos_error_a[None,:,None,None]*error_mask)
		#mul_attn_mask = 1-torch.sigmoid(error_mask)

		x, scores = super().forward(x, onehot, mul_attn_mask=mul_attn_mask, **kwargs)
		return x, scores

class MLTimeErrorSelfAttn(nn.Module):
	def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list, te_features, max_te_period,
		max_curve_length=None,
		num_heads=2,
		activation=_C.DEFAULT_ACTIVATION,
		last_activation=_C.DEFAULT_LAST_ACTIVATION,
		in_dropout=0.0,
		dropout=0.0,
		out_dropout=0.0,
		attn_dropout=0.0,
		bias=True,
		uses_length_wise_batchnorm=1,
		**kwargs):
		super().__init__()

		### CHECKS
		assert in_dropout>=0 and in_dropout<=1
		assert dropout>=0 and dropout<=1
		assert out_dropout>=0 and out_dropout<=1
		assert attn_dropout>=0 and attn_dropout<=1

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.embd_dims_list = [self.input_dims]+embd_dims_list+[self.output_dims]
		self.te_features = te_features
		self.max_te_period = max_te_period
		self.max_curve_length = max_curve_length
		self.num_heads = num_heads
		self.activation = activation
		self.last_activation = last_activation
		self.in_dropout = in_dropout
		self.dropout = dropout
		self.out_dropout = out_dropout
		self.attn_dropout = attn_dropout
		self.bias = bias
		self.uses_length_wise_batchnorm = uses_length_wise_batchnorm

		activations = [activation]*(len(self.embd_dims_list)-1) # create activations along
		if not self.last_activation is None:
			activations[-1] = self.last_activation

		### MODULES
		self.te_mods = nn.ModuleList()
		self.self_attns = nn.ModuleList()
		self.te_films = nn.ModuleList()
		for k in range(len(self.embd_dims_list)-1):
			input_dims_ = self.embd_dims_list[k]
			output_dims_ = self.embd_dims_list[k+1]
			attn_kwargs = {
				'max_curve_length':self.max_curve_length,
				'num_heads':self.num_heads,
				'activation':activations[k],
				'in_dropout':self.in_dropout if k==0 else self.dropout,
				'out_dropout':self.out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
				'attn_dropout':self.attn_dropout,
				'bias':self.bias,
				'uses_length_wise_batchnorm':self.uses_length_wise_batchnorm,
			}
			self_attn = TimeErrorSelfAttn(input_dims_, output_dims_, **attn_kwargs)
			self.self_attns += [self_attn]
			te_mod = TemporalEncoding(self.te_features, self.max_te_period)
			print('te_mod:',te_mod)
			self.te_mods += [te_mod]

			film_kwargs = {
				#'in_dropout':self.dropout,
			}
			film = FILM(te_mod.get_output_dims(), input_dims_, **film_kwargs)
			self.te_films += [film]

		self.reset()

	def reset(self):
		for self_attn in self.self_attns:
			self_attn.reset()

	def get_embd_dims_list(self):
		return self.embd_dims_list

	def __len__(self):
		return utils.count_parameters(self)

	def __repr__(self):
		resume = ''
		for k,self_attn in enumerate(self.self_attns):
			resume += f'  ({k}) - {str(self_attn)}\n'
		txt = f'MLTimeErrorSelfAttn(\n{resume})({len(self):,}[p])'
		return txt

	def get_info(self):
		d = {
			'te_mod':[te_mod.get_info() for te_mod in self.te_mods],
			}
		return d

	def forward(self, x, onehot, time, error, **kwargs):
		'''
		Parameters
		----------
		x (b,t,in): input tensor.
		onehot (b,t)
		time (b,t)

		Return
		----------
		x: (b,t,out): output tensor.
		scores: (b,layers,h,t,qt)
		'''
		assert onehot.dtype==torch.bool
		assert len(onehot.shape)==2
		assert x.shape[:-1]==onehot.shape
		assert len(x.shape)==3
		assert len(time.shape)==2

		outs = []
		scores = []

		for k,(te_film,self_attn,te_mod) in enumerate(zip(self.te_films, self.self_attns, self.te_mods)):
			te = te_mod(time)
			x = te_film(x, te)
			x, _scores = self_attn(x, onehot, error, **kwargs)
			outs += [x]
			scores += [_scores]

		return outs, scores