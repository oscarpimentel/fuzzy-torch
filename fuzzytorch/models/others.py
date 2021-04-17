from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..utils import tensor_to_numpy
from .basics import Linear
from . import utils as utils
import flamingchoripan.strings as strings
import numpy as np
import math

###################################################################################################################################################

def softclamp(x, a, b,
	alpha=0.1,
	):
	assert a<b
	#z = torch.clamp(x, a, s)
	z = F.elu(x-a, alpha=alpha)+a
	z = -(F.elu(-z+b, alpha=alpha)-b)
	return z

###################################################################################################################################################

class FILM(nn.Module):
	def __init__(self, mod_input_dims:int, mod_output_dims:int,
		in_dropout=0.0,
		out_dropout=0.0,
		mod_dropout=0.0,
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
		self.mod_dropout = mod_dropout
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
			self.mod_dropout_f = nn.Dropout(self.mod_dropout)

	def forward(self, x, mod, **kwargs):
		# x (b,t,fx)
		# mod (b,t,fm)
		assert x.shape[-1]==self.mod_output_dims

		if not self.is_dummy:
			x = self.in_dropout_f(x)
			gamma, beta = self.mod_f(self.mod_dropout_f(mod))
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
		'mod_dropout':self.mod_dropout,
		'bias':self.bias,
		}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'FILM({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

###################################################################################################################################################

class TemporalEncoding(nn.Module):
	def __init__(self, te_features, max_te_period,
		min_te_period=2,
		out_dropout=0.0,
		**kwargs):
		super().__init__()

		### CHECKS
		assert te_features>0
		assert te_features%2==0

		self.te_features = te_features
		self.max_te_period = max_te_period
		self.min_te_period = min_te_period
		self.out_dropout = out_dropout
		self.reset()

	def reset(self):
		len_periods = self.te_features//2
		if self.min_te_period is None:
			_periods = np.array([self.max_te_period]*len_periods/2**np.arange(len_periods)) # fixme!!
			#self.min_te_period = 
			assert 0

		_periods = np.linspace(self.max_te_period, self.min_te_period, len_periods)
		self.te_ws = torch.nn.Parameter(torch.as_tensor(2*math.pi/_periods))
		self.te_phases = torch.nn.Parameter(torch.zeros((len_periods)))
		self.te_scales = torch.nn.Parameter(torch.zeros((len_periods)))
		self.out_dropout_f = nn.Dropout(self.out_dropout)

	def extra_repr(self):
		txt = strings.get_string_from_dict({
			'te_features':self.te_features,
			'min_te_period':self.min_te_period,
			'max_te_period':self.max_te_period,
			'out_dropout':self.out_dropout,
			#'te_periods':self.te_periods,
			}, ', ', '=')
		return txt

	def get_info(self):
		assert not self.training, 'you can not access this method in trining mode'
		d = {
			'te_ws':tensor_to_numpy(self.get_te_ws()),
			'te_periods':tensor_to_numpy(self.get_te_periods()),
			'te_phases':tensor_to_numpy(self.get_te_phases()),
			'te_scales':tensor_to_numpy(self.get_te_scales()),
			}
		return d

	def __repr__(self):
		txt = f'TemporalEncoding({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

	def get_output_dims(self):
		return len(self.te_ws)*2

	def get_te_ws(self):
		te_ws = softclamp(self.te_ws, 2*math.pi/self.max_te_period, 2*math.pi/self.min_te_period)
		return te_ws

	def get_te_periods(self):
		te_ws = self.get_te_ws()
		te_periods = 2*math.pi/te_ws
		return te_periods

	def get_te_phases(self):
		return self.te_phases

	def get_te_scales(self):
		return self.te_scales

	def forward(self, time, **kwargs):
		# time (b,t)
		assert len(time.shape)==2

		b,t = time.size()
		encoding = torch.zeros((b, t, self.get_output_dims()), device=time.device) # (b,t,f)

		#te_ws = torch.sqrt(self.te_ws**2+C_.EPS) # positive operation
		te_ws = self.get_te_ws()
		#te_ws = self.te_ws
		#te_periods = torch.clamp(self.te_periods, self.min_te_period, self.max_te_period)
		#te_periods = torch.sigmoid(self.te_periods)*self.max_te_period
		te_phases = self.te_phases
		#te_phases = torch.tanh(self.te_phases)*te_periods
		te_scales = torch.sigmoid(self.te_scales)
		#print(self.max_te_period, self.get_te_periods(), te_phases, te_scales)
		for k in range(0, len(te_ws)):
			w = te_ws[k]
			phi = te_phases[k]
			scale = te_scales[k]

			encoding[...,2*k] = scale*torch.sin(w*time+phi)
			encoding[...,2*k+1] = scale*torch.cos(w*time+phi)
		encoding = self.out_dropout_f(encoding)
		#print(encoding.shape, encoding.device)
		return encoding

	def __len__(self):
		return utils.count_parameters(self)

'''
class SetFunction(nn.Module):
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
		txt = f'SetFunction({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

'''