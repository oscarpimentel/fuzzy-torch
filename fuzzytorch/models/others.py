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

def softclamp_lrelu(x, a, b):
	assert a<b
	#z = torch.clamp(x, a, s)
	z = F.leaky_relu(x-a)+a
	z = -(F.leaky_relu(-z+b)-b)
	return z

def softclamp_elu(x, a, b):
	assert a<b
	#z = torch.clamp(x, a, s)
	z = F.elu(x-a)+a
	z = -(F.elu(-z+b)-b)
	return z

def softclamp(x, a, b):
	return softclamp_lrelu(x, a, b)

def _te(te_ws, te_phases, te_scales, ntime):
	'''
	te_ws (f)
	te_phases (f)
	te_scales (f)
	ntime (b,t)
	'''
	b,t = ntime.size()
	f = len(te_ws)
	_te_ws = te_ws[None,None,:] # (f) > (1,1,f)
	_te_phases = te_phases[None,None,:] # (f) > (1,1,f)
	_te_scales = te_scales[None,None,:] # (f) > (1,1,f)
	_ntime = ntime[...,None] # (b,t) > (b,t,1)
	if True:
		encoding2 = _te_scales*torch.sin(_te_ws*_ntime+_te_phases) # (b,t,f)
	else:
		encoding1 = _te_scales[...,0][...,None]*(_te_ws[...,0][...,None]*_ntime+_te_phases[...,0][...,None]) # (b,t,f)
		encoding2 = _te_scales[...,1:]*torch.sin(_te_ws[...,1:]*_ntime+_te_phases[...,1:]) # (b,t,f)
		#print(encoding1.shape, encoding2.shape)
		encoding = torch.cat([encoding1, encoding2], axis=-1)
	#print(te_ws.dtype, te_phases.dtype, te_scales.dtype, ntime.dtype)
	return encoding

def _te_old(te_ws, te_phases, te_scales, ntime):
	'''
	te_ws (f)
	te_phases (f)
	te_scales (f)
	ntime (b,t)
	'''
	#print(self.max_te_period, self.get_te_periods(), te_phases, te_scales)
	b,t = time.size()
	encoding = torch.zeros((b, t, self.get_output_dims()), device=time.device) # (b,t,f)
	for i in range(0, len(te_ws)):
		w = te_ws[i]
		phi = te_phases[i]
		scale = te_scales[i]

		encoding[...,i] = scale*torch.sin(w*ntime+phi)
		#encoding[...,2*i] = scale*(w*ntime+phi)
		#encoding[...,2*i] = scale*torch.sin(w*ntime+phi)
		#encoding[...,2*i+1] = scale*torch.cos(w*ntime+phi)
	return encoding

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
		ktime=1,
		**kwargs):
		super().__init__()

		### CHECKS
		assert te_features>0
		assert te_features%2==0

		self.te_features = te_features
		self.max_te_period = max_te_period
		self.min_te_period = min_te_period
		self.out_dropout = out_dropout
		self.ktime = ktime
		self.reset()

	def reset(self):
		periods = self.generate_initial_periods()
		ws = self.period2w(self.time2ntime(periods))
		self.min_w = np.min(ws)
		self.max_w = np.max(ws)
		self.initial_ws = torch.as_tensor(ws)
		self.te_ws = torch.nn.Parameter(self.initial_ws.clone(), requires_grad=True) # True False
		self.te_phases = torch.nn.Parameter(torch.zeros((self.get_output_dims())), requires_grad=True) # True False
		self.te_scales = torch.nn.Parameter(torch.zeros((self.get_output_dims())), requires_grad=False) # True False*
		self.out_dropout_f = nn.Dropout(self.out_dropout)

	def generate_initial_periods(self):
		#if self.min_te_period is None:
			#periods = np.array([self.max_te_period]*len_periods/2**np.arange(len_periods)) # fixme!!
		#	assert 0

		periods = np.linspace(self.max_te_period, self.min_te_period, self.get_output_dims()).astype(np.float32)
		if self.ktime is None:
			self.ktime = 1/(np.max(periods)-np.min(periods))
		return periods

	def time2ntime(self, t):
		return t*self.ktime

	def ntime2time(self, nt):
		return nt/self.ktime

	def w2period(self, w):
		return 2*math.pi/w

	def period2w(self, period):
		return 2*math.pi/period

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
			'te_features':self.te_features,
			'initial_ws':tensor_to_numpy(self.initial_ws),
			'te_ws':tensor_to_numpy(self.get_te_ws()),
			'te_periods':tensor_to_numpy(self.get_te_periods()),
			'te_phases':tensor_to_numpy(self.get_te_phases()),
			'te_scales':tensor_to_numpy(self.get_te_scales()),
			'ktime':self.ktime,
			}
		return d

	def __repr__(self):
		txt = f'TemporalEncoding({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

	def get_output_dims(self):
		return self.te_features+1

	def get_te_ws(self):
		te_ws = softclamp(self.te_ws, self.min_w, self.max_w)
		return te_ws

	def get_te_periods(self):
		te_ws = self.get_te_ws()
		te_nperiods = self.w2period(te_ws)
		te_periods = self.ntime2time(te_nperiods)
		return te_periods

	def get_te_phases(self):
		#te_phases = torch.tanh(self.te_phases)*te_periods
		return self.te_phases

	def get_te_scales(self):
		k = 1
		te_scales = torch.sigmoid(k*self.te_scales)
		#te_scales = (torch.tanh(k*self.te_scales)+1.)/2.
		return te_scales

	def forward(self, time, **kwargs):
		# time (b,t)
		assert len(time.shape)==2

		#te_ws = torch.sqrt(self.te_ws**2+C_.EPS) # positive operation
		ntime = self.time2ntime(time)
		te_ws = self.get_te_ws()
		te_phases = self.get_te_phases()
		te_scales = self.get_te_scales()

		encoding = _te(te_ws, te_phases, te_scales, ntime)
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