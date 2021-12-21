from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..utils import tensor_to_numpy
from .basics import Linear, MLP, ResidualBlockHandler
from . import non_linear
from . import utils as utils
import fuzzytools.strings as strings
import numpy as np
import math
from .attn.batch_norms import LayerNorm, MaskedBatchNorm1d

REMOVES_TIME_OFFSET = False
WS_PHASES_REQUIRES_GRAD = False

###################################################################################################################################################

def _get_te(te_ws, te_phases, te_scales, time,
	linear_term_k=0,
	vectorized=True,
	):
	if vectorized:
		return _vectorized_te(te_ws, te_phases, te_scales, time,
			linear_term_k=linear_term_k,
			)
	else:
		return _nonvectorized_te(te_ws, te_phases, te_scales, time,
			linear_term_k=linear_term_k,
			)

def _vectorized_te(te_ws, te_phases, te_scales, time,
	linear_term_k=0,
	):
	'''
	te_ws (f)
	te_phases (f)
	te_scales (f)
	time (n,t)
	'''
	b,t = time.size()
	_te_ws = te_ws[None,None,:] # (f) > (1,1,f)
	_te_phases = te_phases[None,None,:] # (f) > (1,1,f)
	_te_scales = te_scales[None,None,:] # (f) > (1,1,f)
	_time = time[...,None] # (n,t) > (n,t,1)
	if linear_term_k==0:
		sin_arg = _te_ws*_time+_te_phases # (n,t,f)
		encoding = _te_scales*torch.sin(sin_arg) # (n,t,f)
	else:
		assert 0, 'not implemented'
		# encoding1 = _te_ws[...,0][...,None]*_time+_te_phases[...,0][...,None] # (n,t,f)
		# encoding2 = torch.sin(_te_ws[...,1:]*_time+_te_phases[...,1:]) # (n,t,f)
		# encoding = torch.cat([encoding1, encoding2], axis=-1)
	return encoding

def _nonvectorized_te(te_ws, te_phases, te_scales, time,
	linear_term_k=0,
	):
	'''
	te_ws (f)
	te_phases (f)
	te_scales (f)
	time (n,t)
	'''
	n,t = time.size()
	encoding = torch.zeros((n, t, len(te_phases)), device=time.device) # (n,t,f)
	if linear_term_k==0:
		for i in range(0, len(te_ws)):
			sin_arg = te_ws[i]*time+te_phases[i]
			encoding[...,i] = te_scales[i]*torch.sin(sin_arg)
	else:
		assert 0, 'not implemented'
	return encoding

###################################################################################################################################################

class TemporalEncoder(nn.Module):
	def __init__(self, te_features, max_te_period,
		min_te_period=None,
		time_noise_window=0, # regularization in time units
		init_k_exp=.5,
		ws_phases_requires_grad=WS_PHASES_REQUIRES_GRAD,
		removes_time_offset=REMOVES_TIME_OFFSET,
		**kwargs):
		super().__init__()
		### CHECKS
		assert te_features>0
		assert te_features%2==0
		assert init_k_exp>=0

		self.te_features = te_features
		self.max_te_period = max_te_period
		self.min_te_period = min_te_period
		self.time_noise_window = eval(time_noise_window) if isinstance(time_noise_window, str) else time_noise_window
		self.init_k_exp = init_k_exp
		self.ws_phases_requires_grad = ws_phases_requires_grad
		self.removes_time_offset = removes_time_offset
		self.reset()

	def reset(self):
		periods, phases = self.generate_initial_tensors()
		self.initial_ws = self.period2w(periods)
		self.initial_phases = phases

		self.te_ws = torch.nn.Parameter(torch.as_tensor(self.initial_ws), requires_grad=self.ws_phases_requires_grad) # from lower to higher frequencies
		self.te_phases = torch.nn.Parameter(torch.as_tensor(self.initial_phases), requires_grad=self.ws_phases_requires_grad)

		n = self.get_output_dims()//2
		te_scales = np.array([math.exp(-math.floor(i/2)*self.init_k_exp) for i in range(0, 2*n)]).astype(np.float32) # exponential initialization to start training with smooth functions
		self.te_scales = torch.nn.Parameter(torch.as_tensor(te_scales), requires_grad=False)

	def generate_initial_tensors(self):
		'''
		# Tmax/1, Tmax/1, Tmax/2, TMax/2, , Tmax/3, TMax/3, ...
		'''
		if self.min_te_period is None:
			n = self.get_output_dims()//2
			periods = np.repeat(np.array([self.max_te_period/(i+1) for i in np.arange(0, n)]), 2, axis=0).astype(np.float32) # from higher to lower periods
			phases = np.array([math.pi/2 if i%2==0 else 0 for i in range(0, 2*n)]).astype(np.float32) # for sin, cos
		else:
			periods = np.linspace(self.max_te_period, self.min_te_period, self.get_output_dims()).astype(np.float32)
			phases = np.zeros_like(periods).astype(np.float32)
		
		return periods, phases

	def w2period(self, w):
		return (2*math.pi)/w

	def period2w(self, period):
		return (2*math.pi)/period

	def extra_repr(self):
		txt = strings.get_string_from_dict({
			'te_features':self.te_features,
			'min_te_period':self.min_te_period,
			'max_te_period':self.max_te_period,
			'te_ws':[f'{p:.3f}' for p in tensor_to_numpy(self.get_te_ws())],
			'te_periods':[f'{p:.3f}' for p in tensor_to_numpy(self.get_te_periods())],
			'te_phases':[f'{p:.3f}' for p in tensor_to_numpy(self.get_te_phases())],
			'te_scales':[f'{p:.5f}' for p in tensor_to_numpy(self.get_te_scales())],
			'time_noise_window':self.time_noise_window,
			'init_k_exp':self.init_k_exp,
			'removes_time_offset':self.removes_time_offset,
			}, ', ', '=')
		return txt

	def get_info(self):
		assert not self.training, 'you can not access this method in trining mode'
		d = {
			'te_features':self.te_features,
			'initial_ws':self.initial_ws,
			'initial_phases':self.initial_phases,
			'te_ws':tensor_to_numpy(self.get_te_ws()),
			'te_periods':tensor_to_numpy(self.get_te_periods()),
			'te_phases':tensor_to_numpy(self.get_te_phases()),
			'te_scales':tensor_to_numpy(self.get_te_scales()),
			}
		return d

	def __repr__(self):
		txt = f'TemporalEncoder({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

	def get_output_dims(self):
		return self.te_features

	def get_te_ws(self):
		te_ws = self.te_ws
		return te_ws

	def get_te_periods(self):
		te_ws = self.get_te_ws()
		te_periods = self.w2period(te_ws)
		return te_periods

	def get_te_scales(self):
		return self.te_scales

	def get_te_phases(self):
		return self.te_phases

	def forward(self, time, **kwargs):
		# time (n,t)
		assert len(time.shape)==2
		time = time-time[:,0][...,None] if self.removes_time_offset else time # (n,t)
		if self.training and self.time_noise_window>0:
			uniform_noise = torch.rand(size=(1, time.shape[1]), device=time.device) # (1,t) # (0,1) noise
			uniform_noise = self.time_noise_window*(uniform_noise-0.5) # k*(-0.5,0.5)
			time = time+uniform_noise # (n,t)+(1,t)>(n,t)

		te_ws = self.get_te_ws()
		te_phases = self.get_te_phases()
		te_scales = self.get_te_scales()
		encoding = _get_te(te_ws, te_phases, te_scales, time)
		return encoding

	def __len__(self):
		return utils.get_nof_parameters(self)

###################################################################################################################################################

class TimeFILM(nn.Module):
	def __init__(self, input_dims, te_features, max_te_period,
		kernel_size=1,
		time_noise_window=0, # regularization in time units
		activation='relu',
		uses_norm=False,
		removes_time_offset=REMOVES_TIME_OFFSET,
		**kwargs):
		super().__init__()
		### CHECKS
		self.input_dims = input_dims
		self.te_features = te_features
		self.max_te_period = max_te_period

		self.kernel_size = kernel_size
		self.time_noise_window = time_noise_window
		self.activation = activation
		self.uses_norm = uses_norm
		self.removes_time_offset = removes_time_offset
		self.reset()

	def reset(self):
		self.dummy = self.te_features<=0
		if not self.is_dummy():
			assert self.input_dims>0
			self.temporal_encoder = TemporalEncoder(self.te_features, self.max_te_period,
				time_noise_window=self.time_noise_window,
				removes_time_offset=self.removes_time_offset,
				)
			
			self.gamma_beta_f = Linear(self.te_features, self.input_dims,
				split_out=2,
				bias=False, # BIAS MUST BE FALSE
				activation='linear',
				)

		self.cnn_pad = nn.ConstantPad1d([self.kernel_size-1, 0], 0)
		self.cnn = nn.Conv1d(self.input_dims, self.input_dims,
			kernel_size=self.kernel_size,
			padding=0,
			bias=True,
			)

		if self.uses_norm:
			self.norm = torch.nn.LayerNorm([self.input_dims])

		self.activation_f = non_linear.get_activation(self.activation)

	def get_info(self):
		if not self.is_dummy():
			assert not self.training, 'you can not access this method in training mode'
			d = {
				'weight':tensor_to_numpy(self.gamma_beta_f.linear.weight), # (2K,2M)
				}
			d.update(self.temporal_encoder.get_info())
			return d
		else:
			return {}

	def is_dummy(self):
		return self.dummy

	def f_mod(self, x, time, onehot):
		if self.is_dummy():
			x_mod = x*1+0 # for ablation
		else:
			temporal_encoding = self.temporal_encoder(time) # (n,t,2M)
			gamma, beta = self.gamma_beta_f(temporal_encoding) # (n,t,2M)>(n,t,2K)>[(n,t,K),(n,t,K)]
			x_mod = x*torch.tanh(gamma)+beta # element-wise modulation

		if self.uses_norm:
			x_mod = self.norm(x_mod)

		x_mod = x_mod.permute(0,2,1) # (n,t,f)>(n,f,t)
		x_mod = self.cnn(self.cnn_pad(x_mod)) # (n,f,t)
		x_mod = x_mod.permute(0,2,1) # (n,f,t)>(n,t,fx)
		return x_mod

	def forward(self, x, time, onehot, **kwargs):
		# x: (n,t,f)
		# time: (n,t)
		assert x.shape[-1]==self.input_dims

		x_mod = self.f_mod(x, time, onehot)
		x_mod = self.activation_f(x_mod, dim=-1)
		return x_mod

	def __len__(self):
		return utils.get_nof_parameters(self)
		
	def extra_repr(self):
		txt = strings.get_string_from_dict({
			'\ntemporal_encoder':f'{self.temporal_encoder}\n',
			'activation':self.activation,
			'input_dims':self.input_dims,
			'kernel_size':self.kernel_size,
			}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'TimeFILM({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt