from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..utils import tensor_to_numpy
from .basics import Linear, MLP
#from cnn.basics import ConvLinear
from . import utils as utils
import flamingchoripan.strings as strings
import numpy as np
import math
from .attn.batch_norms import LayerNorm, MaskedBatchNorm1d

###################################################################################################################################################

def softclamp_lrelu(x, a, b,
	negative_slope=0.001,
	):
	assert a<b
	#z = torch.clamp(x, a, s)
	z = F.leaky_relu(x-a, negative_slope=negative_slope)+a
	z = -(F.leaky_relu(-z+b, negative_slope=negative_slope)-b)
	return z

def softclamp_elu(x, a, b,
	alpha=0.1,
	):
	assert a<b
	#z = torch.clamp(x, a, s)
	z = F.elu(x-a)+a
	z = -(F.elu(-z+b)-b)
	return z

def softclamp(x, a, b):
	return softclamp_lrelu(x, a, b)

def cyclic_mod(x, a, b):
	assert b>a
	return (x-a)%(b-a)+a
	
def xxx(te_ws, te_phases, ntime,
	uses_linear_term=False, # False* True
	):
	'''
	te_ws (f)
	te_phases (f)
	ntime (b,t)
	'''
	#print(te_ws.shape, te_phases.shape, ntime.shape)
	b,t = ntime.size()
	f = len(te_ws)
	_te_ws = te_ws[None,None,:] # (f) > (1,1,f)
	_te_phases = te_phases[None,None,:] # (f) > (1,1,f)
	_ntime = ntime[...,None] # (b,t) > (b,t,1)
	if uses_linear_term:
		encoding1 = _te_ws[...,0][...,None]*_ntime+_te_phases[...,0][...,None] # (b,t,f)
		encoding2 = torch.sin(_te_ws[...,1:]*_ntime+_te_phases[...,1:]) # (b,t,f)
		#print(encoding1.shape, encoding2.shape)
		encoding = torch.cat([encoding1, encoding2], axis=-1)
	else:
		encoding = torch.sin(_te_ws*_ntime+_te_phases) # (b,t,f)
	#print(encoding.shape)
	#te_ws.dtype, te_phases.dtype, ntime.dtype)
	return encoding

def _te(te_ws, te_phases, te_scales, ntime):
	'''
	te_ws (f)
	te_phases (f)
	ntime (b,t)
	'''
	b,t = ntime.size()
	encoding = torch.zeros((b, t, len(te_phases)), device=ntime.device) # (b,t,f)
	for i in range(0, len(te_ws)):
		w = te_ws[i]
		phi = te_phases[i]
		scale = te_scales[i]
		encoding[...,i] = scale*torch.sin(w*ntime+phi)
	return encoding



###################################################################################################################################################

class TemporalEncoding(nn.Module):
	def __init__(self, te_features, max_te_period,
		min_te_period=None, # 2 None
		out_dropout=0.0,
		scale_dropout=0.0,
		ktime=1,
		requires_grad=False, # False True
		random_init=False, # True False
		scale_mode=None, # None sigmoid hardsigmoid softmax
		time_noise=1/24, # regularization in time units: 0 None
		**kwargs):
		super().__init__()

		### CHECKS
		assert te_features>0
		assert te_features%2==0
		assert time_noise is None or time_noise>=0

		self.te_features = te_features
		self.max_te_period = max_te_period
		self.min_te_period = min_te_period
		self.out_dropout = out_dropout
		self.scale_dropout = scale_dropout
		self.ktime = ktime
		self.requires_grad = requires_grad
		self.random_init = random_init
		self.scale_mode = scale_mode
		self.time_noise = self.max_te_period*1e-4 if time_noise is None else time_noise
		self.reset()

	def reset(self):
		periods, phases = self.generate_initial_tensors()
		self.initial_ws = self.period2w(self.time2ntime(periods))
		self.initial_phases = phases

		self.min_w = np.min(self.initial_ws)
		self.max_w = np.max(self.initial_ws)

		if self.random_init:
			#self.te_ws = torch.nn.Parameter(torch.normal(0., 0.1, size=[self.get_output_dims()]), requires_grad=True) # True False
			#self.te_phases = torch.nn.Parameter(torch.normal(0., 0.1, size=[self.get_output_dims()]), requires_grad=True) # True False
			pass
		else:
			self.te_ws = torch.nn.Parameter(torch.as_tensor(self.initial_ws), requires_grad=False) # True False
			self.te_phases = torch.nn.Parameter(torch.as_tensor(self.initial_phases), requires_grad=False) # True False

		self.scale_mode = None#'sigmoid'
		self.te_gate = torch.nn.Parameter(torch.zeros_like(self.te_ws), requires_grad=False if self.scale_mode is None else True) # True False # ahmm
		#self.te_gate = torch.nn.Parameter(torch.zeros_like(self.te_ws), requires_grad=True)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		self.scale_dropout_f = nn.Dropout(self.scale_dropout)

	def generate_initial_tensors(self):
		if self.min_te_period is None:
			n = self.get_output_dims()//2
			#periods = np.repeat(np.array([self.max_te_period/2**i for i in np.arange(n)]), 2, axis=0).astype(np.float32) # juxta
			periods = np.repeat(np.array([self.max_te_period/(i+1) for i in np.arange(n)]), 2, axis=0).astype(np.float32) # fourier
			phases = np.array([math.pi/2 if i%2==0 else 0 for i in range(0, 2*n)]).astype(np.float32)
		else:
			periods = np.linspace(self.max_te_period, self.min_te_period, self.get_output_dims()).astype(np.float32)
			phases = np.zeros_like(periods).astype(np.float32)
		
		return periods, phases

	def xafasf(self):
		if self.ktime is None:
			self.ktime = 1/(np.max(periods)-np.min(periods))

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
			'te_periods':[f'{p:.1f}' for p in tensor_to_numpy(self.get_te_periods())],
			'te_phases':[f'{p:.1f}' for p in tensor_to_numpy(self.get_te_phases())],
			'scale_mode':self.scale_mode,
			'time_noise':self.time_noise,
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
			'te_gate':tensor_to_numpy(self.get_te_gate()),
			'ktime':self.ktime,
			}
		return d

	def __repr__(self):
		txt = f'TemporalEncoding({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

	def get_output_dims(self):
		#return self.te_features+1
		return self.te_features

	def get_te_ws(self):
		if self.requires_grad:
			#te_ws = cyclic_mod(self.te_ws, self.min_w, self.max_w) # horrible
			te_ws = softclamp(self.te_ws, self.min_w, self.max_w)
			#te_ws = softclamp(self.te_ws, 0., self.max_w)
		else:
			te_ws = self.te_ws
		return te_ws

	def get_te_periods(self):
		te_ws = self.get_te_ws()
		te_nperiods = self.w2period(te_ws)
		te_periods = self.ntime2time(te_nperiods)
		return te_periods

	def get_te_gate(self):
		if self.scale_mode is None:
			te_gate = self.te_gate+1
		elif self.scale_mode=='sigmoid':
			te_gate = torch.sigmoid(self.te_gate)
		elif self.scale_mode=='hardsigmoid':
			#te_gate = F.hardsigmoid(self.te_gate)
			te_gate = torch.sigmoid(self.te_gate*1e1)
		elif self.scale_mode=='softmax':
			te_gate = torch.softmax(self.te_gate, dim=-1)
		else:
			raise Exception(f'no mode {self.scale_mode}')

		if self.training:
			#print('te_gate',te_gate)
			pass
		return te_gate

	def get_te_phases(self):
		#te_phases = torch.tanh(self.te_phases)*te_periods
		return self.te_phases

	def forward(self, time, **kwargs):
		# time (b,t)
		assert len(time.shape)==2

		if self.training and self.time_noise>0:
			#print(time, time.device)
			uniform_noise = torch.rand(size=time.shape, device=time.device)
			uniform_noise = self.time_noise*(uniform_noise-0.5)
			#print(uniform_noise)
			time = time+uniform_noise
			#print("2",time)

		ntime = self.time2ntime(time)
		te_ws = self.get_te_ws()
		te_phases = self.get_te_phases()
		te_scales = self.get_te_gate()

		#if self.scale_dropout>0:
		#	te_scales = torch.cat([te_scales[0][None], self.scale_dropout_f(te_scales[1:])], dim=0)
		#print('te_scales',te_scales[0])
		encoding = _te(te_ws, te_phases, te_scales, ntime)
		#encoding = self.get_te_gate()*encoding # element wise gate
		encoding = self.out_dropout_f(encoding)
		#print(encoding.shape, encoding.device)
		return encoding

	def __len__(self):
		return utils.count_parameters(self)

###################################################################################################################################################

class TimeFILM(nn.Module):
	def __init__(self, input_dims:int, te_features, max_te_period,
		in_dropout=0.0,
		out_dropout=0.0,
		mod_dropout=0.0,
		#bias=True, # True False # useful only when using an activation function?
		**kwargs):
		super().__init__()

		### CHECKS
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1

		self.input_dims = input_dims
		self.te_features = te_features
		self.max_te_period = max_te_period

		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.mod_dropout = mod_dropout
		#self.bias = bias
		self.reset()

	def reset(self):
		linear_kwargs = {
			'activation':'linear',
			#'bias':self.bias,
			}
		self.is_dummy = self.input_dims==0
		if not self.is_dummy:

			self.te_mod_alpha = TemporalEncoding(self.te_features, self.max_te_period)
			#self.te_mod_beta = TemporalEncoding(self.te_features, self.max_te_period)
			print('te_mod_alpha:',self.te_mod_alpha)

			self.fourier_dims = int(self.input_dims*1)

			#self.gamma_f = Linear(self.te_features, self.fourier_dims, bias=False, **linear_kwargs) # BIAS MUST BE FALSE
			#self.beta_f = Linear(self.te_features, self.fourier_dims, bias=False, **linear_kwargs) # BIAS MUST BE FALSE
			self.gamma_beta_f = Linear(self.te_features, self.fourier_dims, split_out=2, bias=False, **linear_kwargs) # BIAS MUST BE FALSE

			#self.gamma_w = nn.Parameter(torch.ones((self.mod_output_dims, self.mod_input_dims)), requires_grad=False)
			#self.gamma_f = nn.Linear(self.mod_input_dims, self.mod_output_dims, bias=False)
			
			self.x_proj = Linear(self.input_dims, self.fourier_dims, bias=False, **linear_kwargs) # BIAS MUST BE FALSE
			
			self.z_proj = Linear(self.fourier_dims, self.input_dims, bias=True, **linear_kwargs)
			print('z_proj',self.z_proj)

			### BUG when using conv???
			kernel_size = 2
			self.cnn_pad = nn.ConstantPad1d([kernel_size-1, 0], 0)
			self.cnn = nn.Conv1d(self.fourier_dims, self.input_dims, kernel_size=kernel_size, padding=0, bias=True)


			#self.gamma_beta_mlp = MLP(self.mod_input_dims, self.mod_output_dims*2, [self.mod_input_dims], activation='relu')
			#self.bn_fourier = MaskedBatchNorm1d(self.fourier_dims, affine=False)# if self.uses_length_wise_batchnorm else LayerNorm(self.input_dims)
			#self.bn = MaskedBatchNorm1d(self.input_dims)# if self.uses_length_wise_batchnorm else LayerNorm(self.input_dims)

			self.in_dropout_f = nn.Dropout(self.in_dropout)
			self.mod_dropout_f = nn.Dropout(self.mod_dropout)
			self.out_dropout_f = nn.Dropout(self.out_dropout)

	def get_info(self):
		assert not self.training, 'you can not access this method in trining mode'
		d = {
			'weight':tensor_to_numpy(self.gamma_beta_f.linear.weight),
			}
		d.update(self.te_mod_alpha.get_info())
		return d

	def mod_x(self, x, te_alpha, onehot):
		#gamma = self.gamma_f(torch.cat([x,mod], dim=-1))
		#gamma = self.gamma_f(te_alpha)
		#beta = self.beta_f(te_beta)
		_gamma, beta = self.gamma_beta_f(te_alpha)
		#mod_x = self.x_proj(x)*gamma+beta
		#mod_x = x*gamma+beta
		#gamma = _gamma+1
		gamma = _gamma
		mod_x = self.x_proj(x)*gamma+beta

		#mod_x = self.z_proj(mod_x)
		mod_x = mod_x.permute(0,2,1);mod_x = self.cnn_pad(mod_x);mod_x = self.cnn(mod_x);mod_x = mod_x.permute(0,2,1)

		return mod_x

	def forward(self, x, time, onehot, **kwargs):
		# x (b,t,fx)
		# time (b,t)
		assert x.shape[-1]==self.input_dims

		if not self.is_dummy:
			x = self.in_dropout_f(x)
			te_alpha = self.te_mod_alpha(time)
			#te_beta = self.te_mod_beta(time)
			
			#x = self.mod_x(x, te_alpha, te_beta)

			#x = self.mod_x(x, te_alpha, onehot)
			#x = self.bn(x, onehot) # PRE NORM
			#x = sub_x
			x = self.mod_x(x, te_alpha, onehot)+x # RES
			#x = self.bn(x, onehot) # POST NORM
			x = F.relu(x) # act
			#print(x)

		return x

	def __len__(self):
		return utils.count_parameters(self)
		
	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'input_dims':self.input_dims,
		'in_dropout':self.in_dropout,
		'out_dropout':self.out_dropout,
		'mod_dropout':self.mod_dropout,
		'fourier_dims':self.fourier_dims,
		#'bias':self.bias,
		}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'TimeFILM({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

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