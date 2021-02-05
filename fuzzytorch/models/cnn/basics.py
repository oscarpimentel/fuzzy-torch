from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import non_linear
from .. import utils
from . import utils as cnn_utils
from torch.nn.init import xavier_uniform_, constant_, eye_
from flamingchoripan import strings as strings
from flamingchoripan import lists as lists

###################################################################################################################################################

class ConvLinear(nn.Module):
	def __init__(self, input_dims:int, input_space:list, output_dims:int,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		auto_update_input_space=True,

		cnn_kwargs=C_.DEFAULT_CNN_KWARGS,
		pool_kwargs=C_.DEFAULT_POOL_KWARGS,
		padding_mode=C_.DEFAULT_PADDING_MODE,
		**kwargs):
		super().__init__()

		### CHECKS
		assert len(input_space)==self.len_input_space_shape
		assert in_dropout>=0 and in_dropout<=1
		assert out_dropout>=0 and out_dropout<=1

		self.input_dims = input_dims
		self.input_space = input_space.copy()
		self.output_dims = output_dims
		self.activation = activation
		self.in_dropout = in_dropout
		self.out_dropout = out_dropout
		self.bias = bias
		self.uses_custom_non_linear_init = uses_custom_non_linear_init
		self.auto_update_input_space = auto_update_input_space

		self.cnn_kwargs = cnn_utils.get_correct_cnn_kwargs(self.len_input_space_shape, cnn_kwargs)
		self.pool_kwargs = cnn_utils.get_correct_cnn_kwargs(self.len_input_space_shape, pool_kwargs)
		self.padding_mode = padding_mode

		self.in_dropout_f = nn.Dropout(self.in_dropout)
		self.out_dropout_f = nn.Dropout(self.out_dropout)
		
		self.cnn_spatial_field = [cnn_utils.get_spatial_field(self.cnn_kwargs, k) for k in range(len(self.input_space))]
		self.pool_spatial_field = [cnn_utils.get_spatial_field(self.pool_kwargs, k) for k in range(len(self.input_space))]

		self.cnn_padding_lr = [cnn_utils.get_pad_value(self.padding_mode, True, self.cnn_kwargs, k) for k in range(len(self.input_space))]
		self.pool_padding_lr = [cnn_utils.get_pad_value(self.padding_mode, False, self.pool_kwargs, k) for k in range(len(self.input_space))]

		self.cnn_padding = [sum(self.cnn_padding_lr[k]) for k in range(len(self.input_space))]
		self.pool_padding = [sum(self.pool_padding_lr[k]) for k in range(len(self.input_space))]
		self.cnn_pad = self.class_dict['pad'](lists.flat_list(self.cnn_padding_lr), 0)
		self.pool_pad = self.class_dict['pad'](lists.flat_list(self.pool_padding_lr), 0)

		self.cnn = self.class_dict['cnn'](self.input_dims, self.output_dims, **self.cnn_kwargs, padding=0, bias=self.bias)
		self.activation_f = non_linear.get_activation(self.activation)
		self.pool = self.class_dict['pool'](**self.pool_kwargs, padding=0)
		self.reset()

	def reset(self):
		if self.uses_custom_non_linear_init:
			torch.nn.init.xavier_uniform_(cnn2d.weight, gain=non_linear.get_xavier_gain(self.activation))
			if self.bias is not None:
				constant_(self.cnn2d.bias, 0.0)

	def get_output_space(self):
		output_space = self.input_space
		output_space = [cnn_utils.get_output_space(output_space, self.cnn_kwargs, self.cnn_padding, k) for k in range(len(self.input_space))]
		output_space = [cnn_utils.get_output_space(output_space, self.pool_kwargs, self.pool_padding, k) for k in range(len(self.input_space))]
		return output_space
	
	def get_spatial_field(self):
		spatial_field = [self.cnn_spatial_field[k]+self.pool_spatial_field[k]-1 for k in range(len(self.input_space))]
		return spatial_field
		
	def get_output_dims(self):
		return self.output_dims

	def forward(self, x):
		self.forward_checks(x)
		x = self.in_dropout_f(x)
		#print('pre-pad',x.shape);print('pre-pad',x[0,0,:]);print()
		x = self.cnn_pad(x)
		#print('pre-cnn',x.shape);print('pre-cnn',x[0,0,:]);print()
		x = self.cnn(x)
		#print('pre-pad-pool',x.shape);print('pre-pad',x[0,0,:]);print()
		x = self.pool_pad(x)
		#print('pre-pool',x.shape);print('pre-pool',x[0,0,:]);print()
		x = self.pool(x)
		#print('post-pool',x.shape);print('post-pool',x[0,0,:]);print()
		x = self.activation_f(x, dim=-1)
		x = self.out_dropout_f(x)
		assert list(x.shape)[-self.len_input_space_shape:]==self.get_output_space()
		return x

	def __len__(self):
		return utils.count_parameters(self)

	def extra_repr(self):
		txt = strings.get_string_from_dict({
		'input_dims':self.input_dims,
		'input_space':self.input_space,
		'output_dims':self.output_dims,
		'output_space':self.get_output_space(),
		'spatial_field':self.get_spatial_field(),

		'cnn_kwargs':self.cnn_kwargs,
		'pool_kwargs':self.pool_kwargs,
		'padding_mode':self.padding_mode,

		'activation':self.activation,
		'in_dropout':self.in_dropout,
		'out_dropout':self.out_dropout,
		'bias':self.bias,
		}, ', ', '=')
		return txt

	def __repr__(self):
		txt = f'{self.class_name}({self.extra_repr()})'
		txt += f'({len(self):,}[p])'
		return txt

class MLConv(nn.Module):
	def __init__(self, input_dims:int, input_space:list, output_dims:int, embd_dims_list:list,
		activation=C_.DEFAULT_ACTIVATION,
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		auto_update_input_space=True,

		cnn_kwargs=C_.DEFAULT_CNN_KWARGS,
		pool_kwargs=C_.DEFAULT_POOL_KWARGS,
		padding_mode=C_.DEFAULT_PADDING_MODE,

		dropout=0.,
		last_activation=C_.DEFAULT_LAST_ACTIVATION,
		**kwargs):
		super().__init__()
		### CHECKS
		assert isinstance(embd_dims_list, list) and len(embd_dims_list)>=0
		assert dropout>=0 and dropout<=1

		self.input_dims = input_dims
		self.input_space = input_space
		self.output_dims = output_dims
		self.embd_dims_list = [self.input_dims]+embd_dims_list+[self.output_dims]
		self.dropout = dropout
		self.last_activation = last_activation

		activations = [activation]*(len(self.embd_dims_list)-1) # create activations along
		if not self.last_activation is None:
			activations[-1] = self.last_activation

		input_space_ = self.input_space
		self.cnns = nn.ModuleList()
		for k in range(len(self.embd_dims_list)-1):
			input_dims_ = self.embd_dims_list[k]
			output_dims_ = self.embd_dims_list[k+1]
			cnn_linear_kwargs = {
				'activation':activations[k],
				'in_dropout':in_dropout if k==0 else self.dropout,
				'out_dropout':out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
				'bias':bias,
				'uses_custom_non_linear_init':uses_custom_non_linear_init,
				'auto_update_input_space':auto_update_input_space,

				'cnn_kwargs':cnn_kwargs,
				'pool_kwargs':pool_kwargs,
				'padding_mode':padding_mode,
			}
			cnn = self.cnn_class(input_dims_, input_space_, output_dims_, **cnn_linear_kwargs)
			output_space_ = cnn.get_output_space()
			#print(f'{input_space_}({input_dims_})>{output_space_}({output_dims_})')
			self.cnns.append(cnn)
			input_space_ = output_space_

		self.reset()

	def reset(self):
		for cnn in self.cnns:
			cnn.reset()

	def get_embd_dims_list(self):
		return self.embd_dims_list
		
	def get_output_dims(self):
		return self.cnns[-1].get_output_dims()

	def get_output_space(self):
		return self.cnns[-1].get_output_space()

	def forward(self, x):
		for cnn in self.cnns:
			x = cnn(x)
		return x

	def __len__(self):
		return utils.count_parameters(self)

###################################################################################################################################################

class Conv1DLinear(ConvLinear):
	def __init__(self, input_dims:int, input_space:list, output_dims:int,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		auto_update_input_space=True,

		cnn_kwargs=C_.DEFAULT_CNN_KWARGS,
		pool_kwargs=C_.DEFAULT_POOL_KWARGS,
		padding_mode=C_.DEFAULT_PADDING_MODE,
		**kwargs):
		self.class_name = 'Conv1DLinear'
		self.len_input_space_shape = 1
		self.class_dict = {
			'pad':nn.ConstantPad1d,
			'cnn':nn.Conv1d,
			'pool':nn.MaxPool1d,
		}
		super().__init__(input_dims, input_space, output_dims,
			activation,
			in_dropout,
			out_dropout,
			bias,
			uses_custom_non_linear_init,
			auto_update_input_space,

			cnn_kwargs,
			pool_kwargs,
			padding_mode,
			)

	def forward_checks(self, x):
		'''
		x: (b,f,t)
		'''
		if self.auto_update_input_space:
			self.input_space = [x.shape[2]] # auto
		assert len(x.shape)==3
		b,f,t = x.size()
		assert all([f==self.input_dims, t==self.input_space[0]])

class MLConv1D(MLConv):
	def __init__(self, input_dims:int, input_space:list, output_dims:int, embd_dims_list:list,
		activation=C_.DEFAULT_ACTIVATION,
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		auto_update_input_space=True,

		cnn_kwargs=C_.DEFAULT_CNN_KWARGS,
		pool_kwargs=C_.DEFAULT_POOL_KWARGS,
		padding_mode=C_.DEFAULT_PADDING_MODE,

		dropout=0.,
		last_activation=C_.DEFAULT_LAST_ACTIVATION,
		**kwargs):
		self.cnn_class = Conv1DLinear
		super().__init__(input_dims, input_space, output_dims, embd_dims_list,
			activation,
			in_dropout,
			out_dropout,
			bias,
			uses_custom_non_linear_init,
			auto_update_input_space,

			cnn_kwargs,
			pool_kwargs,
			padding_mode,

			dropout,
			last_activation,
		)

	def __repr__(self):
		resume = ''
		for k,cnn in enumerate(self.cnns):
			resume += f'  ({k}) - {str(cnn)}\n'
		txt = f'MLConv1D(\n{resume})({len(self):,}[p])'
		return txt

###################################################################################################################################################

class Conv2DLinear(ConvLinear):
	def __init__(self, input_dims:int, input_space:list, output_dims:int,
		activation='linear',
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		auto_update_input_space=True,

		cnn_kwargs=C_.DEFAULT_CNN_KWARGS,
		pool_kwargs=C_.DEFAULT_POOL_KWARGS,
		padding_mode=C_.DEFAULT_PADDING_MODE,
		**kwargs):
		self.class_name = 'Conv2DLinear'
		self.len_input_space_shape = 2
		self.class_dict = {
			'pad':nn.ConstantPad2d,
			'cnn':nn.Conv2d,
			'pool':nn.MaxPool2d,
		}
		super().__init__(input_dims, input_space, output_dims,
			activation,
			in_dropout,
			out_dropout,
			bias,
			uses_custom_non_linear_init,
			auto_update_input_space,

			cnn_kwargs,
			pool_kwargs,
			padding_mode,
			)

	def forward_checks(self, x):
		'''
		x: (b,f,w,h)
		'''
		if self.auto_update_input_space:
			self.input_space = list(x.shape[2:]) # auto
		assert len(x.shape)==4
		b,f,w,h = x.size()
		assert all([f==self.input_dims, w==self.input_space[0], h==self.input_space[1]])

class MLConv2D(MLConv):
	def __init__(self, input_dims:int, input_space:list, output_dims:int, embd_dims_list:list,
		activation=C_.DEFAULT_ACTIVATION,
		in_dropout=0.0,
		out_dropout=0.0,
		bias=True,
		uses_custom_non_linear_init=False,
		auto_update_input_space=True,

		cnn_kwargs=C_.DEFAULT_CNN_KWARGS,
		pool_kwargs=C_.DEFAULT_POOL_KWARGS,
		padding_mode=C_.DEFAULT_PADDING_MODE,

		dropout=0.,
		last_activation=C_.DEFAULT_LAST_ACTIVATION,
		**kwargs):
		self.cnn_class = Conv2DLinear
		super().__init__(input_dims, input_space, output_dims, embd_dims_list,
			activation,
			in_dropout,
			out_dropout,
			bias,
			uses_custom_non_linear_init,
			auto_update_input_space,

			cnn_kwargs,
			pool_kwargs,
			padding_mode,

			dropout,
			last_activation,
		)

	def __repr__(self):
		resume = ''
		for k,cnn in enumerate(self.cnns):
			resume += f'  ({k}) - {str(cnn)}\n'
		txt = f'Conv2D(\n{resume})({len(self):,}[p])'
		return txt