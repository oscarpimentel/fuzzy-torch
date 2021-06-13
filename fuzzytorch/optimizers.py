from __future__ import print_function
from __future__ import division
from . import C_

import torch.nn as nn
import pandas as pd
import torch

###################################################################################################################################################

class LossOptimizer:
	def __init__(self, to_optimize_model, opt_class, opt_kwargs_f,
		clip_grad=None,
		model_get_parameters_f=None,
		**kwargs):
		assert isinstance(to_optimize_model, nn.Module)
		self.to_optimize_model = to_optimize_model
		self.opt_class = opt_class
		self.opt_kwargs_f = opt_kwargs_f
		self.clip_grad = clip_grad
		self.model_get_parameters_f = model_get_parameters_f
		self.reset()

	def reset(self):
		self.epoch_counter = 0
		self.calcule_opt_kwargs()
		self.to_optimize_model = self.to_optimize_model if self.model_get_parameters_f is None else getattr(self.to_optimize_model, self.model_get_parameters_f)()
		self.generate_mounted_optimizer()
		self.set_gradient_clip_hooks()

	def get_model_parameters(self):
		return self.to_optimize_model.parameters() # iterable that exahuste

	def set_gradient_clip_hooks(self):
		if not self.clip_grad is None:
			assert self.clip_grad>0
			for p in self.get_model_parameters():
				if p.requires_grad:
					p.register_hook(lambda grad:torch.clamp(grad, -self.clip_grad, self.clip_grad))

	def calcule_opt_kwargs(self):
		self.opt_kwargs = {}
		for k in self.opt_kwargs_f.keys():
			v = self.opt_kwargs_f[k](self.epoch_counter)
			self.opt_kwargs[k] = v

	def generate_mounted_optimizer(self):
		self.optimizer = self.opt_class(self.get_model_parameters(), **self.opt_kwargs)

	def __len__(self):
		return sum(p.numel() for p in self.get_model_parameters() if p.requires_grad)
		
	def get_device(self):
		return next(self.get_model_parameters()).device

	def zero_grad(self,
		set_to_none=False,
		):
		return self.zero_grad_model(
		# return self.zero_grad_grads(
			set_to_none=set_to_none,
			)

	def zero_grad_grads(self,
		set_to_none=False,
		):
		for group in self.optimizer.param_groups:
			for p in group['params']:
				if p.grad is not None:
					if set_to_none:
						p.grad = None
					else:
						if p.grad.grad_fn is not None:
							p.grad.detach_()
						else:
							p.grad.requires_grad_(False)
						p.grad.zero_()

	def zero_grad_model(self,
		set_to_none=False,
		):
		for param in self.get_model_parameters():
			param.grad = None if set_to_none else 0

	def step(self):
		self.optimizer.step()

	def get_opt_kwargs(self):
		return list(self.opt_kwargs.keys())

	def update(self):
		self.epoch_counter += 1
		self.calcule_opt_kwargs()
		self.update_opt_kwargs()

	def update_opt_kwargs(self):
		for param_group in self.optimizer.param_groups:
			# print('keys',param_group.keys(), self.opt_kwargs.keys())
			for k in self.opt_kwargs.keys():
				param_group[k] = self.opt_kwargs[k]

	def get_kwarg_value(self, key):
		for param_group in self.optimizer.param_groups:
			return param_group[key]

	def get_info(self):
		opt_kwargs = self.get_opt_kwargs()
		d = {opt_kwarg:self.get_kwarg_value(opt_kwarg) for opt_kwarg in opt_kwargs}
		return d
