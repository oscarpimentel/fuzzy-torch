from __future__ import print_function
from __future__ import division
from . import _C

import torch.nn as nn
import pandas as pd
import torch

###################################################################################################################################################

def _check_model(to_optimize_models):
	for m in to_optimize_models:
		assert isinstance(m, nn.Module)

class LossOptimizer:
	def __init__(self, to_optimize_models, opt_class, opt_kwargs_f,
		clip_grad=None,
		**kwargs):
		self.to_optimize_models = to_optimize_models if isinstance(to_optimize_models, list) else [to_optimize_models]
		self.opt_class = opt_class
		self.opt_kwargs_f = opt_kwargs_f
		self.clip_grad = clip_grad
		self.reset()

	def reset(self):
		self.epoch_counter = 0
		_check_model(self.to_optimize_models)
		self.calcule_opt_kwargs()
		self.generate_mounted_optimizer()
		self.set_gradient_clip_hooks()

	def get_model_parameters(self):
		model_parameters = []
		for m in self.to_optimize_models:
			model_parameters += list(m.parameters()) # .parameters() is an iterable that exahuste
		return model_parameters

	def calcule_opt_kwargs(self):
		self.opt_kwargs = {}
		for k in self.opt_kwargs_f.keys():
			v = self.opt_kwargs_f[k](self.epoch_counter)
			self.opt_kwargs[k] = v

	def generate_mounted_optimizer(self):
		self.optimizer = self.opt_class(self.get_model_parameters(), **self.opt_kwargs)

	def set_gradient_clip_hooks(self):
		if not self.clip_grad is None:
			assert self.clip_grad>0
			for p in self.get_model_parameters():
				if p.requires_grad:
					p.register_hook(lambda grad:torch.clamp(grad, -self.clip_grad, self.clip_grad))

	def __len__(self):
		return sum(p.numel() for p in self.get_model_parameters() if p.requires_grad)
		
	def get_device(self):
		# device = next(self.get_model_parameters()).device
		device = self.get_model_parameters()[0].device
		return device

	def zero_grad(self,
		set_to_none=False,
		reset_model_grad=True,
		):
		if reset_model_grad:
			self.zero_grad_model(
				set_to_none=set_to_none,
				)
		else:
			return self.zero_grad_grads(
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
		for p in self.get_model_parameters():
			p.grad = None if set_to_none else 0

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
