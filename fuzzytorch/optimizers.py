from __future__ import print_function
from __future__ import division
from . import C_

import torch.nn as nn

class NewOptimizer:
	def __init__(self, opt_class,
		opt_kwargs:dict={},
		decay_epochs_delta:int=1,
		decay_kwargs:dict={},
		clip_grad:float=None,
		model_get_parameters_f=None,
		):
		self.opt_class = opt_class
		self.opt_kwargs = opt_kwargs
		self.decay_epochs_delta = decay_epochs_delta
		self.decay_kwargs = decay_kwargs
		self.clip_grad = clip_grad
		self.model_get_parameters_f = model_get_parameters_f
		self.epoch_counter = 0

	def generate_mounted_optimizer(self, model):
		assert isinstance(model, nn.Module)
		self.model = model
		self.optimizer = self.opt_class(self.get_model_parameters(), **self.opt_kwargs)

	def get_model_parameters(self):
		return self.model.parameters() if self.model_get_parameters_f is None else getattr(self.model, self.model_get_parameters_f)()

	def get_count_parameters(self):
		return sum(p.numel() for p in self.get_model_parameters() if p.requires_grad)

	def zero_grad(self):
		self.optimizer.zero_grad()

	def apply_clip_grad(self):
		if not self.clip_grad is None:
			torch.nn.utils.clip_grad_norm_(self.get_model_parameters(), self.clip_grad)

	def step(self):
		self.apply_clip_grad()
		self.optimizer.step()

	def get_opt_kwargs(self):
		return list(self.opt_kwargs.keys())

	def get_decay_kwargs(self):
		return list(self.decay_kwargs.keys())

	def epoch_update(self):
		if len(self.get_decay_kwargs())>0:
			self.epoch_counter += 1
			if self.epoch_counter >= self.decay_epochs_delta:
				self.epoch_counter = 0
				for key in self.decay_kwargs.keys():
					self.optimizer.param_groups[0][key] *= self.decay_kwargs[key]

	def get_kwarg_value(self, key):
		return self.optimizer.param_groups[0][key]