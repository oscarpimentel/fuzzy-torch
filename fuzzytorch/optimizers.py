from __future__ import print_function
from __future__ import division
from . import C_

import torch.nn as nn
import pandas as pd

###################################################################################################################################################

class LossOptimizer:
	def __init__(self, to_optimize_model, opt_class,
		opt_kwargs:dict={},
		decay_epochs_delta:int=1,
		decay_kwargs:dict={},
		clip_grad:float=None,
		model_get_parameters_f=None,
		):
		self.to_optimize_model = to_optimize_model
		self.opt_class = opt_class
		self.opt_kwargs = opt_kwargs
		self.decay_epochs_delta = decay_epochs_delta
		self.decay_kwargs = decay_kwargs
		self.clip_grad = clip_grad
		self.model_get_parameters_f = model_get_parameters_f
		self.epoch_counter = 0
		self.generate_mounted_optimizer()

	def generate_mounted_optimizer(self):
		assert isinstance(self.to_optimize_model, nn.Module)
		self.optimizer = self.opt_class(self.get_model_parameters(), **self.opt_kwargs)

	def get_model_parameters(self):
		return self.to_optimize_model.parameters() if self.model_get_parameters_f is None else getattr(self.to_optimize_model, self.model_get_parameters_f)()

	def __len__(self):
		return sum(p.numel() for p in self.get_model_parameters() if p.requires_grad)

	def train(self):
		self.to_optimize_model.train()

	def eval(self):
		self.to_optimize_model.eval()
		
	def device(self):
		return next(self.get_model_parameters()).device

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

	def get_info_df(self):
		opt_kwargs = self.get_opt_kwargs()
		values = [self.get_kwarg_value(opt_kwarg) for opt_kwarg in opt_kwargs]
		df = pd.DataFrame([values], columns=opt_kwargs)
		return df
