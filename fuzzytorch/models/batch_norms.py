from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################################################################################################

class MaskedBatchNorm1d(nn.BatchNorm1d):
	"""
	Masked verstion of the 1D Batch normalization.
	
	Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
	
	Receives a N-dim tensor of sequence lengths per batch element
	along with the regular input for masking.
	
	Check pytorch's BatchNorm1d implementation for argument details.
	"""
	def __init__(self, num_features, eps=1e-5, momentum=0.1,
				 affine=True, track_running_stats=True):
		super().__init__(
			num_features,
			eps,
			momentum,
			affine,
			track_running_stats
		)

	def forward(self, x, onehot):
		return self.forward_(x, onehot)

	def forward_(self, inp, mask):
		inp = inp.permute(0,2,1)
		self._check_input_dim(inp)

		exponential_average_factor = 0.0
		
		# We transform the mask into a sort of P(inp) with equal probabilities
		# for all unmasked elements of the tensor, and 0 probability for masked
		# ones.
		n = mask.sum()
		mask = mask / n
		mask = mask.unsqueeze(1).expand(inp.shape)

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				self.num_batches_tracked += 1
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		# calculate running estimates
		if self.training and n > 1:
			# Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
			# variance, we do not need to make any tensor shape manipulation.
			# mean = E[X] is simply the sum-product of our "probability" mask with the input...
			mean = (mask * inp).sum([0, 2])
			# ...whereas Var(X) is directly derived from the above formulae
			# This should be numerically equivalent to the biased sample variance
			var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
			with torch.no_grad():
				self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
				# Update running_var with unbiased var
				self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
		else:
			mean = self.running_mean
			var = self.running_var

		inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
		if self.affine:
			inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

		inp = inp.permute(0,2,1)
		return inp