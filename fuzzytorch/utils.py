from __future__ import print_function
from __future__ import division
from . import C_

import torch
import flamingchoripan.strings as strings

###################################################################################################################################################

def get_model_name(model_name_dict):
	return strings.get_string_from_dict(model_name_dict)

###################################################################################################################################################

class TensorDict():
	def __init__(self, d):
		assert isinstance(d, dict)
		assert all([isinstance(d[key], torch.Tensor)  for key in d.keys()]) or all([isinstance(d[key], TensorDict) for key in d.keys()])
		assert all([not key=='d' for key in d.keys()])
		self.d = d

	def to(self, device):
		for key in self.d.keys():
			self.d[key] = self.d[key].to(device)
		return self

	def add(self, key, x):
		assert isinstance(x, torch.Tensor) or isinstance(x, TensorDict)
		self.d[key] = x

	def __getitem__(self, key):
		return self.d[key]

	def __repr__(self):
		txt = '{'
		for key in self.d.keys():
			x = self.d[key]
			txt += f"'{key}': "
			if isinstance(x, TensorDict):
				txt += str(x)
			else:
				shape = '' if len(x.shape)==0 else ', '.join([str(i) for i in x.shape])
				t = str(x.dtype)[6:]
				txt += f'({shape})-{t}-{x.device}'
			txt += ', '

		txt = txt[:-2]+'}'
		return txt

	def get_dict(self):
		d = {}
		for key in self.d.keys():
			x = self.d[key]
			d[key] = x if isinstance(x, torch.Tensor) else x.get_dict()
		return d