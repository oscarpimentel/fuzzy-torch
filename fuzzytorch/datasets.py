from __future__ import print_function
from __future__ import division
from . import C_

import torch
from torch.utils.data._utils.collate import default_collate

###################################################################################################################################################

def dict_to_tensor_dict(d):
	all_tensors = all([isinstance(d[key], torch.Tensor) for key in d.keys()])
	if all_tensors:
		return TensorDict(d)
	else:
		return TensorDict({key:dict_to_tensor_dict(d[key]) for key in d.keys()})

def tensor_data_collate(batch):
	d = default_collate([b.get_dict() for b in batch])
	return dict_to_tensor_dict(d)

class TensorDict():
	def __init__(self, d):
		assert isinstance(d, dict)
		assert all([isinstance(d[key], torch.Tensor) for key in d.keys()]) or all([isinstance(d[key], TensorDict) for key in d.keys()])
		assert all([not key=='d' for key in d.keys()])
		self.d = d

	def to(self, device):
		for key in self.keys():
			x = self.d[key]
			x.to(device)

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