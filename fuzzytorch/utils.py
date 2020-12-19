from __future__ import print_function
from __future__ import division
from . import C_

import torch
import flamingchoripan.strings as strings

###################################################################################################################################################

def get_model_name(model_name_dict):
	return strings.get_string_from_dict(model_name_dict)

def tdict_to_device(d, device):
	if isinstance(d, dict):
		for k in d.keys():
			x = d[k]
			tdict_to_device(x, device)
	elif isinstance(d, torch.Tensor):
		d.to(device)
	else:
		pass

def get_tdict_repr(d):
	if isinstance(d, dict):
		return '{'+', '.join([f'{k}: {get_tdict_repr(d[k])}' for k in d.keys()])+'}'
	elif isinstance(d, torch.Tensor):
		x = d
		shape_txt = '' if len(x.shape)==0 else ', '.join([str(i) for i in x.shape])
		return f'({shape_txt})-{str(x.dtype)[6:]}-{x.device}'
	else:
		return ''

def print_tdict(d):
	print(get_tdict_repr(d))

###################################################################################################################################################

class TDictHolder():
	def __init__(self, d):
		assert isinstance(d, dict)
		self.d = d

	def to(self, device):
		tdict_to_device(self.d, device)
		return self.d

	def __getitem__(self, key):
		return self.d[key]

	def __repr__(self):
		return get_tdict_repr(self.d)

	def keys(self):
		return self.d.keys()

	def get_tdict(self):
		return self.d