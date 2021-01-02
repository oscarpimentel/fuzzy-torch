from __future__ import print_function
from __future__ import division
from . import C_

import torch
import flamingchoripan.strings as strings
from torch.utils.data._utils.collate import default_collate

###################################################################################################################################################

def get_model_name(model_name_dict):
	return strings.get_string_from_dict(model_name_dict)

def tdict_to_device(d, device):
	if isinstance(d, dict):
		return {k:tdict_to_device(d[k], device) for k in d.keys()}
	elif isinstance(d, torch.Tensor):
		return d.to(device)
	else:
		raise Exception(f'not supported {type(d)}')

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

	def to(self, device,
		add_dummy_dim=False,
		):
		d = tdict_to_device(self.d, device)
		return default_collate([d]) if add_dummy_dim else d

	def __getitem__(self, key):
		return self.d[key]

	def __repr__(self):
		return get_tdict_repr(self.d)

	def keys(self):
		return self.d.keys()

	def get_tdict(self):
		return self.d