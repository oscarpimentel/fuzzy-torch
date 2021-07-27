from __future__ import print_function
from __future__ import division
from . import _C

import torch
import fuzzytools.strings as strings
import numpy as np

NUMPY_TO_TORCH_DTYPE_DICT = {
	np.bool       : torch.bool,
	np.uint8      : torch.uint8,
	np.int8       : torch.int8,
	np.int16      : torch.int16,
	np.int32      : torch.int32,
	np.int64      : torch.int64,
	np.float16    : torch.float16,
	np.float32    : torch.float32,
	np.float64    : torch.float64,
	np.complex64  : torch.complex64,
	np.complex128 : torch.complex128
	}
TORCH_TO_NUMPY_DTYPE_DICT = {
	torch.bool       : np.bool,
	torch.uint8      : np.uint8,
	torch.int8       : np.int8,
	torch.int16      : np.int16,
	torch.int32      : np.int32,
	torch.int64      : np.int64,
	torch.float16    : np.float16,
	torch.float32    : np.float32,
	torch.float64    : np.float64,
	torch.complex64  : np.complex64,
	torch.complex128 : np.complex128
	}

###################################################################################################################################################

def get_numpy_dtype(torch_dtype):
	return TORCH_TO_NUMPY_DTYPE_DICT[torch_dtype]

def tensor_to_numpy(x):
	return x.detach().cpu().numpy()

def get_model_name(model_name_dict):
	return strings.get_string_from_dict(model_name_dict)

def nested_tdict_to_device(d, device):
	if isinstance(d, dict):
		return {k:nested_tdict_to_device(d[k], device) for k in d.keys()}
	elif isinstance(d, torch.Tensor):
		return d if d.device==device else d.to(device)
	else:
		raise Exception(f'not supported {type(d)}')

def print_tdict(d):
	def get_tdict_repr(d):
		if isinstance(d, dict):
			return '{'+', '.join([f'{k}: {get_tdict_repr(d[k])}' for k in d.keys()])+'}'
		elif isinstance(d, torch.Tensor):
			x = d
			shape_txt = '' if len(x.shape)==0 else ', '.join([str(i) for i in x.shape])
			return f'({shape_txt})-{str(x.dtype)[6:]}-{x.device}'
		else:
			return ''
	print(get_tdict_repr(d))

def create_d(links):
	#print(links)
	tree = {}
	for path in links:                # for each path
		node = tree                   # start from the very top
		for level in path.split('/'): # split the path into a list
			if level:                 # if a name is non-empty
				node = node.setdefault(level, dict())
									  # move to the deeper level
									  # (or create it if unexistent)
	return tree

def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def iter_paths(d):
	def iter1(d, path):
		paths = []
		for k, v in d.items():
			if isinstance(v, dict):
				paths += iter1(v, path + [k])
			paths.append((path + [k], v))
		return paths
	return iter1(d, [])

def minibatch_dict_collate(batch_dict_list):
	dict_list = []
	for minibatch_dict_list in batch_dict_list:
		dpaths = iter_paths(minibatch_dict_list)
		batch_size = len(dpaths[0][1])
		for i in range(batch_size):
			d = create_d(['/'.join(dp) for dp,v in dpaths])
			for dp,v in dpaths:
				if isinstance(v, torch.Tensor):
					#print(v_.shape)
					nested_set(d, dp, v[i])
			#print_tdict(d)
			dict_list.append(d)

	new_d = torch.utils.data._utils.collate.default_collate(dict_list)
	#print_tdict(new_d)
	return new_d

###################################################################################################################################################

class TDictHolder():
	def __init__(self, d):
		assert isinstance(d, dict)
		self.d = d

	def to(self, device,
		add_dummy_dim=False,
		):
		d = nested_tdict_to_device(self.d, device)
		return torch.utils.data._utils.collate.default_collate([d]) if add_dummy_dim else d

	def __getitem__(self, key):
		return self.d[key]

	def __repr__(self):
		return get_tdict_repr(self.d)

	def keys(self):
		return self.d.keys()

	def get_tdict(self):
		return self.d