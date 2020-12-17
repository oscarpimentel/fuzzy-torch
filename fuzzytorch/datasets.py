from __future__ import print_function
from __future__ import division
from . import C_

import torch
from torch.utils.data._utils.collate import default_collate
from . utils import TensorDict

###################################################################################################################################################

def dict_to_tensor_dict(d):
	all_tensors = all([isinstance(d[key], torch.Tensor) for key in d.keys()])
	if all_tensors:
		return TensorDict(d)
	else:
		return TensorDict({key:dict_to_tensor_dict(d[key]) for key in d.keys()})

def tensor_data_collate(batch):
	batch_dicts = [b.get_dict() for b in batch]
	return dict_to_tensor_dict(default_collate(batch_dicts))