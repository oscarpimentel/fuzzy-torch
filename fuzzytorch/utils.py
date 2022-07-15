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

###################################################################################################################################################

def minibatch_dict_collate(batch_dict_list):
    '''
    batch is first!
    '''
    dict_list = []
    for minibatch_dict in batch_dict_list:
        keys = list(minibatch_dict.keys())
        batch_size = len(minibatch_dict[keys[0]])
        for k in range(0, batch_size):
            d = {}
            for key in keys:
                assert len(minibatch_dict[key])==batch_size
                d[key] = minibatch_dict[key][k]
            dict_list += [d]
    new_d = torch.utils.data._utils.collate.default_collate(dict_list)
    return new_d

###################################################################################################################################################

def get_tdict_repr(tdict):
    if type(tdict)==dict:
        return '{'+', '.join([f'{k}: {get_tdict_repr(tdict[k])}' for k in tdict.keys()])+'}'
    elif type(tdict)==torch.Tensor:
        x = tdict
        shape_txt = '' if len(x.shape)==0 else ', '.join([str(i) for i in x.shape])
        return f'({shape_txt})-{str(x.dtype)[6:]}-{x.device}'
    else:
        return ''

def print_tdict(tdict):
    print(get_tdict_repr(tdict))

###################################################################################################################################################

class TDictHolder():
    def __init__(self, tdict):
        assert type(tdict)==dict
        self.tdict = tdict

    def to(self, device,
        add_dummy_dim=False,
        ):
        out_tdict = {key:self.tdict[key].to(device) for key in self.tdict.keys()}
        out_tdict = torch.utils.data._utils.collate.default_collate([out_tdict]) if add_dummy_dim else out_tdict
        return out_tdict

    def __getitem__(self, key):
        return self.tdict[key]

    def __repr__(self):
        return get_tdict_repr(self.tdict)

    def keys(self):
        return self.tdict.keys()

    def get_tdict(self):
        return self.tdict