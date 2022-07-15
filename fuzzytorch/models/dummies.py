from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################################################################################################

def dummy_f(x, *args, **kwargs):
    return x

class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for name, val in kwargs.items():
            setattr(self, name, val)

    def _reset_parameters(self):
        pass

    def forward(self, x, *args, **kwargs):
        return x

    def repr(self):
        return 'DummyModule'

    def __repr__(self):
        txt = _C.DUMMY_TEXT
        try:
            txt += f' output_dims={self.output_dims},'
        except:
            pass
        return f'{self.repr()}({txt[:-1]})'

    def get_output_dims(self):
        return self.output_dims