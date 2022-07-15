from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn

###################################################################################################################################################

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask