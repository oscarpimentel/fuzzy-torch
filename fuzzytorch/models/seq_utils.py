from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .basics import MLP, Linear
from fuzzytools import strings as strings
from . import utils

PADDING_VALUE = 0
EMPTY_SEQ_VALUE = PADDING_VALUE
EPS = _C.EPS
INF = float('inf')

###################################################################################################################################################

class LinearSEFT(nn.Module):
    def __init__(self, input_dims,
        in_dropout=0.,
        out_dropout=0.,
        dummy=False,
        dummy_method='last',
        **kwargs):
        super().__init__()
        ### CHECKS
        assert in_dropout>=0 and in_dropout<=1
        assert out_dropout>=0 and out_dropout<=1

        self.input_dims = input_dims
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self.dummy = dummy
        self.dummy_method = dummy_method
        self.reset()

    def reset(self):
        if not self.is_dummy():
            self.h = Linear(self.input_dims, self.input_dims,
                in_dropout=self.in_dropout,
                activation='linear',
                bias=True,
                )
            self.g = Linear(self.input_dims, self.input_dims,
                out_dropout=self.out_dropout,
                activation='linear',
                bias=False,
                )

    def is_dummy(self):
        return self.dummy

    def __len__(self):
        return utils.get_nof_parameters(self)

    def extra_repr(self):
        txt = strings.get_string_from_dict({
            'input_dims':self.input_dims,
            'in_dropout':self.in_dropout,
            'out_dropout':self.out_dropout,
            'dummy':self.dummy,
            'dummy_method':self.dummy_method,
            }, ', ', '=')
        return txt

    def __repr__(self):
        txt = f'LinearSEFT({self.extra_repr()})'
        txt += f'({len(self):,}[p])'
        return txt

    def forward(self, x, onehot,
        **kwargs):
        # x (n,t,f)
        # onehot (n,t)
        assert len(x.shape)==3
        assert len(onehot.shape)==2

        if self.is_dummy():
            # print(onehot[0,:])
            # print(x[0,:,:2])
            if self.dummy_method=='last':
                return seq_last_element(x, onehot) # (n,t,f)>(n,f)
            elif self.dummy_method=='avg':
                return seq_avg_pooling(x, onehot) # (n,t,f)>(n,f)
            else:
                raise Exception(f'dummy_method={dummy_method}')
        else:
            hx = seq_avg_pooling(torch.relu(self.h(x)), onehot) # (n,t,f)>(n,f)
            gx = self.g(hx) # (n,f)>(n,f)
            return gx

###################################################################################################################################################

def _check(x, onehot):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    assert onehot.dtype==torch.bool
    assert len(onehot.shape)==2
    assert x.shape[:-1]==onehot.shape
    assert len(x.shape)==3
    n,t,f = x.size()
    return n,t,f

###################################################################################################################################################

def get_dummy_onehot(x):
    n,t,f = x.size()
    onehot = torch.ones((n,t), device=x.device, dtype=bool)
    return onehot

def get_dummy_not_missing_mask(x):
    assert 0

def get_seq_onehot_mask(seqlengths, max_seqlength,
    device=None,
    ):
    assert len(seqlengths.shape)==1
    assert seqlengths.dtype==torch.long

    batch_size = len(seqlengths)
    mask = torch.arange(max_seqlength, device=seqlengths.device if device is None else device).expand(batch_size, max_seqlength)
    mask = (mask < seqlengths[...,None])
    return mask.bool() # (n,t)

###################################################################################################################################################

def seq_clean(x, onehot,
    padding_value=PADDING_VALUE,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    b,t,f = _check(x, onehot)
    if padding_value is None:
        return x
    else:
        new_onehot = utils.get_onehot_clone(onehot)
        x = x.masked_fill(~new_onehot[...,None], padding_value) # clean using onehot
    return x

###################################################################################################################################################

def seq_fill_missing(raw_x, not_missing_mask, onehot,
    nan_value=0,
    ):
    '''
    x (n,t,f)
    not_missing_mask (n,t,f)
    onehot (n,t)
    '''
    x = raw_x.clone()
    x[torch.isnan(x)] = nan_value
    assert not_missing_mask.dtype==torch.bool
    b,t,f = _check(x, onehot)
    b,t,f = _check(not_missing_mask, onehot)

    new_x = torch.zeros_like(x)
    x_last = x[:,0,:] # (n,f)
    for i in range(0, t): # sadly...we need a for
        #print('i',i)
        miss = (1-not_missing_mask[:,i,:].int()) # (n,f)
        x_actual = x[:,i,:]
        _x = x_actual*(1-miss)+x_last*(miss)
        new_x[:,i,:] = _x
        x_last = x_actual*(1-miss)+x_last*(miss)
    return new_x

def seq_dtimes(times, not_missing_mask, onehot):
    '''
    times (n,t,f)
    not_missing_mask (n,t,f)
    onehot (n,t)
    '''
    assert not_missing_mask.dtype==torch.bool
    b,t,_ = _check(times[...,None], onehot)
    b,t,f = _check(not_missing_mask, onehot)

    ftimes = times[...,None].repeat(1,1,f) # (n,t)>(n,t,f)
    cache_ftimes = seq_fill_missing(ftimes, not_missing_mask, onehot, nan_value=0) # (n,t,f)
    #print(cache_ftimes.shape, cache_ftimes[0].permute(1,0))
    new_cache_ftimes = torch.cat([cache_ftimes[:,0,:][:,None,:], cache_ftimes], dim=1) # (n,t,f)>(n,t+1,f)
    #print(new_cache_ftimes.shape, new_cache_ftimes[0].permute(1,0))
    dtimes = ftimes-new_cache_ftimes[:,:-1,:] # (n,t+1,f)>(n,t,f)
    return dtimes

###################################################################################################################################################

def _seq_dtimes(times, onehot):
    '''
    times (n,t)
    onehot (n,t)
    '''
    b,t,_ = _check(times[...,None], onehot)
    new_times = torch.cat([times[:,0][...,None], times], dim=1)
    dtimes = times-new_times[:,:-1]
    return dtimes

def seq_avg_pooling(x, onehot,
    empty_seq_value=EMPTY_SEQ_VALUE,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    n,t,f = _check(x, onehot)
    new_onehot = utils.get_onehot_clone(onehot)
    new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
    x = seq_clean(x, onehot, empty_seq_value) # important
    x = seq_clean(x, new_onehot, 0) # important
    x = x.sum(dim=1)/(new_onehot.sum(dim=1)[...,None]) # (n,t,f)>(n,f)
    return x

def seq_sum_pooling(x, onehot,
    empty_seq_value=EMPTY_SEQ_VALUE,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    n,t,f = _check(x, onehot)
    new_onehot = utils.get_onehot_clone(onehot)
    new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
    x = seq_clean(x, onehot, empty_seq_value) # important
    x = seq_clean(x, new_onehot, 0) # important
    x = x.sum(dim=1) # (n,t,f) > (n,f)
    return x

def seq_last_element(x, onehot,
    empty_seq_value=EMPTY_SEQ_VALUE,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    n,t,f = _check(x, onehot)
    new_onehot = utils.get_onehot_clone(onehot)
    new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
    x = seq_clean(x, onehot, empty_seq_value) # important
    x = seq_clean(x, new_onehot, 0) # important
    indexs = torch.sum(onehot[...,None], dim=1)-1 # (n,t,1)>(n,1) # -1 because index is always 1 unit less than length
    indexs = torch.clamp(indexs, 0, None) # force -1=0 to avoid errors of empty sequences!!
    last_x = torch.gather(x, 1, indexs[:,:,None].expand(-1,-1,f)) # index (n,t,f)>(n,1,f)
    last_x = last_x[:,0,:] # (n,1,f)>(n,f)
    return last_x

def seq_min_pooling(x, onehot,
    empty_seq_value=EMPTY_SEQ_VALUE,
    inf=INF,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    n,t,f = _check(x, onehot)
    new_onehot = utils.get_onehot_clone(onehot)
    new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
    x = seq_clean(x, onehot, empty_seq_value) # important
    x = x.masked_fill(~new_onehot[...,None], inf) # inf imputation using onehot
    x,_ = torch.min(x, dim=1)
    return x

def seq_max_pooling(x, onehot,
    empty_seq_value=EMPTY_SEQ_VALUE,
    inf=INF,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    n,t,f = _check(x, onehot)
    new_onehot = utils.get_onehot_clone(onehot)
    new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
    x = seq_clean(x, onehot, empty_seq_value) # important
    x = x.masked_fill(~new_onehot[...,None], -inf) # inf imputation using onehot
    x,_ = torch.max(x, dim=1)
    return x

def seq_min_max_norm(x, onehot,
    padding_value=PADDING_VALUE,
    zero_diff_value=1,
    eps=EPS,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    assert zero_diff_value>=0 and zero_diff_value<=1
    n,t,f = _check(x, onehot)

    _min = seq_min_pooling(x, onehot)[:,None,:] # (n,f)>(n,1,f)
    _max = seq_max_pooling(x, onehot)[:,None,:] # (n,f)>(n,1,f)
    diff = _max-_min
    new_x = (x-_min)/(diff+eps)
    zero_diff = (diff==0).repeat((1,t,1))
    new_x = new_x.masked_fill(zero_diff, zero_diff_value) # inf imputation  using onehot
    new_onehot = utils.get_onehot_clone(onehot)
    new_onehot[:,0] = True # forced true to avoid errors of empty sequences!!
    new_x = new_x.masked_fill(~new_onehot[...,None], padding_value) # inf imputation using onehot
    return new_x

###################################################################################################################################################

def seq_avg_norm(x, onehot, # FIXME
    padding_value=PADDING_VALUE,
    zero_diff_value=1,
    eps=EPS,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    n,t,f = _check(x, onehot)
    assert torch.all(x>=0)

    _avg = seq_avg_pooling(x, onehot)[:,None,:] # (n,f)>(n,1,f)
    return x/(_avg+eps)

def seq_sum_norm(x, onehot, # FIXME
    padding_value=PADDING_VALUE,
    zero_diff_value=1,
    eps=EPS,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    n,t,f = _check(x, onehot)
    assert torch.all(x>=0)

    _sum = seq_sum_pooling(x, onehot)[:,None,:] # (n,f)>(n,1,f)
    return x/(_sum+eps)

###################################################################################################################################################

def seq_index_mapping_(source, idxs, output,
    dim=1,
    ):
    assert source.dtype==output.dtype
    assert source.shape[0]==output.shape[0]
    assert source.shape[1]==output.shape[1]-1
    assert source.shape[2]==output.shape[2]
    assert len(source.shape)==3
    assert len(idxs.shape)==2
    assert idxs.shape==source.shape[:-1]

    fixed_indexs = idxs.detach().clone()
    IMD = source.shape[1]
    fixed_indexs[fixed_indexs==IMD] = output.shape[dim]-1
    fixed_indexs = fixed_indexs.unsqueeze(-1).expand(-1,-1,source.shape[-1])
    #print(output.device, fixed_indexs.device, source.device)
    output.scatter_(dim, fixed_indexs, source)

def serial_to_parallel(x, onehot,
    padding_value=PADDING_VALUE,
    ):
    '''
    x (n,t,f)
    onehot (n,t)
    '''
    _check(x, onehot)

    IMD = onehot.shape[1]
    s2p_mapping_indexs = (torch.cumsum(onehot, 1)-1).masked_fill(~onehot, IMD)
    #print('s2p_mapping_indexs', s2p_mapping_indexs.shape, s2p_mapping_indexs)
    new_shape = (x.shape[0], x.shape[1]+1, x.shape[2])
    new_x = torch.full(new_shape, padding_value, device=x.device, dtype=x.dtype)
    seq_index_mapping_(x, s2p_mapping_indexs, new_x)
    return new_x[:,:-1,:]

def parallel_to_serial(list_x, s_onehot,
    padding_value=PADDING_VALUE,
    ):
    '''
    list_x list[(n,t,f)]
    onehot (n,t,d)
    '''
    assert isinstance(list_x, list)
    assert len(list_x)>0
    assert s_onehot.dtype==torch.bool
    assert len(s_onehot.shape)==3
    for x in list_x:
        assert x.shape[:-1]==s_onehot.shape[:-1]
        assert len(x.shape)==3

    modes = s_onehot.shape[-1]
    x_ = list_x[0]
    new_shape = (x_.shape[0], x_.shape[1]+1, x_.shape[2])
    x_s = torch.full(new_shape, padding_value, device=x.device, dtype=x.dtype)
    for i in range(modes):
        x = list_x[i]
        onehot = s_onehot[...,i]

        IMD = onehot.shape[1]
        s2p_mapping_indexs = (torch.cumsum(onehot, dim=1)-1).masked_fill(~onehot, IMD)
        source = torch.cumsum(torch.ones_like(s2p_mapping_indexs, device=x.device, dtype=x.dtype)[...,None], dim=1)
        source = source-1 # important step to handle missing directions
        
        p2s_mapping_indexs = torch.full((x_.shape[0], x_.shape[1]+1, 1), IMD, device=x.device, dtype=x.dtype)
        #print(source.shape)
        #print(p2s_mapping_indexs.shape)
        seq_index_mapping_(source, s2p_mapping_indexs, p2s_mapping_indexs)
        #print(p2s_mapping_indexs.shape)
        #idxs = torch.nonzero(onehot)
        #print(idxs.shape, idxs)
        #assert 0
        #print(x_s.shape, onehot.shape, x.shape)
        seq_index_mapping_(x, p2s_mapping_indexs[:,:-1,0].long(), x_s)
        #p2s_mapping_indexs = torch.where(onehot)
        #print(p2s_mapping_indexs)
        #assert 0
        #print('p2s_mapping_indexs', p2s_mapping_indexs.shape, p2s_mapping_indexs)
        #index_mapping(x, p2s_mapping_indexs, x_s)
    return x_s[:,:-1,:]

def get_random_onehot(x, modes):
    '''
    x (n,t,f)
    '''
    assert len(x.shape)==3
    assert modes>=2

    shape = list(x.shape)[:-1]+[modes]
    r = np.random.uniform(0, modes, size=shape)
    r_max = r.max(axis=-1)[...,None]
    onehot = torch.as_tensor(r>=r_max).bool()
    return onehot


def get_seq_clipped_shape(x, new_len,
    padding_value=PADDING_VALUE,
    ):
    '''
    Used in dataset creation
    x (t,f)
    '''
    assert len(x.shape)==2
    if new_len is None:
        return x
    assert new_len>0

    t,f = x.size()
    if new_len<=t:
        return x[:new_len]
    else:
        new_x = torch.full(size=(new_len,f), fill_value=padding_value, device=x.device, dtype=x.dtype)
        new_x[:t] = x
        return new_x