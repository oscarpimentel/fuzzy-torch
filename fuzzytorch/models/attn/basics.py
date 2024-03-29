from __future__ import print_function
from __future__ import division
from . import _C

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import non_linear
from .. import utils
from . import utils as attn_utils
from ..basics import MLP, ResidualBlockHandler
from torch.nn.init import xavier_uniform_, constant_, eye_
from fuzzytools import strings as strings
from fuzzytools import lists as lists
from .pytorch_multihead_clone import MultiheadAttention
# from torch.nn import MultiheadAttention
from ..others import TimeFILM
from .. import seq_utils
import numpy as np
import fuzzytorch.models.rnn.basics as ft_rnn # sanity_check

MHSELFATTN_NORM_MODE = 'pre_norm' # none pre_norm post_norm (optional)
MLP_NORM_MODE = 'none' # none pre_norm post_norm (optional)
NUM_HEADS = 4
MLP_K = 1
REMOVES_TIME_OFFSET = False
PADDING_VALUE = None # None 0 (optional)

###################################################################################################################################################

class SelfAttnWrapper(nn.Module):
    def __init__(self, attn_module,
        uses_permutation=True,
        **kwargs):
        super().__init__()
        self.attn_module = attn_module
        self.uses_permutation = uses_permutation
        self.reset()

    def reset(self):
        if hasattr(self.attn_module, 'reset'):
            self.attn_module.reset()
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.attn_module, 'reset_parameters'):
            self.attn_module.reset_parameters()

    def __len__(self):
        return len(self.attn_module)

    def __repr__(self):
        return str(self.attn_module)

    def forward(self, x,
        **kwargs):
        '''
        wrapper for self-attention operation
        '''
        queries = x.permute(1,0,2) if self.uses_permutation else x # (n,t,f) > (t,n,f)
        keys = x.permute(1,0,2) if self.uses_permutation else x # (n,t,f) > (t,n,f)
        values = x.permute(1,0,2) if self.uses_permutation else x # (n,t,f) > (t,n,f)
        if kwargs.get('need_weights', True):
            contexts, scores = self.attn_module(queries, keys, values, **kwargs) # (t,n,f) > (t,n,f)
            contexts = contexts.permute(1,0,2) if self.uses_permutation else contexts # (t,n,f) > (n,t,f)
            return contexts, scores
        else:
            contexts = self.attn_module(queries, keys, values, **kwargs) # (t,n,f) > (t,n,f)
            contexts = contexts.permute(1,0,2) if self.uses_permutation else contexts # (t,n,f) > (n,t,f)
            return contexts

###################################################################################################################################################

class SelfAttn(nn.Module):
    def __init__(self, input_dims:int, output_dims:int,
        max_curve_length=None,
        num_heads=NUM_HEADS,
        in_dropout=0.0,
        out_dropout=0.0,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        bias=True,
        mlp_k=MLP_K,
        mhselfattn_norm_mode=MHSELFATTN_NORM_MODE,
        mlp_norm_mode=MLP_NORM_MODE,
        padding_value=PADDING_VALUE,
        **kwargs):
        super().__init__()
        ### CHECKS
        assert num_heads==0 or input_dims%num_heads==0
        assert in_dropout>=0 and in_dropout<=1
        assert out_dropout>=0 and out_dropout<=1
        assert attn_dropout>=0 and attn_dropout<=1
        assert mlp_dropout>=0 and mlp_dropout<=1
        assert residual_dropout>=0 and residual_dropout<=1

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.max_curve_length = max_curve_length
        self.num_heads = num_heads
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.residual_dropout = residual_dropout
        self.bias = bias
        self.mlp_k = mlp_k
        self.mhselfattn_norm_mode = mhselfattn_norm_mode
        self.mlp_norm_mode = mlp_norm_mode
        self.padding_value = padding_value
        self.reset()

    def reset(self):
        self.dummy = self.num_heads==0
        self.head_dim = 0 if self.is_dummy() else self.input_dims//self.num_heads
        self.in_dropout_f = nn.Dropout(self.in_dropout)
        self.out_dropout_f = nn.Dropout(self.out_dropout)
        self.bypass_mlp = self.mlp_k is None or self.mlp_k==0

        ### attn
        if not self.is_dummy():
            mhattn = MultiheadAttention(self.input_dims, self.num_heads,
                dropout=self.attn_dropout,
                bias=self.bias,
                kdim=None,
                vdim=None,
                add_bias_kv=False,
                add_zero_attn=False,
                )
            self.self_mhattn = SelfAttnWrapper(mhattn)
            self.attn_res_block = ResidualBlockHandler(self.self_mhattn,
                torch.nn.LayerNorm([self.input_dims]),
                norm_mode=self.mhselfattn_norm_mode,
                residual_dropout=self.residual_dropout,
                )

        ### mlp
        if self.bypass_mlp:
            pass
        else:
            self.mlp = MLP(self.input_dims, self.output_dims, [int(self.input_dims*self.mlp_k)]*1,
                activation='relu', # transformer
                in_dropout=0.,
                out_dropout=0.,
                bias=self.bias,
                dropout=self.mlp_dropout,
                last_activation='linear', # transformer
                )
            self.mlp_res_block = ResidualBlockHandler(self.mlp,
                torch.nn.LayerNorm([self.input_dims]),
                norm_mode=self.mlp_norm_mode,
                residual_dropout=self.residual_dropout,
                )

    def is_dummy(self):
        return self.dummy

    def register_src_mask(self, max_curve_length, device):
        max_curve_length_changed = not max_curve_length==self.max_curve_length
        if max_curve_length_changed:
            self.max_curve_length = max_curve_length
            self.src_mask = attn_utils.generate_square_subsequent_mask(self.max_curve_length).to(device) # slow to use .to() but it's not always

    def get_output_dims(self):
        return self.output_dims

    def __len__(self):
        return utils.get_nof_parameters(self)
        
    def extra_repr(self):
        txt = strings.get_string_from_dict({
            'input_dims':self.input_dims,
            'output_dims':self.output_dims,
            'max_curve_length':self.max_curve_length,
            'num_heads':self.num_heads,
            'head_dim':self.head_dim,
            'in_dropout':self.in_dropout,
            'out_dropout':self.out_dropout,
            'attn_dropout':self.attn_dropout,
            'mlp_dropout':self.mlp_dropout,
            'residual_dropout':self.residual_dropout,
            'bias':self.bias,
            'mlp_k':self.mlp_k,
            'mhselfattn_norm_mode':self.mhselfattn_norm_mode,
            'mlp_norm_mode':self.mlp_norm_mode,
            'padding_value':self.padding_value,
            'mlp':self.mlp,
            }, ', ', '=')
        return txt

    def __repr__(self):
        txt = f'SelfAttn({self.extra_repr()})'
        txt += f'({len(self):,}[p])'
        return txt

    def forward(self, x, onehot,
        return_only_actual_scores=False,
        **kwargs):
        '''
        Parameters
        ----------
        x (n,t,f): input tensor.
        onehot (n,t)

        Return
        ----------
        x: (n,t,out): output tensor.
        scores: (n,h,t,qt)
        '''
        self.register_src_mask(x.shape[1], x.device) # create traingular mask
        new_onehot = utils.get_onehot_clone(onehot)
        new_onehot[:,0] = True # forced to avoid errors of empty bands sequences
        x = self.in_dropout_f(x) # (n,t,f)>(n,t,f)

        ### attn
        if self.is_dummy():
            n,h,t,qt = x.shape[0], 1, x.shape[1], x.shape[1]
            x, scores = x, torch.zeros(size=(n,h,t,qt), device=x.device) # (n,h,t,qt)
            scores = scores.detach()
        else:
            mhattn_kwargs = {
                'key_padding_mask':~new_onehot, # key_padding_mask ignore the True values
                'attn_mask':self.src_mask, # attn_mask will add
                }
            x, scores = self.attn_res_block(x, f_returns_tuple=True, f_kwargs=mhattn_kwargs) # (n,t,f)>(n,t,f)
            scores = scores.detach() # (n,h,t,qt) should sum 1 in dim qt
            # print(new_onehot[0,:])
            # for k in range(0, new_onehot.shape[-1]):
            #     print(scores[0,0,k,:])

        ### mlp
        if self.bypass_mlp:
            x = x
        else:
            x = self.mlp_res_block(x) # (n,t,f)>(n,t,f)

        ### dropout
        x = self.out_dropout_f(x) # (n,t,f)>(n,t,f)
        
        ### scores
        if return_only_actual_scores:
            scores_size = scores.size()
            if len(scores_size)==4: # from clone version
                n,h,t,qt = scores_size # (n,h,t,qt)
                scores = scores.permute(0,2,1,3) # (n,h,t,qt)>(n,t,h,qt)
                scores = scores.reshape(n,t,h*qt) # (n,t,h,qt)>(n,t,h*qt)
                scores = seq_utils.seq_last_element(scores, new_onehot) # last element (n,t,h*qt)>(n,h*qt)
                scores = scores.reshape(n,h,qt) # (n,h,qt) should sum 1 in dim qt
            else: # from source version
                pass # for now...

        x = seq_utils.seq_clean(x, new_onehot, padding_value=self.padding_value) # (n,t,f)>(n,t,f)
        return x, scores

###################################################################################################################################################

class MLSelfAttn(nn.Module):
    def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list,
        max_curve_length=None,
        num_heads=NUM_HEADS,
        in_dropout=0.0,
        dropout=0.0,
        out_dropout=0.0,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        bias=True,
        mlp_k=MLP_K,
        mhselfattn_norm_mode=MHSELFATTN_NORM_MODE,
        mlp_norm_mode=MLP_NORM_MODE,
        padding_value=PADDING_VALUE,
        **kwargs):
        super().__init__()

        ### CHECKS
        assert in_dropout>=0 and in_dropout<=1
        assert dropout>=0 and dropout<=1
        assert out_dropout>=0 and out_dropout<=1
        assert attn_dropout>=0 and attn_dropout<=1
        assert mlp_dropout>=0 and mlp_dropout<=1
        assert residual_dropout>=0 and residual_dropout<=1

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.embd_dims_list = [self.input_dims]+embd_dims_list+[self.output_dims]
        self.max_curve_length = max_curve_length
        self.num_heads = num_heads
        self.in_dropout = in_dropout
        self.dropout = dropout
        self.out_dropout = out_dropout
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.residual_dropout = residual_dropout
        self.bias = bias
        self.mlp_k = mlp_k
        self.mhselfattn_norm_mode = mhselfattn_norm_mode
        self.mlp_norm_mode = mlp_norm_mode
        self.padding_value = padding_value

        ### MODULES
        self.self_attns = nn.ModuleList()
        for k in range(0, len(self.embd_dims_list)-1):
            _input_dims = self.embd_dims_list[k]
            _output_dims = self.embd_dims_list[k+1]
            self_attn = SelfAttn(_input_dims, _output_dims,
                max_curve_length=self.max_curve_length,
                num_heads=self.num_heads,
                in_dropout=self.in_dropout if k==0 else self.dropout,
                out_dropout=self.out_dropout if k==len(self.embd_dims_list)-2 else 0.0,
                attn_dropout=self.attn_dropout,
                mlp_dropout=self.mlp_dropout,
                residual_dropout=self.residual_dropout,
                bias=self.bias,
                mlp_k=self.mlp_k,
                mhselfattn_norm_mode=self.mhselfattn_norm_mode,
                mlp_norm_mode=self.mlp_norm_mode,
                padding_value=self.padding_value,
                )
            self.self_attns += [self_attn]

        self.reset()

    def reset(self):
        for self_attn in self.self_attns:
            self_attn.reset()

    def get_embd_dims_list(self):
        return self.embd_dims_list

    def __len__(self):
        return utils.get_nof_parameters(self)

    def extra_repr(self):
        txt = ''
        for k,self_attn in enumerate(self.self_attns):
            txt += f'  ({k}) - {str(self_attn)}\n'
        return txt

    def __repr__(self):
        txt = f'MLSelfAttn({self.extra_repr()})'
        txt += f'({len(self):,}[p])'
        return txt

    def forward(self, x, onehot,
        return_only_actual_scores=False,
        **kwargs):
        '''
        Parameters
        ----------
        x (n,t,f): input tensor.
        onehot (n,t)

        Return
        ----------
        x: (n,t,f): output tensor.
        scores: (n,l,h,t,qt) or (n,l,h,qt)
        '''
        assert onehot.dtype==torch.bool
        assert len(onehot.shape)==2
        assert x.shape[:-1]==onehot.shape
        assert len(x.shape)==3

        scores = []
        for k,self_attn in enumerate(self.self_attns):
            x, _scores = self_attn(x, onehot, # (n,t,f)>(n,t,f)
                return_only_actual_scores=return_only_actual_scores,
                **kwargs)
            scores += [_scores[:,None,...]]
        scores = torch.cat(scores, dim=1)
        return x, scores

###################################################################################################################################################

class MLTimeSelfAttn(nn.Module):
    def __init__(self, input_dims:int, output_dims:int, embd_dims_list:list, te_features, max_period,
        kernel_size=1,
        time_noise_window=0,
        removes_time_offset=REMOVES_TIME_OFFSET,
        max_curve_length=None,
        num_heads=NUM_HEADS,
        in_dropout=0.0,
        dropout=0.0,
        out_dropout=0.0,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        bias=True,
        mlp_k=MLP_K,
        mhselfattn_norm_mode=MHSELFATTN_NORM_MODE,
        mlp_norm_mode=MLP_NORM_MODE,
        padding_value=PADDING_VALUE,
        **kwargs):
        super().__init__()

        ### CHECKS
        assert in_dropout>=0 and in_dropout<=1
        assert dropout>=0 and dropout<=1
        assert out_dropout>=0 and out_dropout<=1
        assert attn_dropout>=0 and attn_dropout<=1
        assert mlp_dropout>=0 and mlp_dropout<=1
        assert residual_dropout>=0 and residual_dropout<=1

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.embd_dims_list = embd_dims_list
        self.te_features = te_features
        self.max_period = max_period
        self.kernel_size = kernel_size
        self.time_noise_window = time_noise_window
        self.removes_time_offset = removes_time_offset
        self.max_curve_length = max_curve_length
        self.num_heads = num_heads
        self.in_dropout = in_dropout
        self.dropout = dropout
        self.out_dropout = out_dropout
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.residual_dropout = residual_dropout
        self.bias = bias
        self.mlp_k = mlp_k
        self.mhselfattn_norm_mode = mhselfattn_norm_mode
        self.mlp_norm_mode = mlp_norm_mode
        self.padding_value = padding_value

        ### MODULES
        self.time_film = TimeFILM(self.input_dims, self.te_features, self.max_period,
            kernel_size=self.kernel_size,
            time_noise_window=self.time_noise_window,
            removes_time_offset=self.removes_time_offset,
            )

        self.ml_self_attn = MLSelfAttn(input_dims, output_dims, embd_dims_list,
            max_curve_length=self.max_curve_length,
            num_heads=self.num_heads,
            in_dropout=self.in_dropout,
            dropout=self.dropout,
            out_dropout=self.out_dropout,
            attn_dropout=self.attn_dropout,
            mlp_dropout=self.mlp_dropout,
            residual_dropout=self.residual_dropout,
            bias=self.bias,
            mlp_k=self.mlp_k,
            mhselfattn_norm_mode=self.mhselfattn_norm_mode,
            mlp_norm_mode=self.mlp_norm_mode,
            padding_value=self.padding_value,
            **kwargs)

        self.reset()

    def reset(self):
        self.time_film.reset()
        self.ml_self_attn.reset()

    def get_embd_dims_list(self):
        return self.ml_self_attn.get_embd_dims_list()

    def __len__(self):
        return utils.get_nof_parameters(self)


    def extra_repr(self):
        txt = strings.get_string_from_dict({
            '\ntime_film':f'{self.time_film}\n',
            'ml_self_attn':f'{self.ml_self_attn}\n',
            }, ', ', '=')
        return txt

    def __repr__(self):
        txt = f'MLTimeModSelfAttn({self.extra_repr()})'
        txt += f'({len(self):,}[p])'
        return txt

    def get_info(self):
        return {
            'time_film':self.time_film.get_info(),
            }

    def forward(self, x, onehot, time,
        return_only_actual_scores=False,
        **kwargs):
        '''
        Parameters
        ----------
        x (n,t,f): input tensor.
        onehot (n,t)
        time (n,t)

        Return
        ----------
        x: (n,t,f): output tensor.
        scores: (n,l,h,t,qt) or (n,l,h,qt)
        '''
        assert onehot.dtype==torch.bool
        assert len(onehot.shape)==2
        assert x.shape[:-1]==onehot.shape
        assert len(x.shape)==3
        assert len(time.shape)==2

        x = self.time_film(x, time, onehot) # (n,t,f)>(n,t,f)
        x, scores = self.ml_self_attn(x, onehot, # (n,t,f)>(n,t,f)
            return_only_actual_scores=return_only_actual_scores,
            **kwargs)
        return x, scores