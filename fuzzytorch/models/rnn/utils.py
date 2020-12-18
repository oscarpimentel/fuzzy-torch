from __future__ import print_function
from __future__ import division
from . import C_

import torch
import torch.nn as nn
from .basics import MLRNN

###################################################################################################################################################

def get_rnn_cell_class(cell_name:str):
	if cell_name in ['GRU', 'LSTM', 'RNN']:
		return getattr(tfmodels, cell_name)

	elif cell_name=='PLSTM':
		return torch.nn.LSTMCell #PLSTM

	raise Exception(f'no RNN class with cell_name: {cell_name}')

def get_multilayer_rnn(rnn_cell_name:str, rnn_args, rnn_kwargs):
	if rnn_cell_name=='PLSTM':
		return PLSTM_stack(rnn_cell_name, input_dim, output_dims, layers, dropout, is_bi)

	elif rnn_cell_name=='BLSTM':
		return BN_LSTM_stack(rnn_cell_name, input_dim, output_dims, layers, dropout, is_bi, max_curve_length)

	elif rnn_cell_name in ['RNN', 'GRU', 'LSTM']:
		return MLRNN(rnn_cell_name, *rnn_args, **rnn_kwargs)
		
	else:
		raise Exception(f'No rnn_cell_name: {rnn_cell_name} supported yet')
