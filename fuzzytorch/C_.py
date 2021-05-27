import sys
import numpy as np
from fuzzytools import C_ as FCC_
import torch

###################################################################################################################################################

EPS = 1e-5

DEFAULT_PADDING_MODE = 'same'
DEFAULT_ACTIVATION = 'relu'
DEFAULT_LAST_ACTIVATION = 'linear'

DEFAULT_CNN_KWARGS = {
	'kernel_size':3,
	'stride':1,
	'dilation':1,
}
DEFAULT_POOL_KWARGS = {
	'kernel_size':2,
	'stride':2,
	'dilation':1,
}

K_COUNTER_DURATION = 0 # 0
VAL_EPOCH_COUNTER_DURATION = 1
EARLYSTOP_EPOCH_DURATION = 21

SM_NO_SAVE = 'no_save'
SM_ALL = 'all'
SM_ONLY_ALL = 'only_all'
SM_ONLY_INF_LOSS = 'only_inf_loss'
SM_ONLY_INF_METRIC = 'only_inf_metric'
SM_ONLY_SUP_METRIC = 'only_sup_metric'

SAVE_FEXT = 'tfes'
KEY_KEY_SEP_CHAR = FCC_.KEY_KEY_SEP_CHAR
KEY_VALUE_SEP_CHAR = FCC_.KEY_VALUE_SEP_CHAR

DUMMY_TEXT = 'Dummy'

PLOT_FIGSIZE = (13,4)
PLOT_GRID_ALPHA = 0.25
PLOT_DPI = 80
C_MAIN_LOSS = 'k'

numpy_to_torch_dtype_dict = {
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

torch_to_numpy_dtype_dict = {
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