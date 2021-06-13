import sys
import numpy as np
from fuzzytools import C_ as FCC_
import torch

###################################################################################################################################################

EPS = 1e-5

DEFAULT_NON_LINEAR_ACTIVATION = 'relu'

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