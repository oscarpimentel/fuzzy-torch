from __future__ import print_function
from __future__ import division
from . import C_

import os
import torch.nn as nn
import numpy as np
from . import losses
from . import metrics
from . import optimizers
from .models import count_parameters
from . import exceptions as exc
from ..myUtils import files

###################################################################################################################################################

class NewTrainHandler(object):
	def __init__(self, name:str, model, optimizer, loss_crit, metric_crits,
		early_stop_epochcheck_epochs:int=1,
		early_stop_patience_epochchecks:int=1000,
		save_mode:str=C_.SM_NO_SAVE,
		target_metric_crit:str=None,
		**kwargs):
		assert isinstance(model, nn.Module)
		assert isinstance(optimizer, optimizers.NewOptimizer)
		assert isinstance(loss_crit, losses.NewLossCrit)
		assert isinstance(metric_crits, list) or isinstance(metric_crits, metrics.NewMetricCrit)
		assert early_stop_epochcheck_epochs>=1
		assert early_stop_patience_epochchecks>=2

		### ATTRIBUTES
		self.name = name
		self.model = model
		self.optimizer = optimizer
		self.loss_crit = loss_crit
		self.metric_crits = metric_crits
		self.early_stop_epochcheck_epochs = early_stop_epochcheck_epochs
		self.early_stop_patience_epochchecks = early_stop_patience_epochchecks
		self.save_mode = save_mode
		self.target_metric_crit = target_metric_crit

		self.epoch_counter = 0
		self.epochcheck_counter = 0
		self.uses_early_stop = self.early_stop_patience_epochchecks > 0
		self.history_dict = {
			'ks_epochs':None,
			'early_stop_epochcheck_epochs':self.early_stop_epochcheck_epochs,
			'early_stop_patience_epochchecks':self.early_stop_patience_epochchecks,
			'save_mode':self.save_mode,
		}

		### LOSSES K
		self.history_dict['finalloss_evolution_k'] = {
			'train':[],
		}
		self.history_dict['sublosses_evolution_k'] = {
			'train':{},
		}
		### LOSSES EPOCHCHECK
		self.history_dict['finalloss_evolution_epochcheck'] = {
			'train':[],
			'val':[],
		}
		self.history_dict['sublosses_evolution_epochcheck'] = {
			'train':{},
			'val':{},
		}
		self.set_metrics_hist()
		self.set_opt_hist()

		### TIMES AND CONV
		self.history_dict['mins_evolution_epoch'] = {
			'train':[],
		}
		self.history_dict['global_mins_evolution_epoch'] = {
			'train':[],
		}
		self.history_dict['best_epoch'] = 0
		self.last_saved_filedir = None

	def set_opt_hist(self):
			self.history_dict['opt_kwargs_evolution_epoch'] = {opt_key:[] for opt_key in self.optimizer.get_opt_kwargs()}

	def get_sublosses_names(self):
		return list(self.history_dict['sublosses_evolution_k']['train'].keys())

	def add_subloss_history(self, dict_name, set_name, subloss_name, subloss_value):
		if not subloss_name in self.history_dict[dict_name][set_name].keys():
			self.history_dict[dict_name][set_name][subloss_name] = []
		self.history_dict[dict_name][set_name][subloss_name].append(subloss_value)

	def need_to_save(self):
		return not self.save_mode==C_.SM_NO_SAVE

	def print_info(self):
		print(f'[th:{self.name}]')
		print(f'\topt parameters: {self.optimizer.get_count_parameters():,}[p]')
		metric_info = f'(target_metric_crit: {self.target_metric_crit})' if self.save_mode in [C_.SM_ONLY_INF_METRIC, C_.SM_ONLY_SUP_METRIC] else ''
		print(f'\tsave mode: {self.save_mode}{metric_info} - device: {next(self.model.parameters()).device}')
		print(f'\tearly_stop_epochcheck_epochs: {self.early_stop_epochcheck_epochs} - early_stop_patience_epochchecks: {self.early_stop_patience_epochchecks}')

	def set_last_saved_filedir(self, last_saved_filedir):
		self.last_saved_filedir = last_saved_filedir

	def set_metrics_hist(self):
		if isinstance(self.metric_crits, metrics.NewMetricCrit): # transform to list
			self.metric_crits = [self.metric_crits]

		self.metric_crits_names = [m.name for m in self.metric_crits]
		self.history_dict['metrics_evolution_epochcheck'] = {
			'train':{mn:[] for mn in self.metric_crits_names},
			'val':{mn:[] for mn in self.metric_crits_names},
		}
		self.target_metric_crit = self.metric_crits_names[0] if self.target_metric_crit is None and len(self.metric_crits)>0 else self.target_metric_crit # by default

	def nan_check(self, loss):
		if np.any(np.isnan(loss)) or np.any(loss==np.infty) or np.any(loss==-np.infty):
			raise exc.NanLossException()

	def early_stop_check(self):
		if self.epochcheck_counter >= self.early_stop_patience_epochchecks:
			raise exc.TrainingStopException()

	def remove_filedir(self, filedir):
		files.delete_filedir(filedir, verbose=0) # remove last best model

	def reset_early_stop(self):
		#print('reset')
		self.epochcheck_counter = 0

	def can_be_evaluated(self):
		return self.epoch_counter >= self.early_stop_epochcheck_epochs

	def evaluated(self):
		self.epoch_counter = 0
		if not self.save_mode==C_.SM_NO_SAVE:
			self.epochcheck_counter += 1

	def epoch_update(self):
		self.optimizer.epoch_update()
		self.epoch_counter += 1
		#print('epoch_counter',self.epoch_counter)

	def mount_model(self, gpu):
		self.model.to(gpu) # model to GPU
		self.optimizer.generate_mounted_optimizer(self.model)

	def get_mins_per_epoch(self):
		mins_per_epoch = self.history_dict['mins_evolution_epoch']['train']
		return np.array(mins_per_epoch).mean()

	def get_mins_to_convergence(self):
		mins_per_epoch = self.history_dict['mins_evolution_epoch']['train']
		best_epoch = self.get_best_epoch()
		return np.array(mins_per_epoch)[:best_epoch].sum() # in mins

	def get_best_epoch(self):
		return self.history_dict['best_epoch']

	def set_best_epoch(self, best_epoch):
		self.history_dict['best_epoch'] = best_epoch

	def check_save_condition(self, set_name):
		if self.save_mode==C_.SM_NO_SAVE:
			return False

		elif self.save_mode==C_.SM_ALL:
			return True

		elif self.save_mode==C_.SM_ONLY_ALL:
			self.remove_filedir(self.last_saved_filedir) # remove last best model
			return True

		elif self.save_mode==C_.SM_ONLY_INF_LOSS:
			loss_evolution = np.array(self.history_dict['finalloss_evolution_epochcheck'][set_name])
			if len(loss_evolution)<=1:
				return True # always save first and dont delete anything

			actual_loss = loss_evolution[-1]
			loss_history = loss_evolution[:-1]
			#print(actual_loss<np.min(loss_history), actual_loss, loss_history,np.min(loss_history))
			if actual_loss<np.min(loss_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				return True
			return False

		elif self.save_mode==C_.SM_ONLY_INF_METRIC:
			metric_evolution = np.array(self.history_dict['metrics_evolution_epochcheck'][set_name][self.target_metric_crit])
			if len(metric_evolution)<=1:
				return True # always save first and dont delete anything

			actual_metric_val = metric_evolution[-1]
			metric_history = metric_evolution[:-1]

			if actual_metric_val<np.min(metric_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				return True
			return False
		
		elif self.save_mode==C_.SM_ONLY_SUP_METRIC:
			metric_evolution = np.array(self.history_dict['metrics_evolution_epochcheck'][set_name][self.target_metric_crit])
			if len(metric_evolution)<=1:
				return True # always save first and dont delete anything

			actual_metric_val = metric_evolution[-1]
			metric_history = metric_evolution[:-1]

			if actual_metric_val>np.max(metric_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				return True
			return False

		else:
			raise Exception(f'save mode {self.save_mode} not supported')