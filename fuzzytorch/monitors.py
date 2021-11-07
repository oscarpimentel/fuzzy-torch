from __future__ import print_function
from __future__ import division
from . import _C

import os
import torch.nn as nn
import numpy as np
from . import losses as ft_losses
from . import metrics as ft_metrics
from . import optimizers as ft_optimizers
from . import exceptions as ex
import fuzzytools.files as files
from fuzzytools.counters import Counter
from fuzzytools.datascience.xerror import XError
import pandas as pd
from fuzzytools.dataframes import DFBuilder
from copy import copy, deepcopy

###################################################################################################################################################

class LossMonitor(object):
	def __init__(self, loss, optimizer, metrics,
		save_mode:str=_C.SM_NO_SAVE,
		target_metric_crit:str=None,
		k_counter_duration:int=_C.K_COUNTER_DURATION,
		val_epoch_counter_duration:int=_C.VAL_EPOCH_COUNTER_DURATION,
		earlystop_epoch_duration:int=_C.EARLYSTOP_EPOCH_DURATION,
		**kwargs):

		### CHECKS
		assert isinstance(loss, ft_losses.FTLoss)
		metrics = [metrics] if isinstance(metrics, ft_metrics.FTMetric) else metrics
		assert isinstance(metrics, list) and all([isinstance(metric, ft_metrics.FTMetric) for metric in metrics])
		assert len([metric.name for metric in metrics])==len(set([metric.name for metric in metrics]))
		assert isinstance(optimizer, ft_optimizers.LossOptimizer)

		self.loss = loss
		self.optimizer = optimizer
		self.metrics = metrics
		self.save_mode = save_mode
		self.target_metric_crit = metrics[0].name if target_metric_crit is None else target_metric_crit
		self.counter_k = Counter({'k': k_counter_duration})
		self.counter_epoch = Counter({'val_epoch':val_epoch_counter_duration, 'earlystop_epoch':earlystop_epoch_duration})
		
		self.name = loss.name
		self.best_epoch = np.infty
		self.last_saved_filedir = None
		self.reset()

	def reset(self):
		self.best_value = None
		self.loss_df = DFBuilder()
		self.opt_df = DFBuilder()
		self.loss_df_epoch = DFBuilder()
		self.metrics_df_epoch = DFBuilder()
		self.counter_k.reset()
		self.counter_epoch.reset()

	### repr
	def __repr__(self):
		def get_metrics_repr():
			return f' (target_metric_crit={self.target_metric_crit})' if self.save_mode in [_C.SM_ONLY_INF_METRIC, _C.SM_ONLY_SUP_METRIC] else ''
		txt = ''
		txt += f'[{self.name}]'+'\n'
		txt += f' - opt-parameters={len(self.optimizer):,}[p] - device={self.optimizer.get_device()}'+'\n'
		txt += f' - save-mode={self.save_mode}{get_metrics_repr()}'+'\n'
		txt += f' - counter_k={self.counter_k} - counter_epoch={self.counter_epoch}'+'\n'
		return txt[:-1]

	def get_save_dict(self):
		info = {
			'save_mode':self.save_mode,
			'target_metric_crit':self.target_metric_crit,
			'counter_k':self.counter_k,
			'counter_epoch':self.counter_epoch,
			'best_epoch':self.best_epoch,
			'last_saved_filedir':self.last_saved_filedir,
			}
		d = {
			'info':info,
			'loss_df':self.loss_df,
			'opt_df':self.opt_df,
			'loss_df_epoch':self.loss_df_epoch,
			'metrics_df_epoch':self.metrics_df_epoch,
			}
		return d

	def load_from_dict(self, _d):
		d = deepcopy(_d)
		info = d['info']
		self.save_mode = info['save_mode']
		self.target_metric_crit = info['target_metric_crit']
		self.counter_k = info['counter_k']
		self.counter_epoch = info['counter_epoch']
		self.best_epoch = info['best_epoch']
		self.last_saved_filedir = info['last_saved_filedir']

		self.loss_df = d['loss_df']
		self.opt_df = d['opt_df']
		self.loss_df_epoch = d['loss_df_epoch']
		self.metrics_df_epoch = d['metrics_df_epoch']

	### history methods
	def add_loss_history_k(self, loss,
		dt=0,
		):
		if self.counter_k.check_counter_name_upper_bound('k'):
			assert isinstance(loss, ft_losses.BatchLoss)
			d = loss.get_info()
			#index = self.counter_k.get_global_count()
			index = None
			d.update({
				'_dt':dt,
				})
			self.loss_df.append(index, d)

	def add_opt_history_epoch(self):
		d = self.optimizer.get_info()
		#index = self.counter_epoch.get_global_count()
		index = None
		d.update({
			'_k':self.counter_k.get_global_count(),
			})
		self.opt_df.append(index, d)

	def add_loss_history_epoch(self, loss,
		dt=0,
		set_name=None,
		):
		if self.counter_epoch.check_counter_name_upper_bound('val_epoch'):
			assert isinstance(loss, ft_losses.BatchLoss)
			d = loss.get_info()
			#index = self.counter_epoch.get_global_count()
			index = None
			d.update({
				'_dt':dt,
				'_set':set_name,
				})
			self.loss_df_epoch.append(index, d)

	def add_metric_history_epoch(self, metrics_dict,
		dt=0,
		set_name=None,
		):
		if self.counter_epoch.check_counter_name_upper_bound('val_epoch'):
			d = {}
			for mn in metrics_dict.keys():
				metric = metrics_dict[mn]
				assert isinstance(metric, ft_metrics.BatchMetric)
				d[mn] = metric.get_info()['_metric']
			d.update({
				'_dt':dt,
				'_set':set_name,
				})
			#index = f'{self.counter_epoch.get_global_count()}.set_name'
			index = None
			self.metrics_df_epoch.append(index, d)

		#print(self.metrics_df_epoch.get_df())

	def get_metric_names(self):
		return [m.name for m in self.metrics]

	### along training methods
	def k_update(self):
		self.counter_k.update()

	def epoch_update(self):
		self.optimizer.update()
		self.counter_epoch.update()
		if self.counter_epoch.check_counter_name_upper_bound('earlystop_epoch'):
			raise ex.TrainingInterruptedError()

	def set_last_saved_filedir(self, last_saved_filedir):
		self.last_saved_filedir = last_saved_filedir

	def needs_save(self):
		return not self.save_mode==_C.SM_NO_SAVE

	def train(self):
		self.optimizer.train()

	def eval(self):
		self.optimizer.eval()

	def needs_evaluation(self):
		return self.counter_epoch.check_counter_name_upper_bound('val_epoch')

	def reset_early_stop(self):
		self.counter_epoch.reset_counter_name('earlystop_epoch')

	### get statistics
	def get_best_epoch(self):
		return self.best_epoch

	def set_best_epoch(self, best_epoch):
		self.best_epoch = best_epoch

	def get_time_per_iteration(self):
		loss_df = self.loss_df.get_df()
		return XError([v for v in loss_df['_dt'].values])

	def get_evaluation_set_names(self):
		loss_df_epoch = self.loss_df_epoch.get_df()
		return list(np.unique(loss_df_epoch['_set'].values))

	def get_time_per_epoch_set(self, set_name):
		loss_df_epoch = self.loss_df_epoch.get_df()
		return XError([v for v in loss_df_epoch['_dt'][loss_df_epoch['_set'].isin([set_name])].values])

	def get_time_per_epoch(self): # fixme only eval times
		evaluation_set_names = self.get_evaluation_set_names()
		return sum([self.get_time_per_epoch_set(set_name) for set_name in evaluation_set_names])

	def get_total_time(self):
		evaluation_set_names = self.get_evaluation_set_names()
		loss_df = self.loss_df.get_df()
		loss_df_epoch = self.loss_df_epoch.get_df()
		total_time = 0
		total_time += loss_df['_dt'].values.sum()
		total_time += sum([loss_df_epoch['_dt'][loss_df_epoch['_set'].isin([set_name])].values.sum() for set_name in evaluation_set_names]) # fixme
		return total_time

	### file methods
	def remove_filedir(self, filedir):
		if filedir is None:
			return
		files.delete_filedir(filedir, verbose=0) # remove last best model

	def check_save_condition(self, set_name):
		if self.save_mode==_C.SM_NO_SAVE:
			return False

		elif self.save_mode==_C.SM_ALL:
			return True

		elif self.save_mode==_C.SM_ONLY_ALL:
			self.remove_filedir(self.last_saved_filedir) # remove last best model
			return True

		elif self.save_mode==_C.SM_ONLY_INF_LOSS:
			loss_df_epoch = self.loss_df_epoch.get_df()
			loss_evolution = [np.inf]+[v for v in loss_df_epoch['_loss'][loss_df_epoch['_set'].isin([set_name])].values]
			loss_history = loss_evolution[:-1] # history
			actual_loss = loss_evolution[-1] # last one

			if actual_loss<np.min(loss_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				self.best_value = actual_loss
				return True
			else:
				return False

		elif self.save_mode==_C.SM_ONLY_INF_METRIC:
			metrics_df_epoch = self.metrics_df_epoch.get_df()
			metric_evolution = [np.inf]+[v for v in metrics_df_epoch[self.target_metric_crit][metrics_df_epoch['_set'].isin([set_name])].values]
			metric_history = metric_evolution[:-1] # history
			actual_metric = metric_evolution[-1] # last one

			if actual_metric<np.min(metric_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				self.best_value = actual_metric
				return True
			else:
				return False

		elif self.save_mode==_C.SM_ONLY_SUP_METRIC:
			metrics_df_epoch = self.metrics_df_epoch.get_df()
			metric_evolution = [-np.inf]+[v for v in metrics_df_epoch[self.target_metric_crit][metrics_df_epoch['_set'].isin([set_name])].values]
			metric_history = metric_evolution[:-1] # history
			actual_metric = metric_evolution[-1] # last one

			if actual_metric>np.max(metric_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				self.best_value = actual_metric
				return True
			else:
				return False

		else:
			raise Exception(f'save mode {self.save_mode} not supported')