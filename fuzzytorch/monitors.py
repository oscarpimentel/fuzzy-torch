from __future__ import print_function
from __future__ import division
from . import C_

import os
import torch.nn as nn
import numpy as np
from . import losses as ft_losses
from . import metrics as ft_metrics
from . import optimizers as ft_optimizers
from . import exceptions as ex
import flamingchoripan.files as files
from flamingchoripan.counters import Counter
from flamingchoripan.datascience.statistics import XError
import pandas as pd

###################################################################################################################################################

def get_formated_df(df, index, index_name):
	df.index = [index]
	df.index.rename(index_name, inplace=True)
	return df

###################################################################################################################################################

class LossMonitor(object):
	def __init__(self, loss, optimizer, metrics,
		save_mode:str=C_.SM_NO_SAVE,
		target_metric_crit:str=None,
		k_counter_duration:int=C_.K_COUNTER_DURATION,
		val_epoch_counter_duration:int=C_.VAL_EPOCH_COUNTER_DURATION,
		earlystop_epoch_duration:int=C_.EARLYSTOP_EPOCH_DURATION,
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
		self.counter_k.reset()
		self.counter_epoch.reset()

	### repr
	def __repr__(self):
		def get_metrics_repr():
			return f'(target_metric_crit: {self.target_metric_crit})' if self.save_mode in [C_.SM_ONLY_INF_METRIC, C_.SM_ONLY_SUP_METRIC] else ''
		txt = ''
		txt += f'[{self.name}]'+'\n'
		txt += f' - opt-parameters: {len(self.optimizer):,}[p] - device: {self.optimizer.device()}'+'\n'
		txt += f' - save-mode: {self.save_mode}{get_metrics_repr()}'+'\n'
		txt += f' - counter_k: {self.counter_k} - counter_epoch: {self.counter_epoch}'+'\n'
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
		return {
			'info':info,
			'loss_df':self.loss_df,
			'opt_df':self.opt_df,
			'loss_df_epoch':self.loss_df_epoch,
			'metrics_df_epoch':self.metrics_df_epoch,
		}

	### history methods
	def add_loss_history_k(self, loss,
		dt=0,
		):
		if self.counter_k.check('k'):
			assert isinstance(loss, ft_losses.LossResult)
			new_df = loss.get_info_df()
			new_df = pd.concat([pd.DataFrame([[dt]], columns=['__dt__']), new_df], axis=1)
			new_df = get_formated_df(new_df, self.counter_k.get_global_count(), 'k')
			self.loss_df = new_df if not hasattr(self, 'loss_df') else pd.concat([self.loss_df, new_df])

	def add_opt_history_epoch(self):
		new_df = self.optimizer.get_info_df()
		new_df = pd.concat([pd.DataFrame([[self.counter_k.get_global_count()]], columns=['__k__']), new_df], axis=1)
		new_df = get_formated_df(new_df, self.counter_epoch.get_global_count(), 'epoch')
		self.opt_df = new_df if not hasattr(self, 'opt_df') else pd.concat([self.opt_df, new_df])

	def add_loss_history_epoch(self, loss,
		dt=0,
		set_name=None,
		):
		if self.counter_epoch.check('val_epoch'):
			assert isinstance(loss, ft_losses.LossResult)
			new_df = loss.get_info_df()
			c = ['__dt__'] if set_name is None else ['__dt__', '__set__']
			v = [dt] if set_name is None else [dt, set_name]
			new_df = pd.concat([pd.DataFrame([v], columns=c), new_df], axis=1)
			new_df = get_formated_df(new_df, self.counter_epoch.get_global_count(), 'val_epoch')
			self.loss_df_epoch = new_df if not hasattr(self, 'loss_df_epoch') else pd.concat([self.loss_df_epoch, new_df])

	def add_metric_history_epoch(self, metrics_dict,
		dt=0,
		set_name=None,
		):
		if self.counter_epoch.check('val_epoch'):
			new_dfs = []
			for mn in metrics_dict.keys():
				metric = metrics_dict[mn]
				assert isinstance(metric, ft_metrics.MetricResult)
				df = metric.get_info_df()
				df.rename(columns={'__metric__':mn}, inplace=True)
				new_dfs.append(df)

			c = ['__dt__'] if set_name is None else ['__dt__', '__set__']
			v = [dt] if set_name is None else [dt, set_name]
			new_df = pd.concat([pd.DataFrame([v], columns=c)]+new_dfs, axis=1)
			new_df = get_formated_df(new_df, self.counter_epoch.get_global_count(), 'val_epoch')
			self.metrics_df_epoch = new_df if not hasattr(self, 'metrics_df_epoch') else pd.concat([self.metrics_df_epoch, new_df])

	def get_metric_names(self):
		return [m.name for m in self.metrics]

	### along training methods
	def k_update(self):
		self.counter_k.update()

	def epoch_update(self):
		self.counter_epoch.update()
		if self.counter_epoch.check('earlystop_epoch'):
			raise ex.TrainingInterruptedError()

	def set_last_saved_filedir(self, last_saved_filedir):
		self.last_saved_filedir = last_saved_filedir

	def needs_save(self):
		return not self.save_mode==C_.SM_NO_SAVE

	def train(self):
		self.optimizer.train()

	def eval(self):
		self.optimizer.eval()

	def needs_evaluation(self):
		return self.counter_epoch.check('val_epoch')

	def reset_early_stop(self):
		self.counter_epoch.reset_cn('earlystop_epoch')

	### get statistics
	def get_best_epoch(self):
		return self.best_epoch

	def set_best_epoch(self, best_epoch):
		self.best_epoch = best_epoch

	def get_time_per_iteration(self):
		return XError(self.loss_df['__dt__'].values)

	def get_evaluation_set_names(self):
		return list(np.unique(self.loss_df_epoch['__set__'].values))

	def get_time_per_epoch_set(self, set_name):
		return XError(self.loss_df_epoch['__dt__'][self.loss_df_epoch['__set__'].isin([set_name])].values)

	def get_time_per_epoch(self):
		return sum([self.get_time_per_epoch_set(set_name) for set_name in self.get_evaluation_set_names()])

	def get_total_time(self):
		t = self.loss_df['__dt__'].values.sum()
		t += sum([self.loss_df_epoch['__dt__'][self.loss_df_epoch['__set__'].isin([set_name])].values.sum() for set_name in self.get_evaluation_set_names()])
		return t

	### file methods
	def remove_filedir(self, filedir):
		files.delete_filedir(filedir, verbose=0) # remove last best model

	def check_save_condition(self, set_name):
		if self.save_mode==C_.SM_NO_SAVE:
			return False

		elif self.save_mode==C_.SM_ALL:
			return True

		elif self.save_mode==C_.SM_ONLY_ALL:
			self.remove_filedir(self.last_saved_filedir) # remove last best model
			return True

		elif self.save_mode==C_.SM_ONLY_INF_LOSS:
			loss_evolution = self.loss_df_epoch['__loss__'][self.loss_df_epoch['__set__'].isin([set_name])].values
			if len(loss_evolution)<=1:
				return True # always save first and dont delete anything

			loss_history = loss_evolution[:-1] # history
			actual_loss = loss_evolution[-1] # last one

			if actual_loss<np.min(loss_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				return True
			else:
				return False

		elif self.save_mode==C_.SM_ONLY_INF_METRIC:
			metric_evolution = self.metrics_df_epoch[self.target_metric_crit][self.metrics_df_epoch['__set__'].isin([set_name])].values
			if len(metric_evolution)<=1:
				return True # always save first and dont delete anything

			metric_history = metric_evolution[:-1] # history
			actual_metric = metric_evolution[-1] # last one

			if actual_metric<np.min(metric_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				return True
			else:
				return False

		elif self.save_mode==C_.SM_ONLY_SUP_METRIC:
			metric_evolution = self.metrics_df_epoch[self.target_metric_crit][self.metrics_df_epoch['__set__'].isin([set_name])].values
			if len(metric_evolution)<=1:
				return True # always save first and dont delete anything

			metric_history = metric_evolution[:-1] # history
			actual_metric = metric_evolution[-1] # last one

			if actual_metric>np.max(metric_history): # must save and delete
				self.remove_filedir(self.last_saved_filedir) # remove last best model
				return True
			else:
				return False

		else:
			raise Exception(f'save mode {self.save_mode} not supported')