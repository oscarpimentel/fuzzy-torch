from __future__ import print_function
from __future__ import division
from . import C_

import os
import torch
import numpy as np

from fuzzytools.progress_bars import ProgressBarMultiColor
from fuzzytools import C_ as C_fc
from fuzzytools import times
from fuzzytools import files
from fuzzytools import strings
from fuzzytools import prints
import warnings
from . import monitors as mon
from . import exceptions as ex
from .utils import get_model_name, TDictHolder, minibatch_dict_collate
from .models.utils import count_parameters

###################################################################################################################################################

class ModelTrainHandler(object):
	def __init__(self, model, lmonitors:list,
		save_rootdir='ft-save/',
		id=0,
		epochs_max=1e4,
		uses_train_eval_loader_methods=False,
		delete_all_previous_epochs_files:bool=True,
		extra_model_name_dict={},
		evaluate_train=True,
		):
		### CHECKS
		lmonitors = [lmonitors] if isinstance(lmonitors, mon.LossMonitor) else lmonitors
		assert isinstance(lmonitors, list) and all([isinstance(lmonitor, mon.LossMonitor) for lmonitor in lmonitors])
		assert isinstance(extra_model_name_dict, dict)

		self.model = model
		self.lmonitors = lmonitors

		self.save_rootdir = save_rootdir
		self.id = id
		self.epochs_max = int(epochs_max)
		self.uses_train_eval_loader_methods = uses_train_eval_loader_methods
		self.delete_all_previous_epochs_files = delete_all_previous_epochs_files
		self.extra_model_name_dict = extra_model_name_dict.copy()
		self.evaluate_train = evaluate_train
		self.reset()		
		
	def reset(self):
		self.get_model_name()
		self.get_extra_model_name()
		self.get_complete_model_name()
		self.get_complete_save_roodir() # default save rootdir

		self.device = 'cpu'
		self.device_name = 'cpu'

	def get_model_name(self):
		self.model_name = self.model.get_name()
		return self.model_name

	def get_extra_model_name(self):
		self.extra_model_name = None if len(self.extra_model_name_dict.keys())==0 else get_model_name(self.extra_model_name_dict)
		return self.extra_model_name

	def get_complete_model_name(self):
		self.complete_model_name = f'{self.model_name}'
		self.complete_model_name += '' if self.extra_model_name is None else f'{C_.KEY_KEY_SEP_CHAR}{self.extra_model_name}'
		return self.complete_model_name

	def get_complete_save_roodir(self):
		self.complete_save_roodir = f'{self.save_rootdir}/{self.complete_model_name}'
		return self.complete_save_roodir

	def set_complete_save_roodir(self, complete_save_roodir):
		self.complete_save_roodir = complete_save_roodir

	def clean_cache(self):
		if self.uses_gpu:
			torch.cuda.empty_cache()

	def build_gpu(self,
		gpu_index:int=0,
		):
		self.uses_gpu = torch.cuda.is_available()
		if gpu_index is None or gpu_index<0 or not self.uses_gpu: # is CPU
			warnings.warn('there is not CUDA nor GPUs... Using CPU >:(')
		else: # is GPU
			self.device_name = torch.cuda.get_device_name(gpu_index)
			self.device = torch.device(f'cuda:{gpu_index}')
		
		self.model.to(self.device)

	def __repr__(self):
		txt = ''
		txt += strings.get_bar(char=C_fc.BOT_SQUARE_CHAR) + '\n'
		txt += strings.color_str(f'model_name={self.model.get_name()}({count_parameters(self.model):,}[p])', 'blue')+'\n'
		txt += strings.color_str(f'id={self.id}', 'blue')+'\n'
		txt += strings.color_str(f'device={self.device} - device_name={self.device_name}', 'green')+'\n'
		txt += f'save_rootdir={self.complete_save_roodir}' + '\n'
		for lmonitor in self.lmonitors:
			txt += str(lmonitor) + '\n'
		return txt[:-1]

	def training_save_model(self, epoch, set_name):
		saved_filedir = None
		# check is can be saved
		can_save_model = any([lmonitor.check_save_condition(set_name) for lmonitor in self.lmonitors])
		if can_save_model:
			files.create_dir(self.complete_save_roodir, verbose=0)
			saved_filedir = f'{self.complete_save_roodir}/id{C_.KEY_VALUE_SEP_CHAR}{self.id}{C_.KEY_KEY_SEP_CHAR}epoch{C_.KEY_VALUE_SEP_CHAR}{epoch}.{C_.SAVE_FEXT}'

			dic_to_save = {
				'state_dict':self.model.state_dict(),
				'lmonitors':{lmonitor.name:lmonitor.get_save_dict() for lmonitor in self.lmonitors},
			}
			torch.save(dic_to_save, saved_filedir) # SAVE MODEL
			for lmonitor in self.lmonitors:
				lmonitor.set_last_saved_filedir(saved_filedir)
				lmonitor.reset_early_stop() # refresh counters for all
				#print(lmonitor.name, epoch, lmonitor.counter_epoch.get_global_count())
				lmonitor.set_best_epoch(epoch)

		return saved_filedir

	def evaluate_in_set(self, set_name:str, set_loader, training_kwargs:dict,
		):
		self.model.eval() # model eval mode!
		if self.uses_train_eval_loader_methods:
			set_loader.eval() # dataset eval mode!
		
		text = None
		evaluated = False
		with torch.no_grad():
			for lmonitor in self.lmonitors:
				lmonitor_cr = times.Cronometer()
				#for lmonitor_aux in self.lmonitors:
				#	lmonitor_aux.eval() # just in case

				if lmonitor.needs_evaluation():
					if text is None:
						text = f'[{set_name}]'

					evaluated = True
					set_losses = []
					set_metrics = {metric.name:[] for metric in lmonitor.metrics}
					for ki,in_tdict in enumerate(set_loader): # batches loop
						#print(f'  ({ki}) - {TDictHolder(in_tdict)}')
						out_tdict = self.model(TDictHolder(in_tdict).to(self.device), **training_kwargs)
						#print(f'  ({ki}) - {TDictHolder(out_tdict)}')
						loss_v = lmonitor.loss(out_tdict, **training_kwargs) # (b)
						set_losses += [loss_v]
						for metric in lmonitor.metrics:
							metric_v = metric(out_tdict, **training_kwargs) # (b)
							set_metrics[metric.name] += [metric_v]

					n = len(set_losses)
					## SET LOSS TO HYSTORY
					set_loss = sum(set_losses)
					set_loss = set_loss/n if set_loss.reduction_mode=='mean' else set_loss
					lmonitor.add_loss_history_epoch(set_loss, lmonitor_cr.dt(), set_name)
					text += f'[{lmonitor.name}] _loss={str(set_loss)}'

					### save metrics to history & bar text
					for metric in lmonitor.metrics:
						set_metric = sum(set_metrics[metric.name])
						set_metric = set_metric/n if set_metric.reduction_mode=='mean' else set_metric
						set_metrics[metric.name] = set_metric # replace
						text += f' - {metric.name}={str(set_metric)}'

					lmonitor.add_metric_history_epoch(set_metrics, lmonitor_cr.dt(), set_name)
					text += f' {lmonitor_cr}'

		return text, evaluated

	def update_bar(self, bar, text_dic,
		update:bool=False,
		):
		bar(text_dic, update)

	def create_dir(self):
		if any([int(lmonitor.needs_save()) for lmonitor in self.lmonitors]):
			files.create_dir(self.complete_save_roodir, verbose=1)

	def delete_filedirs(self):
		if self.delete_all_previous_epochs_files:
			to_delete_filedirs = [f for f in files.get_filedirs(self.complete_save_roodir) if files.get_dict_from_filedir(f)['id']==str(self.id)]
			if len(to_delete_filedirs)>0:
				epochs_to_delete = [int(files.get_dict_from_filedir(f)['epoch']) for f in to_delete_filedirs]
				prints.print_red(f'> (id={self.id}) deleting previous epochs={epochs_to_delete} in={self.complete_save_roodir}')
				files.delete_filedirs(to_delete_filedirs, verbose=0)

	def fit_loader(self, train_loader, val_loader,
		load:bool=False,
		k_every:int=1,
		training_kwargs:dict={},
		**kwargs):

		if load:
			self.load_model()
			return True

		self.create_dir()
		self.delete_filedirs()

		### TRAINING - BACKPROP
		if self.uses_train_eval_loader_methods:
			train_loader.train() # dataset train mode!

		print(strings.get_bar())
		ks_epochs = len(train_loader)
		training_bar = ProgressBarMultiColor(self.epochs_max*ks_epochs, ['train', 'eval-train', 'eval-val', 'early-stop'], [None, 'blue', 'red', 'yellow'])
		bar_text_dic = {}
		global_train_cr = times.Cronometer()
		can_be_in_loop = True
		end_with_nan = False
		for ke,epoch in enumerate(range(0, self.epochs_max+1)): # for epochs
			training_kwargs.update({'_epoch':epoch})
			try:
				if can_be_in_loop:
					#with torch.autograd.detect_anomaly(): # really useful but slow af
					self.model.train() # ensure train mode!
					if self.uses_train_eval_loader_methods:
						train_loader.train() # dataset train mode!

					for ki,in_tdict in enumerate(train_loader): # batches loop - k
						backprop_text = f'id={self.id} - _epoch={epoch:,}/{self.epochs_max:,}({ki:,}/{ks_epochs:,})'
						losses_text_list = []
						for kt,lmonitor in enumerate(self.lmonitors): # along train lmonitors
							lmonitor_cr = times.Cronometer()
							#for lmonitor_aux in self.lmonitors: # freeze all other models except actual
							#	lmonitor_aux.eval() # it's neccesary????
							#lmonitor.train() # ensure train mode!
							lmonitor.optimizer.none_grad() # none_grad zero_grad

							#print(f'  ({ki}) - {TDictHolder(in_tdict)}')
							out_tdict = self.model(TDictHolder(in_tdict).to(self.device), **training_kwargs) # Feed forward
							#print(f'  ({ki}) - {TDictHolder(out_tdict)}')
							loss = lmonitor.loss(out_tdict, **training_kwargs)
							loss.get_loss(get_tensor=True).backward() # gradient calculation
							lmonitor.optimizer.step() # step gradient

							with torch.no_grad():
								### save loss to history & bar text
								lmonitor.add_loss_history_k(loss, lmonitor_cr.dt())
								losses_text_list.append(f'[{lmonitor.name}] b={len(loss):,} - _loss={str(loss)} {lmonitor_cr}')
								lmonitor.k_update() # update k

						if ki>0:
							#break # debug
							pass
						
						### TEXT
						bar_text_dic['train'] = backprop_text + ''.join(losses_text_list)
						self.update_bar(training_bar, bar_text_dic)
					
					### save opt to history
					for lmonitor in self.lmonitors:
						lmonitor.add_opt_history_epoch()

					### evaluation in sets
					if self.evaluate_train:
						eval_set = 'train'
						text, evaluated = self.evaluate_in_set(eval_set, train_loader, training_kwargs)
						if evaluated:
							bar_text_dic[f'eval-{eval_set}'] = text
							self.update_bar(training_bar, bar_text_dic)

					eval_set = 'val'
					text, evaluated = self.evaluate_in_set(eval_set, val_loader, training_kwargs)
					if evaluated:
						bar_text_dic[f'eval-{eval_set}'] = text
						self.update_bar(training_bar, bar_text_dic)

					### saving model
					if evaluated:
						saved_filedir = self.training_save_model(epoch, 'val')
						self.update_bar(training_bar, bar_text_dic)

					### end of epoch!!
					text = f'[stop]'
					for lmonitor in self.lmonitors:
						lmonitor.epoch_update() # update epoch
						text += f'[{lmonitor.name}] counter_epoch={lmonitor.counter_epoch}'+('' if lmonitor.best_value is None else f' (best={lmonitor.best_value:.3f})')
					bar_text_dic['early-stop'] = text
					self.update_bar(training_bar, bar_text_dic, True)

			except ex.NanLossError as e:
				can_be_in_loop = False
				end_with_nan = True
				self.escape_training(training_bar, '*** nan loss ***')

			except ex.TrainingInterruptedError as e:
				can_be_in_loop = False
				self.escape_training(training_bar, '*** early stopping ***')

			except KeyboardInterrupt:
				can_be_in_loop = False
				self.escape_training(training_bar, '*** ctrl+c ***')

		training_bar.done()
		print(strings.get_bar())
		print('End of training!!!')
		for lmonitor in self.lmonitors:
			txt = f'[{lmonitor.name}] best_epoch={lmonitor.get_best_epoch()}'
			txt += f' - time_per_iteration={lmonitor.get_time_per_iteration()}[segs]'
			txt += f' - time_per_epoch={lmonitor.get_time_per_epoch()/60.}[mins]'
			txt += f' - total_time={lmonitor.get_total_time()/60.:3f}[mins]'
			print(txt)
		print(strings.get_bar(char=C_fc.TOP_SQUARE_CHAR))
		no_error_train = not end_with_nan
		return no_error_train

	def escape_training(self, training_bar, msj:str=None):
		training_bar.done()
		prints.print_red(msj)

	########### LOAD MODELS
	def load_model_cpu(self,
		target_id:int=None,
		target_epoch:int=None,
		):
		return self.load_model(target_id, target_epoch, 'cpu')

	def load_model(self,
		target_id:int=None,
		target_epoch:int=None,
		map_location:str=None,
		):
		if target_id is None:
			target_id = self.id
		if not target_id==self.id:
			self.id = target_id

		filedirs = files.get_filedirs(self.complete_save_roodir, fext=C_.SAVE_FEXT)
		if len(filedirs)==0:
			prints.print_red(f'*** no files in {self.complete_save_roodir} ***')
			raise Exception(f'*** no files in {self.complete_save_roodir} ***')
			return False

		if target_epoch is None: # seach the last epoch with that id
			filedics = [files.get_dict_from_filedir(filedir) for filedir in filedirs]
			epochs = [int(filedic['epoch']) for filedic in filedics if int(filedic['id'])==target_id]
			if len(epochs)==0:
				prints.print_red(f'*** no files with id {target_id} in {self.complete_save_roodir} ***')
				raise Exception(f'*** no files with id {target_id} in {self.complete_save_roodir} ***')
				return False

			epochs = sorted(epochs)
			target_epoch = epochs[-1]

		to_load_filedir = f'{self.complete_save_roodir}/id{C_.KEY_VALUE_SEP_CHAR}{target_id}{C_.KEY_KEY_SEP_CHAR}epoch{C_.KEY_VALUE_SEP_CHAR}{target_epoch}.{C_.SAVE_FEXT}'
		prints.print_blue(f'> loading model={to_load_filedir}')

		if map_location is None: # is GPU
			loaded_dic = torch.load(to_load_filedir)
		else:
			loaded_dic = torch.load(to_load_filedir, map_location='cpu')

		self.model.load_state_dict(loaded_dic['state_dict'])
		for lmonitor in self.lmonitors:
			#lmonitor.history_dict = loaded_dic['lmonitors'][lmonitor.name]
			pass
			
		return True


'''	def evaluate_in_set(self, set_name:str, set_loader, training_kwargs:dict,
		):
		self.model.eval() # model eval mode!
		if self.uses_train_eval_loader_methods:
			set_loader.eval() # dataset eval mode!
		
		text = None
		evaluated = False
		with torch.no_grad():
			for lmonitor in self.lmonitors:
				lmonitor_cr = times.Cronometer()
				#for lmonitor_aux in self.lmonitors:
				#	lmonitor_aux.eval() # just in case

				if lmonitor.needs_evaluation():
					if text is None:
						text = f'[{set_name}]'

					evaluated = True
					out_tdicts = []
					for ki,in_tdict in enumerate(set_loader): # batches loop
						#print(f'  ({ki}) - {TDictHolder(in_tdict)}')
						out_tdict = self.model(TDictHolder(in_tdict).to(self.device), **training_kwargs)
						#print(f'  ({ki}) - {TDictHolder(out_tdict)}')
						out_tdicts += [out_tdict]
					out_tdict = minibatch_dict_collate(out_tdicts) # slow?

					### save loss to history & bar text
					set_loss = lmonitor.loss(out_tdict, **training_kwargs)
					## SET LOSS TO HYSTORY
					lmonitor.add_loss_history_epoch(set_loss, lmonitor_cr.dt(), set_name)
					text += f'[{lmonitor.name}] _loss={str(set_loss)}'

					### save metrics to history & bar text
					set_metrics_dict = {}
					for metric in lmonitor.metrics:
						metric_name = metric.name
						set_metric = metric(out_tdict, **training_kwargs)
						#print(set_name, ki, metric.name, set_metric)
						text += f' - {metric_name}={str(set_metric)}'
						set_metrics_dict[metric_name] = set_metric

					lmonitor.add_metric_history_epoch(set_metrics_dict, lmonitor_cr.dt(), set_name)

					text += f' {lmonitor_cr}'

		return text, evaluated'''