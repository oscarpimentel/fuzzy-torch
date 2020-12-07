from __future__ import print_function
from __future__ import division
from . import C_

#	VERSION V1.0

import os
import torch
import numpy as np

from flamingchoripan.progress_bars import ProgressBarMultiColor
from .datasets import TensorDict
from flamingchoripan import C_ as C_fc
from flamingchoripan import times
from flamingchoripan import files
from flamingchoripan import strings
from flamingchoripan import prints
import warnings
from . import train_handlers as ths
from . import exceptions as ex

###################################################################################################################################################

class ModelTrainHandler(object):
	def __init__(self, model, train_handlers:list,
		save_rootdir='../save/',
		id=0,
		epochs_max=1e4,
		uses_train_eval_loader_methods=False,
		delete_all_previous_epochs_files:bool=True,
		):
		if isinstance(train_handlers, ths.NewTrainHandler):
			train_handlers = [train_handlers]
		assert isinstance(train_handlers, list) and all([isinstance(train_handler, ths.NewTrainHandler) for train_handler in train_handlers])

		self.model = model
		self.train_handlers = train_handlers
		self.save_rootdir = save_rootdir
		self.complete_save_roodir = f'{self.save_rootdir}/{self.model.get_name()}'
		self.id = id
		self.epochs_max = int(epochs_max)
		self.uses_train_eval_loader_methods = uses_train_eval_loader_methods
		self.delete_all_previous_epochs_files = delete_all_previous_epochs_files

		self.device = 'cpu'
		self.device_name = 'cpu'

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
		txt += strings.color_str(f'model_name: {self.model.get_name()} - id: {self.id}', 'blue') + '\n'
		txt += strings.color_str(f'device: {self.device} - device_name: {self.device_name}', 'green') + '\n'
		txt += f'save_rootdir: {self.complete_save_roodir}' + '\n'
		for trainh in self.train_handlers:
			txt += str(trainh) + '\n'
		return txt

	def training_save_model(self, epoch, set_name):
		saved_filedir = None
		# check is can be saved
		can_save_model = any([trainh.check_save_condition(set_name) for trainh in self.train_handlers])
		if can_save_model:
			files.create_dir(self.complete_save_roodir, verbose=0)
			saved_filedir = f'{self.complete_save_roodir}/id{C_.KEY_VALUE_SEP_CHAR}{self.id}{C_.KEY_KEY_SEP_CHAT}epoch{C_.KEY_VALUE_SEP_CHAR}{epoch}.{C_.SAVE_FEXT}'

			dic_to_save = {
				'state_dict':self.model.state_dict(),
				'train_handler':{trainh.name:trainh.history_dict for trainh in self.train_handlers},
			}
			torch.save(dic_to_save, saved_filedir) # SAVE MODEL
			for trainh in self.train_handlers:
				trainh.set_last_saved_filedir(saved_filedir)
				trainh.reset_early_stop() # refresh counters for all
				trainh.set_best_epoch(epoch)

		return saved_filedir

	def evaluate_in_set(self, set_name:str, set_loader, model_kwargs:dict,
		):
		eval_cr = times.Cronometer()
		text = None
		evaluated = False
		with torch.no_grad():
			for trainh in self.train_handlers:
				if trainh.can_be_evaluated():
					if text is None:
						text = f'[{set_name}]'

					evaluated = True
					set_loss = []
					set_metrics = {mcn:[] for mcn in trainh.metric_crits_names}
					if self.uses_train_eval_loader_methods:
						set_loader.eval() # dataset eval mode!
					self.model.eval() # model eval mode!
					for k,in_tensor_dict in enumerate(set_loader): # batches loop
						out_tensor_dict = self.model(in_tensor_dict.to(self.device), **model_kwargs)
						loss = trainh.loss(out_tensor_dict)
						set_loss.append(loss)
						for metric in trainh.metrics:
							set_metrics[metric.name].append(metric(out_tensor_dict))

					set_loss = sum(set_loss)/len(set_loss)
					
					## SET LOSS TO HYSTORY
					trainh.history_dict['finalloss_evolution_epochcheck'][set_name].append(set_loss.get_loss())
					text += f'[{trainh.name}] *loss*: {loss}'

					## SET SUBLOSSES TO HYSTORY
					for subloss_name in set_loss.get_sublosses_names():
						trainh.add_subloss_history('sublosses_evolution_epochcheck', set_name, subloss_name, set_loss.get_subloss(subloss_name))

					## SET METRICS TO HYSTORY
					for metric_name in set_metrics.keys():
						metric_res = sum(set_metrics[metric_name])/len(set_metrics[metric_name])
						trainh.history_dict['metrics_evolution_epochcheck'][set_name][metric_name].append(metric_res.get_metric())
						text += f' - {metric_name}: {metric_res}'

					text += f' (eval-time: {eval_cr.dt_mins():.4f}[mins])'

		return text, evaluated

	def update_bar(self, bar, text_dic,
		update:bool=False,
		):
		bar(text_dic, update)

	def create_dir(self):
		if any([int(trainh.need_to_save()) for trainh in self.train_handlers]):
			files.create_dir(self.complete_save_roodir, verbose=1)

	def delete_filedirs(self):
		if self.delete_all_previous_epochs_files:
			to_delete_filedirs = [f for f in files.get_filedirs(self.complete_save_roodir) if files.get_dict_from_filedir(f)['id']==str(self.id)]
			if len(to_delete_filedirs)>0:
				epochs_to_delete = [int(files.get_dict_from_filedir(f)['epoch']) for f in to_delete_filedirs]
				prints.print_red(f'> deleting previous epochs {epochs_to_delete} in: {self.complete_save_roodir} (id: {self.id})')
				files.delete_filedirs(to_delete_filedirs, verbose=0)

	def fit_loader(self, train_loader, val_loader,
		model_kwargs:dict={},
		load:bool=False,
		k_every:int=1,
		**kwargs):

		if load:
			self.load_model()
			return True

		self.create_dir()
		self.delete_filedirs()

		### TRAINING - BACKPROP
		print(strings.get_bar())
		total_dataloader_k = len(train_loader)
		training_bar = ProgressBarMultiColor(self.epochs_max*total_dataloader_k, ['train', 'eval_train', 'eval_val', 'early_stop'], [None, 'blue', 'red', 'yellow'])
		bar_text_dic = {}
		global_train_cr = times.Cronometer()
		can_be_in_loop = True
		end_with_nan = False
		for ke,epoch in enumerate(range(1, self.epochs_max+1)): # for epochs
			model_kwargs.update({'epoch':epoch})
			try:
				if can_be_in_loop:
					#with torch.autograd.detect_anomaly(): # really useful but slow af
					train_cr = {trainh.name:times.Cronometer() for trainh in self.train_handlers}
					if self.uses_train_eval_loader_methods:
						train_loader.train() # dataset train mode!

					for k,in_tensor_dict in enumerate(train_loader): # batches loop
						assert isinstance(in_tensor_dict, TensorDict)
						backprop_text = f'id: {self.id} - epoch: {epoch}/{self.epochs_max}({k:,}/{total_dataloader_k:,})'
						losses_text_list = []
						for kt,trainh in enumerate(self.train_handlers):
							trainh.history_dict['ks_epochs'] = total_dataloader_k
							trainh.history_dict['k_every'] = k_every
							for trainh_aux in self.train_handlers: # freeze all other models except actual
								trainh_aux.eval() 

							trainh.train() # model train mode!
							trainh.optimizer.zero_grad() # set gradient to 0
							out_tensor_dict = self.model(in_tensor_dict.to(self.device), **model_kwargs) # Feed forward
							loss = trainh.loss(out_tensor_dict)
							#loss, loss_text, sublosses_dic = get_loss_from_losses(trainh.loss_crit, pred, target)
							loss.get_loss(numpy=False).backward() # gradient calculation
							trainh.optimizer.step() # step gradient

							losses_text_list.append(f'[{trainh.name}] *loss*: {loss}')
							
							### SET LOSS TO HYSTORY ONLY EVERY k_every ITERATIONS
							if k%k_every==0:
								trainh.history_dict['finalloss_evolution_k']['train'].append(loss.get_loss())
								for subloss_name in loss.get_sublosses_names():
									trainh.add_subloss_history('sublosses_evolution_k', 'train', subloss_name, loss.get_subloss(subloss_name))
						### TEXT
						bar_text_dic['train'] = backprop_text + ''.join(losses_text_list)
						self.update_bar(training_bar, bar_text_dic, True)

					### END OF EPOCH
					for trainh in self.train_handlers:
						trainh.history_dict['mins_evolution_epoch']['train'].append(train_cr[trainh.name].dt_mins())
						trainh.history_dict['global_mins_evolution_epoch']['train'].append(global_train_cr.dt_mins())
						trainh.epoch_update()
						### SET OPTIMIZER KWARGS TO HYSTORY
						for opt_key in trainh.optimizer.get_opt_kwargs():
							opt_kwarg_value = trainh.optimizer.get_kwarg_value(opt_key)
							trainh.history_dict['opt_kwargs_evolution_epoch'][opt_key].append(opt_kwarg_value)

					### EVALUATION IN SETS
					text, evaluated = self.evaluate_in_set('train', train_loader, model_kwargs)
					if evaluated:
						bar_text_dic['eval_train'] = text
						self.update_bar(training_bar, bar_text_dic)

					text, evaluated = self.evaluate_in_set('val', val_loader, model_kwargs)
					if evaluated:
						bar_text_dic['eval_val'] = text
						self.update_bar(training_bar, bar_text_dic)
						[trainh.evaluated() for trainh in self.train_handlers]

					### EARLY STOPPING
					text = f'[stop]'
					for trainh in self.train_handlers:
						trainh.early_stop_check() # raise exception
						text += f'[{trainh.name}] epoch_counter: ({trainh.epoch_counter}/{trainh.early_stop_epochcheck_epochs}) - patience: ({trainh.epochcheck_counter}/{trainh.early_stop_patience_epochchecks})'
					
					bar_text_dic['early_stop'] = text
					self.update_bar(training_bar, bar_text_dic)

					### SAVING MODEL
					if evaluated:
						saved_filedir = self.training_save_model(epoch, 'val')
						self.update_bar(training_bar, bar_text_dic)

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
		print('End of training!')
		for trainh in self.train_handlers:
			text = f'[th:{trainh.name}] best epoch: {trainh.get_best_epoch()} - convergence time: {trainh.get_mins_to_convergence():.4f}[mins]'
			text += f' - time per epoch: {trainh.get_mins_per_epoch():.4f}[mins]'
			print(text)
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

		complete_save_dir = f'{self.save_dir}/{self.model_name}'
		filedirs = files.get_filedirs(complete_save_dir, fext=C_.SAVE_FEXT)
		if len(filedirs)==0:
			prints.print_red(f'*** no files in {complete_save_dir} ***')
			return False

		if target_epoch is None: # seach the last epoch with that id
			filedics = [files.get_dict_from_filedir(filedir) for filedir in filedirs]
			epochs = [int(filedic['epoch']) for filedic in filedics if int(filedic['id'])==target_id]
			if len(epochs)==0:
				prints.print_red(f'*** no files with id {target_id} in {complete_save_dir} ***')
				return False

			epochs = sorted(epochs)
			target_epoch = epochs[-1]

		to_load_filedir = f'{complete_save_dir}/id{C_.KEY_VALUE_SEP_CHAR}{target_id}{C_.KEY_KEY_SEP_CHAT}epoch{C_.KEY_VALUE_SEP_CHAR}{target_epoch}.{C_.SAVE_FEXT}'
		prints.print_blue(f'> loading model: {to_load_filedir}')

		if map_location is None: # is GPU
			loaded_dic = torch.load(to_load_filedir)
		else:
			loaded_dic = torch.load(to_load_filedir, map_location='cpu')

		for trainh in self.train_handlers:
			trainh.history_dict = loaded_dic['train_handler'][trainh.name]
			trainh.model.load_state_dict(loaded_dic['state_dict'][trainh.name])
			
		return True


