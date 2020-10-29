from __future__ import print_function
from __future__ import division
from . import C_

#	VERSION V1.0

import os
import torch
import numpy as np

from ..myUtils.progress_bars import ProgressBarMultiColor
from . import utils as tfutils
from .. import myUtils
from ..myUtils import time
from ..myUtils import files
from ..myUtils import strings
from ..myUtils import prints
from .losses import get_loss_from_losses, get_loss_text
from . import train_handlers as thandles
from . import exceptions as exc

###################################################################################################################################################

class ModelsTrainHandler(object):
	def __init__(self, model_name:str, train_handlers:list,
		**kwargs):
		# ATTRIBUTES
		setattr(self, 'model_name', model_name)
		setattr(self, 'train_handlers', train_handlers)
		setattr(self, 'save_dir', 'save')
		setattr(self, 'id', 0)
		setattr(self, 'epochs_max', int(1e4))
		setattr(self, 'main_color', 'gray')
		for name, val in kwargs.items():
			setattr(self, name, val)
		assert isinstance(train_handlers, list) or isinstance(train_handlers, thandles.NewTrainHandler)
		if isinstance(train_handlers, thandles.NewTrainHandler):
			self.train_handlers = [self.train_handlers]

		print(strings.get_bar(char=myUtils.C_.BOT_SQUARE_CHAR))
		print(f'save dir: {self.save_dir}/{self.model_name}')
		prints.print_blue(f'model name: {self.model_name} (id: {self.id})')

	def clean_cache(self):
		if self.cuda:
			torch.cuda.empty_cache()

	def build_gpu(self,
		gpu_index:int=0,
		):
		self.cuda = torch.cuda.is_available()
		if gpu_index is None or gpu_index<0 or not self.cuda: # CPU
			prints.print_green('There is not CUDA nor GPUs... Using CPU >:(')
			self.GPU = 'cpu'
		else: # GPU
			tfutils.print_gpu_info()
			self.GPU = torch.device(f'cuda:{gpu_index}')
			prints.print_green(f'Using: {self.GPU} ({torch.cuda.get_device_name(gpu_index)})')

		for trainh in self.train_handlers:
			trainh.mount_model(self.GPU) # model to GPU
			trainh.print_info()

	def training_save_model(self, epoch, set_name):
		saved_filedir= None
		# check is can be saved
		can_save_model = sum([int(trainh.check_save_condition(set_name)) for trainh in self.train_handlers])>0
		if can_save_model:
			new_folder = f'{self.save_dir}/{self.model_name}'
			files.create_dir(new_folder, verbose=0)
			saved_filedir = f'{new_folder}/id{C_.KEY_VALUE_SEP_CHAR}{self.id}{C_.KEY_KEY_SEP_CHAT}epoch{C_.KEY_VALUE_SEP_CHAR}{epoch}.{C_.SAVE_FEXT}'

			dic_to_save = {
				'state_dict':{trainh.name:trainh.model.state_dict() for trainh in self.train_handlers},
				'train_handler':{trainh.name:trainh.history_dict for trainh in self.train_handlers},
			}
			torch.save(dic_to_save, saved_filedir) # SAVE MODEL
			for trainh in self.train_handlers:
				trainh.set_last_saved_filedir(saved_filedir)
				trainh.reset_early_stop() # refresh counters for all
				trainh.set_best_epoch(epoch)

		return saved_filedir

	def evaluate_in_set(self, set_name:str, set_loader, model_kwargs:dict):
		eval_cr = time.Cronometer()
		text = None
		evaluated = False
		with torch.no_grad():	
			for trainh in self.train_handlers:
				if trainh.can_be_evaluated():
					if text is None:
						text = f'[{set_name}]'

					evaluated = True
					set_finalloss = []
					set_sublosses = {sln:[] for sln in trainh.get_sublosses_names()}
					set_metrics = {mcn:[] for mcn in trainh.metric_crits_names}
					set_loader.eval() # dataset eval mode!
					trainh.model.eval() # model eval mode!
					for k,(data_, target_) in enumerate(set_loader):
						data, target = tfutils.get_data_target_gpu(data_, target_, self.GPU) # mount data in gpu
						pred = trainh.model(data, **model_kwargs)
						loss, _, sublosses_dic = get_loss_from_losses(trainh.loss_crit, pred, target)
						set_finalloss.append(loss.item())

						for subloss_name in sublosses_dic.keys():
							set_sublosses[subloss_name].append(sublosses_dic[subloss_name].item())

						for metric in trainh.metric_crits:
							set_metrics[metric.name].append(metric.fun(pred, target, **metric.kwargs).item())

					set_finalloss = np.mean(set_finalloss)
					set_sublosses = {sln:np.mean(set_sublosses[sln]) for sln in trainh.get_sublosses_names()}
					set_metrics = {mcn:np.mean(set_metrics[mcn]) for mcn in trainh.metric_crits_names}

					## SET LOSS TO HYSTORY
					trainh.history_dict['finalloss_evolution_epochcheck'][set_name].append(set_finalloss)
					text += f'[th:{trainh.name}] *loss*: {get_loss_text(set_finalloss)}'

					## SET SUBLOSSES TO HYSTORY
					sublosses_text = []
					for subloss_name in set_sublosses.keys():
						trainh.add_subloss_history('sublosses_evolution_epochcheck', set_name, subloss_name, set_sublosses[subloss_name])
						sublosses_text.append(get_loss_text(set_sublosses[subloss_name]))
					if len(sublosses_text)>0:
						text += f'={"+".join(sublosses_text)}'

					## SET METRICS TO HYSTORY
					for metric_name in set_metrics.keys():
						metric_value = set_metrics[metric_name]
						trainh.history_dict['metrics_evolution_epochcheck'][set_name][metric_name].append(metric_value)
						text += f' - {metric_name}: {metric_value:.5f}'

					text += f' (eval_time: {eval_cr.dt_mins():.4f}[mins])'

		return text, evaluated

	def update_bar(self, bar, text_dic,
		update:bool=False,
		):
		bar(text_dic, update)

	def fit_loader(self, train_loader, val_loader,
		model_kwargs:dict={},
		load:bool=False,
		k_every:int=1,
		delete_all_previous_epochs_files:bool=True,
		**kwargs):

		if load:
			self.load_model()
			return True

		complete_save_dir = f'{self.save_dir}/{self.model_name}'
		if not os.path.isdir(complete_save_dir):
			if sum([int(trainh.need_to_save()) for trainh in self.train_handlers])>0: # at least one
				files.create_dir(complete_save_dir, verbose=1)

		if delete_all_previous_epochs_files:
			to_delete_filedirs = files.get_filedirs(complete_save_dir)
			to_delete_filedirs = [f for f in to_delete_filedirs if files.get_dict_from_filedir(f)['id']==str(self.id)]
			if len(to_delete_filedirs)>0:
				epochs_to_delete = [int(files.get_dict_from_filedir(f)['epoch']) for f in to_delete_filedirs]
				prints.print_red(f'> deleting previous epochs {epochs_to_delete} in: {complete_save_dir} (id: {self.id})')
				files.delete_filedirs(to_delete_filedirs, verbose=0)

		### TRAINING - BACKPROP
		print(strings.get_bar())
		total_dataloader_k = len(train_loader)
		training_bar = ProgressBarMultiColor(self.epochs_max*total_dataloader_k, ['train', 'eval_train', 'eval_val', 'early_stop'], [None, 'blue', 'red', 'yellow'])
		bar_text_dic = {}
		global_train_cr = time.Cronometer()
		can_be_in_loop = True
		end_with_nan = False
		for ke,epoch in enumerate(range(1, self.epochs_max+1)): # FOR EPOCHS
			model_kwargs.update({'epoch':epoch})
			try:
				if can_be_in_loop:
					#with torch.autograd.detect_anomaly(): # really useful but slow af
					train_cr = {trainh.name:time.Cronometer() for trainh in self.train_handlers}
					train_loader.train() # dataset train mode!
					for k,(data_, target_) in enumerate(train_loader):
						backprop_text = f'id: {self.id} - epoch: {epoch}/{self.epochs_max}({k:d}/{total_dataloader_k:d})'
						losses_text_list = []
						for kt,trainh in enumerate(self.train_handlers):
							trainh.history_dict['ks_epochs'] = total_dataloader_k
							trainh.history_dict['k_every'] = k_every
							[trainh_aux.model.eval() for trainh_aux in self.train_handlers] # freeze all other models except actual

							data, target = tfutils.get_data_target_gpu(data_, target_, self.GPU) # mount data in gpu
							trainh.model.train() # model train mode!
							trainh.optimizer.zero_grad() # set gradient to 0
							pred = trainh.model(data, **model_kwargs) # Feed forward
							loss, loss_text, sublosses_dic = get_loss_from_losses(trainh.loss_crit, pred, target)
							loss.backward() # gradient calculation
							trainh.optimizer.step() # step gradient
							trainh.nan_check(loss.item())

							losses_text_list.append(f'[th:{trainh.name}] *loss*: {loss_text}')
							
							### SET LOSS TO HYSTORY ONLY EVERY k_every ITERATIONS
							if k%k_every==0:
								trainh.history_dict['finalloss_evolution_k']['train'].append(loss.item())
								for subloss_name in sublosses_dic.keys():
									trainh.add_subloss_history('sublosses_evolution_k', 'train', subloss_name, sublosses_dic[subloss_name].item())
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
						text += f'[th:{trainh.name}] epoch_counter: ({trainh.epoch_counter}/{trainh.early_stop_epochcheck_epochs}) - patience: ({trainh.epochcheck_counter}/{trainh.early_stop_patience_epochchecks})'
					
					bar_text_dic['early_stop'] = text
					self.update_bar(training_bar, bar_text_dic)

					### SAVING MODEL
					if evaluated:
						saved_filedir = self.training_save_model(epoch, 'val')
						self.update_bar(training_bar, bar_text_dic)

			except exc.NanLossException as e:
				can_be_in_loop = False
				end_with_nan = True
				self.escape_training(training_bar, '*** nan loss ***')

			except exc.TrainingStopException as e:
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
		print(strings.get_bar(char=myUtils.C_.TOP_SQUARE_CHAR))
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


