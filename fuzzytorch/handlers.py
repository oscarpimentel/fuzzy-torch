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
from .utils import get_model_name, TDictHolder, print_tdict
from .models.utils import count_parameters
from.files import FTFile
import cProfile
from timeit import default_timer as timer
import torch.autograd.profiler as profiler

def xxx(tdict, device,
    chunk_dtype=torch.float32,
    ):
	# return TDictHolder(in_tdict).to(device)
    chunk_d = {tdict[k].shape:{} for k in tdict.keys()}
    for key in tdict.keys():
        chunk_d[tdict[key].shape][key] = tdict[key]
    d = {}
    for shape in chunk_d.keys():
        tensors = [chunk_d[shape][key] for key in chunk_d[shape].keys()]
        # t = torch.cat(tensors, dim=0).to(device)
        t = torch.zeros([len(tensors)]+list(shape), dtype=chunk_dtype)
        for k in range(0, len(tensors)):
            t[k] = tensors[k]
        t = t.to(device)
        for k,key in enumerate(chunk_d[shape].keys()):
            d[key] = t[k].type(tdict[key].dtype)
    return d

###################################################################################################################################################

class ModelTrainHandler(object):
	def __init__(self, model, lmonitors:list,
		save_rootdir='ft-save/',
		id='1000', # fixme > mid
		epochs_max=1e4,
		delete_all_previous_epochs_files:bool=True,
		extra_model_name_dict={},
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
		self.delete_all_previous_epochs_files = delete_all_previous_epochs_files
		self.extra_model_name_dict = extra_model_name_dict.copy()
		self.reset()		
		
	def reset(self):
		torch.cuda.empty_cache()
		self.get_model_name()
		self.get_extra_model_name()
		self.get_complete_model_name()
		self.get_complete_save_roodir() # default save rootdir

		self.device = 'cpu'
		self.device_name = 'cpu'
		self.reset_file()

	def reset_file(self):
		self.file = None

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

	def build_gpu(self, device:str):
		uses_gpu = torch.cuda.is_available()
		device = 'cpu' if device is None or not uses_gpu else device
		if device=='cpu': # is CPU
			warnings.warn('there is not CUDA nor GPUs... Using CPU >:(')
			self.device_name = 'cpu'
			self.device = 'cpu'
		else: # is GPU
			self.device_name = torch.cuda.get_device_name(device)
			self.device = device
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
		# check is can be saved
		can_save_model = any([lmonitor.check_save_condition(set_name) for lmonitor in self.lmonitors])
		if can_save_model:
			saved_filedir = f'{self.complete_save_roodir}/id{C_.KEY_VALUE_SEP_CHAR}{self.id}{C_.KEY_KEY_SEP_CHAR}epoch{C_.KEY_VALUE_SEP_CHAR}{epoch}.{C_.SAVE_FEXT}'
			self.file = FTFile(saved_filedir, self.model, self.lmonitors)
			for lmonitor in self.lmonitors:
				lmonitor.set_last_saved_filedir(saved_filedir)
				lmonitor.reset_early_stop() # refresh counters for all
				lmonitor.set_best_epoch(epoch)
		return

	def evaluate_in_set(self, set_name:str, set_loader, training_kwargs:dict,
		):
		self.model.eval() # model eval mode!
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
						out_tdict = self.model(xxx(in_tdict, self.device), **training_kwargs)
						#print(f'  ({ki}) - {TDictHolder(out_tdict)}')
						loss_v = lmonitor.loss(out_tdict, **training_kwargs) # (b)
						set_losses += [loss_v]
						for metric in lmonitor.metrics:
							metric_v = metric(out_tdict, **training_kwargs) # (b)
							set_metrics[metric.name] += [metric_v]

					## SET LOSS TO HYSTORY
					set_loss = sum(set_losses)
					lmonitor.add_loss_history_epoch(set_loss, lmonitor_cr.dt(), set_name)
					text += f'[{lmonitor.name}] _loss={str(set_loss)}'

					### save metrics to history & bar text
					for metric in lmonitor.metrics:
						set_metric = sum(set_metrics[metric.name])
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

	def _delete_filedirs(self):
		if self.delete_all_previous_epochs_files:
			to_delete_filedirs = [f for f in files.get_filedirs(self.complete_save_roodir) if files.get_dict_from_filedir(f)['id']==self.id]
			if len(to_delete_filedirs)>0:
				epochs_to_delete = [int(files.get_dict_from_filedir(f)['epoch']) for f in to_delete_filedirs]
				prints.print_red(f'> (id={self.id}) deleting previous epochs={epochs_to_delete} in={self.complete_save_roodir}')
				files.delete_filedirs(to_delete_filedirs, verbose=0)

	def fit_loader(self, train_loader, eval_loaders:dict,
		load:bool=False,
		k_every:int=1,
		training_kwargs:dict={},
		always_save=False,
		delete_filedirs=True,
		train_dataset_method_call=None,
		**kwargs):
		eval_set_names = list(eval_loaders.keys())
		assert 'val' in eval_set_names
		if load:
			self.load_model()
			return True

		self.create_dir()
		if delete_filedirs:
			self._delete_filedirs()

		### TRAINING - BACKPROP
		print(strings.get_bar())
		ks_epochs = len(train_loader)
		training_bar = ProgressBarMultiColor(self.epochs_max*ks_epochs,
			['train']+[f'eval:{esn}' for esn in eval_set_names]+['early-stop'],
			[None]+['red' for esn in eval_set_names]+['yellow'],
			)
		bar_text_dic = {}
		global_train_cr = times.Cronometer()
		can_be_in_loop = True
		end_with_nan = False
		#p = cProfile.Profile();p.enable()
		for ke,epoch in enumerate(range(0, self.epochs_max+1)): # for epochs
			training_kwargs.update({'_epoch':epoch})
			try:
				if can_be_in_loop:
					#with torch.autograd.detect_anomaly(): # really useful but slow af
					#print('start')
					self.model.train() # ensure train mode!
					#print('start2')
					if not train_dataset_method_call is None:
						getattr(train_loader.dataset, train_dataset_method_call)()

					in_tdicts = enumerate(train_loader)
					for ki,in_tdict in in_tdicts: # batches loop - k
						losses_text_list = []

						for kt,lmonitor in enumerate(self.lmonitors): # along train lmonitors
							lmonitor_cr = times.Cronometer()
							#for lmonitor_aux in self.lmonitors: # freeze all other models except actual
							#	lmonitor_aux.eval() # it's neccesary????
							#lmonitor.train() # ensure train mode!

							lmonitor.optimizer.zero_grad(set_to_none=True) # False True

							#print(f'  ({ki}) - {TDictHolder(in_tdict)}')
							#print_tdict(in_tdict)
							out_tdict = self.model(xxx(in_tdict, self.device), **training_kwargs) # Feed forward
							#print(f'  ({ki}) - {TDictHolder(out_tdict)}')
							batch_loss = lmonitor.loss(out_tdict, **training_kwargs)
							batch_loss.backward() # gradient calculation
							lmonitor.optimizer.step() # step gradient

							### save loss to history & bar text
							lmonitor.add_loss_history_k(batch_loss, lmonitor_cr.dt())
							losses_text_list += [f'[{lmonitor.name}] b={len(batch_loss):,} - _loss={str(batch_loss)} {lmonitor_cr}']
							lmonitor.k_update() # update k
							pass

						### TEXT
						backprop_text = f'id={self.id} - _epoch={epoch:,}/{self.epochs_max:,}({ki:,}/{ks_epochs:,})'
						bar_text_dic['train'] = backprop_text + ''.join(losses_text_list)
						self.update_bar(training_bar, bar_text_dic)

					### save opt to history
					for lmonitor in self.lmonitors:
						lmonitor.add_opt_history_epoch()

					### evaluation in sets
					'''
					eval_set = 'train'
					text, evaluated = self.evaluate_in_set(eval_set, train_loader, training_kwargs)
					if evaluated:
						bar_text_dic[f'eval-{eval_set}'] = text
						self.update_bar(training_bar, bar_text_dic)
					'''
					for eval_set_name in eval_set_names:
						text, evaluated = self.evaluate_in_set(eval_set_name, eval_loaders[eval_set_name], training_kwargs)
						if evaluated:
							bar_text_dic[f'eval:{eval_set_name}'] = text
							self.update_bar(training_bar, bar_text_dic)

					#print('saving model')
					### saving model
					if evaluated:
						self.training_save_model(epoch, 'val')
						self.update_bar(training_bar, bar_text_dic)
						if always_save:
							self.file.save()

					#print('end of epoch')
					### end of epoch
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

		#p.disable(); p.dump_stats('prof.prof')
		# print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
		
		self.file.save()
		self.reset_file()
		training_bar.done()
		print(strings.get_bar())
		print('End of training!!!')
		for lmonitor in self.lmonitors:
			txt = f'[{lmonitor.name}] best_epoch={lmonitor.get_best_epoch()}'
			txt += f' - time_per_iteration={lmonitor.get_time_per_iteration()}[segs]'
			txt += f' - time_per_epoch={lmonitor.get_time_per_epoch()}[segs]'
			txt += f' - total_time={lmonitor.get_total_time()}[segs]'
			print(txt)
		print(strings.get_bar(char=C_fc.TOP_SQUARE_CHAR))
		no_error_train = not end_with_nan
		return no_error_train

	def escape_training(self, training_bar, msj:str=None):
		training_bar.done()
		prints.print_red(msj)

	def load_model(self,
		target_id:str=None,
		target_epoch:int=None,
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
			epochs = [int(filedic['epoch']) for filedic in filedics if filedic['id']==target_id]
			if len(epochs)==0:
				prints.print_red(f'*** no files with id {target_id} in {self.complete_save_roodir} ***')
				raise Exception(f'*** no files with id {target_id} in {self.complete_save_roodir} ***')
				return False

			epochs = sorted(epochs)
			target_epoch = epochs[-1]

		to_load_filedir = f'{self.complete_save_roodir}/id{C_.KEY_VALUE_SEP_CHAR}{target_id}{C_.KEY_KEY_SEP_CHAR}epoch{C_.KEY_VALUE_SEP_CHAR}{target_epoch}.{C_.SAVE_FEXT}'
		prints.print_blue(f'> loading model={to_load_filedir}')

		loaded_dic = torch.load(to_load_filedir, map_location=self.device)
		self.model.load_state_dict(loaded_dic['state_dict'])
		for lmonitor in self.lmonitors:
			#lmonitor.history_dict = loaded_dic['lmonitors'][lmonitor.name]
			pass
			
		return self.model


'''	def evaluate_in_set(self, set_name:str, set_loader, training_kwargs:dict,
		):
		self.model.eval() # model eval mode!
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