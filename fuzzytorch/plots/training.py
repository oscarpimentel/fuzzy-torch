from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import fuzzytools.matplotlib.colors as cc
from scipy.ndimage import gaussian_filter1d
from fuzzytools.matplotlib.utils import save_fig

###################################################################################################################################################

def plot_loss(train_handler,
	sigma:float=4,
	save_rootdir:str=None,
	fig=None,
	ax=None,
	figsize:tuple=_C.PLOT_FIGSIZE,
	cmap=None,
	alpha:float=0.5,
	xlim:tuple=(1,None),
	ylim:tuple=(None,None),
	verbose:int=0,
	**kwargs):

	fig, ax = plt.subplots(1,1, figsize=figsize, dpi=_C.PLOT_DPI) if fig is None else (fig, ax)
	for lmonitor in train_handler.lmonitors:
		loss = lmonitor.loss_df['_loss']
		sublosses_names = [c for c in lmonitor.loss_df.columns if c[:1]!='_']
		sublosses = lmonitor.loss_df[sublosses_names]
		cmap = cc.get_default_cmap(len(sublosses_names)) if cmap is None else cmap
		colors = cmap.colors
		iterations = np.arange(len(loss))+1
		for k,subloss_name in enumerate(sublosses_names):
			ax.plot(iterations, sublosses[subloss_name], c=colors[k], label=f'{subloss_name}', alpha=alpha)

		ax.plot(iterations, loss, c=_C._CMAIN_LOSS, alpha=alpha, lw=1)
		ax.plot(iterations, gaussian_filter1d(loss, sigma), c=_C._CMAIN_LOSS, label=f'{lmonitor.name}') # gaussian fit for smooth
		
	best_epoch = lmonitor.get_best_epoch()
	#best_iteration = best_epoch*trainh.history_dict['ks_epochs']/trainh.history_dict['k_every']
	#ax.axvline(x=best_iteration, c='k', label='best iteration', lw=1, alpha=1) # vertical line in best epoch
	ax.set_xlabel('iterations')
	ax.set_ylabel('loss')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.legend()
	ax.grid(alpha=_C.PLOT_GRID_ALPHA)
	complete_save_roodir = train_handler.complete_save_roodir.split('/')[-1] # train_handler.get_complete_save_roodir().split('/')[-1]
	title = f'model: {complete_save_roodir} - id: {train_handler.id}' 
	ax.set_title(title)

	if not save_rootdir is None:
		new_save_dir = f'{save_rootdir}/{complete_save_roodir}'
		filedir = f'{new_save_dir}/plot{_C.KEY_VALUE_SEP_CHAR}trainloss{_C.KEY_KEY_SEP_CHAR}id{_C.KEY_VALUE_SEP_CHAR}{train_handler.id}.png'
		save_fig(filedir, fig)

	return fig, ax

def plot_evaluation_loss(train_handler,

	save_dir:str=None,
	title=None,
	fig=None,
	ax=None,
	figsize:tuple=_C.PLOT_FIGSIZE,
	cmap=None,
	alpha:float=1,
	xlim:tuple=(1,None),
	ylim:tuple=(None,None),
	verbose:int=0,
	**kwargs):

	fig, ax = (plt.subplots(1,1, figsize=figsize, dpi=_C.PLOT_DPI) if fig is None else (fig, ax))
	for trainh in train_handler.train_handlers:
		set_names = list(trainh.history_dict['sublosses_evolution_epochcheck'].keys())
		for set_name in set_names:
			is_train = set_name=='train'
			sublosses = trainh.history_dict['sublosses_evolution_epochcheck'][set_name]
			finalloss = trainh.history_dict['finalloss_evolution_epochcheck'][set_name]
			epochs = (np.arange(len(finalloss))+1)*int(trainh.early_stop_epochcheck_epochs)
			cmap = (cpc.get_default_cmap(len(sublosses.keys())) if cmap is None else cmap)
			colors = cmap.colors
			sublosses_names = list(sublosses.keys())
			for kmn,subloss_name in enumerate(sublosses_names): # multiple sublosses
				subloss_evol = np.array(sublosses[subloss_name])
				label = f'{subloss_name} - set: {set_name}'
				ax.plot(epochs, subloss_evol, get_train_style(is_train), alpha=0.5 if is_train else 1, c=colors[kmn], label=label)

			label = f'{trainh.name} - set: {set_name}'
			ax.plot(epochs, np.array(finalloss), get_train_style(is_train), c=_C._CMAIN_LOSS, alpha=(0.5 if is_train else 1), label=label)

	best_epoch = trainh.history_dict['best_epoch']
	ax.axvline(x=best_epoch, c='k', label='best epoch', lw=1, alpha=1) # vertical line in best epoch
	ax.set_xlabel('epochs')
	ax.set_ylabel('loss')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.legend()
	ax.grid(alpha=_C.PLOT_GRID_ALPHA)
	ax.set_title(get_title(train_handler, title))

	if not save_dir is None:
		new_save_dir = f'{save_dir}/{train_handler.model_name}'
		files.create_dir(new_save_dir)
		filedir = f'{new_save_dir}/plot-loss_id-{train_handler.id}.png'
		plt.savefig(filedir)

	return fig, ax


def plot_evaluation_metrics(train_handler,
	ylabel:str='accuracy',

	save_dir:str=None,
	title=None,
	fig=None,
	ax=None,
	figsize:tuple=_C.PLOT_FIGSIZE,
	cmap=None,
	alpha:float=1,
	xlim:tuple=(1,None),
	ylim:tuple=(None,None),
	verbose:int=0,
	**kwargs):
	
	fig, ax = (plt.subplots(1,1, figsize=figsize, dpi=_C.PLOT_DPI) if fig is None else (fig, ax))
	for trainh in train_handler.train_handlers:
		set_names = list(trainh.history_dict['metrics_evolution_epochcheck'].keys())
		for set_name in set_names:
			is_train = set_name=='train'
			metrics = trainh.history_dict['metrics_evolution_epochcheck'][set_name]
			cmap = (cpc.get_default_cmap(len(metrics.keys())) if cmap is None else cmap)
			colors = cmap.colors
			metrics_names = list(metrics.keys())
			for kmn,metric_name in enumerate(metrics_names): # multiple metrics
				metric_evol = np.array(metrics[metric_name])
				epochs = (np.arange(len(metric_evol))+1)*int(trainh.early_stop_epochcheck_epochs)
				label = f'metric: {metric_name} - set: {set_name}'
				ax.plot(epochs, metric_evol, get_train_style(is_train), alpha=(0.5 if is_train else 1), c=colors[kmn], label=label)

	best_epoch = trainh.history_dict['best_epoch']
	ax.axvline(x=best_epoch, c='k', label='best epoch', lw=1, alpha=1) # vertical line in best epoch
	ax.set_xlabel('epochs')
	ax.set_ylabel(ylabel)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.legend()
	ax.grid(alpha=_C.PLOT_GRID_ALPHA)
	ax.set_title(get_title(train_handler, title))

	if not save_dir is None:
		new_save_dir = f'{save_dir}/{train_handler.model_name}'
		files.create_dir(new_save_dir)
		filedir = f'{new_save_dir}/plot-metrics_id-{train_handler.id}.png'
		plt.savefig(filedir)

	return fig, ax

def plot_optimizer(train_handler,

	save_dir:str=None,
	title=None,
	fig=None,
	ax=None,
	figsize:tuple=_C.PLOT_FIGSIZE,
	cmap=None,
	alpha:float=1,
	xlim:tuple=(1,None),
	ylim:tuple=(None,None),
	verbose:int=0,
	**kwargs):

	fig, ax = (plt.subplots(1,1, figsize=figsize, dpi=_C.PLOT_DPI) if fig is None else (fig, ax))
	for trainh in train_handler.train_handlers:
		opt_decay_kwargs = trainh.history_dict['opt_kwargs_evolution_epoch'].keys()
		for key in opt_decay_kwargs:
			opt_values = np.array(trainh.history_dict['opt_kwargs_evolution_epoch'][key])
			epochs = np.arange(len(opt_values))+1
			ax.plot(epochs, opt_values, '--', c='k', label=f'{key}')

	best_epoch = trainh.history_dict['best_epoch']
	ax.axvline(x=best_epoch, c='k', label='best epoch', lw=1, alpha=1) # vertical line in best epoch
	ax.set_xlabel('epochs')
	ax.set_ylabel('value')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.legend()
	ax.grid(alpha=_C.PLOT_GRID_ALPHA)
	ax.set_title(get_title(train_handler, title))

	if not save_dir is None:
		new_save_dir = f'{save_dir}/{train_handler.model_name}'
		files.create_dir(new_save_dir)
		filedir = f'{new_save_dir}/plot-opt_id-{train_handler.id}.png'
		plt.savefig(filedir)

	return fig, ax


