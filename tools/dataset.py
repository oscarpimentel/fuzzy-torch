import torch
from torch.utils.data import Dataset
import numpy as np
import random

class MyDataset(Dataset):
	def __init__(self, x:np.ndarray, y:np.ndarray,
		uses_da:bool=False,
		):
		''' Clase extendida de Dataset encargada de generar samples para el DataLoader.
		Esta clase incluye la normalización y el data agumentation para CIFAR10.
		Al momento de utilizarla en conjunto con DataLoader, el retorno de esta clase resulta en una tupla del tipo (data, target) con dimensiones (b,c,h,w), (b).
		Donde b es el tamaño del batch entregado al DataLoader.

		Args:
		x (N,h,w,c): tensor con el dataset completo de N imagenes CIFAR10. h=height (32), w=width (32), c=channels (3).
		y (N): vector o lista con la información de etiquetas para cada imagen en x.
		'''
		x = torch.tensor(x).permute(0,3,1,2).float()/255. # (N,h,w,c) > (N,c,h,w), friendly dims for pytorch cnn
		y = torch.tensor(y)
		assert len(x.shape)==4
		assert len(x)==len(y)
		print(f'x: {x.shape} - x.max: {x.max()} - y: {y.shape} - y.max: {y.max()}')
		self.x = x
		self.y = y
		self.n_scale = 0.05 # standar noise scale
		self.uses_da = uses_da # data augmentation
		self.vertical_flip = False
		self.horizontal_flip = True
		self.set_norm_values()
		
	def __len__(self):
		return len(self.x)
	
	def set_norm_values(self,
		mean:torch.Tensor=None,
		std:torch.Tensor=None,
		):
		''' Calcula la media y desviacion estandar en el dataset de ser necesario. No es calculada si estos valores se incluyen como argumentos.
		Args:
		mean (c): Media encontrada en el dataset completo de N samples por canal.
		std (c): Desviación estandar encontrada en el dataset completo de N samples por canal.
		'''
		self.mean = torch.mean(self.x, dim=(0,2,3)) if mean is None else mean # (N,c,h,w) > (c)
		self.std = torch.std(self.x, dim=(0,2,3)) if std is None else std # (N,c,h,w) > (c)
		
	def get_norm_values(self):
		''' Usado para traspasar los valores de normalización de train a val/test
		'''
		return self.mean, self.std
	
	def norm_item(self, x:torch.Tensor):
		eps = 1e-10
		mean = self.mean[...,None,None] # (c) > (c,1,1)
		std = self.std[...,None,None] # (c) > (c,1,1)
		#print(f'mean: {mean.shape} - std: {std.shape}')
		x = (x-mean)/(std+eps) # (c,h,w)
		return x
	
	def apply_da(self, x:torch.Tensor):
		if not self.uses_da:
			return x # just return
		
		# random image flips
		x = torch.flip(x, dims=[2]) if self.horizontal_flip and random.random()>0.5 else x # horizontal flip
		x = torch.flip(x, dims=[1]) if self.vertical_flip and random.random()>0.5 else x # vertical flip
		
		# add noise
		noise_tensor = torch.normal(0, 1, size=x.size()) # (c,h,w)
		x = x + noise_tensor * self.n_scale
		return x # (c,h,w)
	
	def __getitem__(self, idx:int):
		''' Obtiene una muestra desde el dataset completo. Funcion utilizada por la clase DataLoader.
		'''
		x = self.x[idx] # (N,c,h,w) > (c,h,w)
		x = self.norm_item(x) # (c,h,w)
		x = self.apply_da(x) # (c,h,w)
		y = self.y[idx] # (N) > ()
		#print(f'x: {x.shape} - y: {y.shape}')
		#return {'x':x}, {'y':self.y[idx]} # dict tuple return type
		return x, y # tuple return type