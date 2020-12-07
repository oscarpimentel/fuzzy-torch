import torch
import torch.nn as nn
from fuzzytorch.models.basics import MLP, Conv2DLinear
from fuzzytorch.datasets import TensorDict
import numpy as np

class MLPClassifier(nn.Module):
	def __init__(self,
		dropout:float=0.0,
		**kwargs):
		super().__init__()
		### ATTRIBUTES
		self.dropout = dropout
		self.input_dims = 3*32*32
		self.output_dims = 10
		self.embd_dims = 100
		self.embd_layers = 1

		embd_dims_list = [self.embd_dims]*self.embd_layers
		mlp_kwargs = {
			'activation':'relu',
			'last_activation':'linear',
			'dropout':self.dropout ,
		}
		self.classifier = MLP(self.input_dims, self.output_dims, embd_dims_list, **mlp_kwargs)
		print('classifier:',self.classifier)
		self.get_name()
	
	def get_name(self):
		name = 'mdl=mlp'
		name += f'°dropout={self.dropout}'
		name += f'°outD={self.output_dims}'
		self.name = name
		return self.name
	
	def get_output_dims(self):
		return self.classifier.get_output_dims()
	
	def forward(self, tensor_dict, **kwargs):
		x = tensor_dict['input']['x']
		x = x.view(x.shape[0],-1) # flatten
		x = self.classifier(x)
		tensor_dict.add('output', TensorDict({'y':x}))
		return tensor_dict

class CNN2DClassifier(nn.Module):
	def __init__(self,
		dropout:float=0.0,
		cnn_features:list=[16, 32, 64],
		uses_mlp_classifier:bool=True,
		**kwargs):
		super().__init__()
		### ATTRIBUTES
		self.dropout = dropout
		self.cnn_features = cnn_features
		self.uses_mlp_classifier = uses_mlp_classifier
		self.input_dims = 3
		self.input2d_dims = [32,32]
		self.output_dims = 10
		self.kernel_size = 5
		self.embd_layers = 1
		self.embd_dims = 100

		### build cnn embedding
		cnn_kwargs = {
			'activation':'relu',
			#'in_dropout':self.dropout,
			'pool_kernel_size':2,
			'padding_mode':'same',
		}
		self.cnn2d_embedding_stack = nn.ModuleList()
		cnn_in = self.input_dims
		input2d_dims = self.input2d_dims
		for cnn_out in self.cnn_features:
			cnn2d = Conv2DLinear(cnn_in, input2d_dims, cnn_out, self.kernel_size, **cnn_kwargs)
			self.cnn2d_embedding_stack.append(cnn2d)
			cnn_out, input2d_dims = cnn2d.get_output_dims()
			cnn_in = cnn_out
		
		self.last_cnn_output_dims = cnn_out
		self.last_cnn_output2d_dims = input2d_dims
		print(f'cnn2d_embedding_stack: {self.cnn2d_embedding_stack}')
		print(f'last_cnn_output_dims: {self.last_cnn_output_dims}')
		print(f'last_cnn_output2d_dims: {self.last_cnn_output2d_dims}')
		### build classifier
		if self.uses_mlp_classifier:
			self.build_mlp_classifier()
		else:
			self.build_custom_classifier()
		
		self.get_name()

	def build_mlp_classifier(self):
		embd_dims_list = [self.embd_dims]*self.embd_layers
		mlp_kwargs = {
			'activation':'relu',
			'last_activation':'linear',
			'dropout':self.dropout,
		}
		mlp_input_dims = np.prod(self.last_cnn_output2d_dims)*self.last_cnn_output_dims # flatten dims
		self.mlp_classifier = MLP(int(mlp_input_dims), self.output_dims, embd_dims_list, **mlp_kwargs)
		print('mlp_classifier:',self.mlp_classifier)

	def build_custom_classifier(self):
		'''
		add code here
		'''
		raise Exception('not implemented')

	def get_name(self):
		name = 'mdl-cnn2d'
		name += f'_dropout-{self.dropout}'
		name += f'_outD-{self.output_dims}'
		name += f'_cnnF-{".".join([str(cnnf) for cnnf in self.cnn_features])}'
		self.name = name
		return self.name
	
	def get_output_dims(self):
		return self.output_dims
	
	def forward(self, x, **kwargs):
		for cnn2d in self.cnn2d_embedding_stack:
			x = cnn2d(x)

		x = self.forward_mlp_classifier(x) if self.uses_mlp_classifier else self.forward_custom_classifier(x)
		return x

	def forward_mlp_classifier(self, x):
		x = x.view(x.shape[0],-1) # flatten
		x = self.mlp_classifier(x)
		return x

	def forward_custom_classifier(self, x):
		'''
		add code here
		'''
		raise Exception('not implemented')
		return x