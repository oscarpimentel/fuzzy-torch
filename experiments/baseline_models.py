import torch
import torch.nn as nn
from fuzzytorch.models.basics import MLP
from fuzzytorch.models.cnn.basics import MLConv2D
from fuzzytorch.utils import get_model_name
import numpy as np

###################################################################################################################################################

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
    
    def get_name(self):
        return get_model_name({
            'mdl':'mlp',
            'dropout':self.dropout,
            'output_dims':self.output_dims,
        })
    
    def get_output_dims(self):
        return self.classifier.get_output_dims()
    
    def forward(self, tdict, **kwargs):
        x = tdict['input']['x']
        x = x.view(x.shape[0],-1) # flatten
        x = self.classifier(x)
        tdict['model'] = {'y':x}
        return tdict

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

        self.output_dims = 10

        ### build cnn embedding
        cnn_kwargs = {
            'activation':'relu',
            'last_activation':'relu',
            #'in_dropout':self.dropout,
            'cnn_kwargs':{'kernel_size':5, 'stride':1, 'dilation':1,},
            #'padding_mode':'same',
        }
        self.ml_cnn2d = MLConv2D(3, [32,32], cnn_features[-1], cnn_features[:-1], **cnn_kwargs)
        self.last_cnn_output_dims = self.ml_cnn2d.get_output_dims()
        self.last_cnn_output_space = self.ml_cnn2d.get_output_space()
        print('ml_cnn2d:', self.ml_cnn2d)

        ### build classifier
        if self.uses_mlp_classifier:
            self.build_mlp_classifier()
        else:
            self.build_custom_classifier()

    def get_name(self):
        return get_model_name({
            'mdl':'cnn2d',
            'dropout':self.dropout,
            'output_dims':self.output_dims,
            'cnn_features':'.'.join([str(cnnf) for cnnf in self.cnn_features]),
        })

    def build_mlp_classifier(self):
        embd_dims_list = [50]
        mlp_kwargs = {
            'activation':'relu',
            'last_activation':'linear',
            'dropout':self.dropout,
        }
        mlp_input_dims = np.prod(self.last_cnn_output_space)*self.last_cnn_output_dims # flatten dims
        self.mlp_classifier = MLP(int(mlp_input_dims), self.output_dims, embd_dims_list, **mlp_kwargs)
        print('mlp_classifier:', self.mlp_classifier)

    def build_custom_classifier(self):
        '''
        add code here
        '''
        raise Exception('not implemented')
    
    def get_output_dims(self):
        return self.output_dims
    
    def forward(self, tdict, **kwargs):
        x = tdict['input']['x']
        x = self.ml_cnn2d(x)
        x = self.forward_mlp_classifier(x) if self.uses_mlp_classifier else self.forward_custom_classifier(x)
        tdict['model'] = {'y':x}
        return tdict

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