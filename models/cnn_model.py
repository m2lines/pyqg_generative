import sys; sys.path.insert(0, '../')

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

from tools.cnn_tools import *
from tools.deep_inversion import DeepInversionGenerator

class SimpleCNN(nn.Module):
    '''
    CNN with n_in input and n_out output channel.
    Operates on torch.tensor data
    '''
    def __init__(self, n_in: int, n_out: int, model):
        super().__init__()
        if model=='block':
            self.mapping = nn.Sequential(*make_block(n_in,n_out,3,'False',False))
        elif model=='Andrew':
            self.mapping = AndrewCNN(n_in,n_out)
        elif model=='DeepInversion':
            self.mapping = DeepInversionGenerator(n_in,n_out)

        self.log_dict = {} # dictionary with training stats

    def forward(self, x: torch.tensor):
        '''Operates on torch.Tensor of size:
        Nbatch x n_in x Ny x Nx
        '''
        return self.mapping(x)

    def compute_loss(self, x: torch.tensor, ytrue: torch.tensor) -> dict:
        '''
        Computes loss on a batch.
        Loss must not depend explicitly 
        on the batch size (i.e. mean loss)!
        The number of output arguments may be arbitrary.
        The 'loss' output is used for optimization,
        while others are used for logger
        '''
        yhat = self.forward(x)
        loss = F.mse_loss(yhat, ytrue)
        with torch.no_grad():
            loss_l1 = F.l1_loss(yhat, ytrue)
        return {'loss': loss, 'loss_l1': loss_l1}

class SimpleCNNModel:
    '''
    Regression model based on SimpleCNN. Operates
    on dataset data.
    '''
    def __init__(self, inputs: list[tuple[str,int]], 
            targets: list[tuple[str,int]], model='block'):
        '''
        Examples of inputs and targets:
        [('u',0), ('v',0)].
        Each tuple corresponds to a channel. Str stands for 
        variable from the dataset and int for the layer id
        '''
        n_in = len(inputs)
        n_out = len(targets)
        self.inputs = inputs
        self.targets = targets
        self.net = SimpleCNN(n_in, n_out, model)
    
    def fit(self, ds_train: xr.DataArray, ds_test: xr.DataArray, 
            num_epochs=50, batch_size=64, learning_rate=0.001):

        X_train = dataset_to_array(ds_train, self.inputs)
        Y_train = dataset_to_array(ds_train, self.targets)

        X_test = dataset_to_array(ds_test, self.inputs)
        Y_test = dataset_to_array(ds_test, self.targets)

        x_scale = ChannelwiseScaler(X_train)
        y_scale = ChannelwiseScaler(Y_train)

        train(self.net, 
            x_scale.direct(X_train), y_scale.normalize(Y_train),
            x_scale.direct(X_test) , y_scale.normalize(Y_test),
            num_epochs, batch_size, learning_rate)

        self.x_scale = x_scale
        self.y_scale = y_scale
    
    def predict(self, ds: xr.DataArray):
        '''
        ds - dataset with inputs and targets
        returns dataset with targets and predictions
        '''
        # form array of predictions
        X = self.x_scale.direct(dataset_to_array(ds, self.inputs))
        preds = self.y_scale.denormalize(
            apply_function(self.net, self.net.forward, X)
        )
        
        ds_predict = extract_tuples(ds, self.targets)
        ds_predict = array_to_dataset(ds, preds, self.targets, '_predictions', ds_predict)
        
        return ds_predict