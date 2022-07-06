import sys; sys.path.insert(0, '../')

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from tools.cnn_tools import *
from models.parameterization import Parameterization

class VarCNN(AndrewCNN):
    def forward(self, x):
        yhat = super().forward(x)
        return F.softplus(yhat)

class MeanVarModel(Parameterization):
    '''
    Model predicting pointwise conditional mean and variance
    https://arxiv.org/pdf/1811.05910.pdf
    '''
    def __init__(self, inputs: list[str], 
            targets: list[str]):
        super().__init__()

        # 2 because 2 layers
        n_in = 2 * len(inputs)
        n_out = 2 * len(targets)
        self.inputs = inputs
        self.targets = targets
        self.net_mean = AndrewCNN(n_in, n_out)
        self.net_var = VarCNN(n_in, n_out)

    @timer
    def fit(self, ds_train: xr.Dataset, ds_test: xr.Dataset, 
            num_epochs=50, batch_size=64, learning_rate=0.001):
        
        [X_train, X_test] = [extract_arrays(ds, self.inputs) for ds in (ds_train, ds_test)]
        [Y_train, Y_test] = [extract_arrays(ds, self.targets) for ds in (ds_train, ds_test)]
        
        [x_scale, y_scale] = [ChannelwiseScaler(XX) for XX in (X_train, Y_train)]
        self.x_scale = x_scale; self.y_scale = y_scale

        print(X_train.shape)

        train(self.net_mean, 
            x_scale.normalize(X_train), y_scale.normalize(Y_train),
            x_scale.normalize(X_test) , y_scale.normalize(Y_test),
            num_epochs, batch_size, learning_rate)

        def predict_mean(X):
            return y_scale.denormalize(
                apply_function(self.net_mean, self.net_mean.forward, x_scale.normalize(X)))
        
        [Yhat_train, Yhat_test] = [predict_mean(XX) for XX in (X_train, X_test)]
        
        rsq_train = (Y_train - Yhat_train)**2
        rsq_test = (Y_test - Yhat_test)**2

        train(self.net_var, 
            x_scale.normalize(X_train), y_scale.normalize_var(rsq_train),
            x_scale.normalize(X_test),  y_scale.normalize_var(rsq_test),
            num_epochs, batch_size, learning_rate)
    
    def nst_ch(self):
        return 2*len(self.targets)

    def predict(self, ds: xr.Dataset, noise=None):
        X = self.x_scale.direct(extract_arrays(ds, self.inputs))
        mean = self.y_scale.denormalize(
            apply_function(self.net_mean, self.net_mean.forward, X))
        var = self.y_scale.denormalize_var(
            apply_function(self.net_var, self.net_var.forward, X))

        if noise is None:
            noise = np.random.randn(*var.shape)

        Y = mean + noise * np.sqrt(var)

        return xr.merge((
            array_to_dataset(ds, Y, self.targets),
            array_to_dataset(ds, mean, self.targets, postfix='_mean'),
            array_to_dataset(ds, var, self.targets, postfix='_var')
        ))