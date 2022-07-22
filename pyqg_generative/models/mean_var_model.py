import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import torch.nn.functional as functional

from os.path import exists
import os

from pyqg_generative.tools.cnn_tools import AndrewCNN, ChannelwiseScaler, log_to_xarray, train, \
    apply_function, extract, prepare_PV_data, save_model_args
from pyqg_generative.models.parameterization import Parameterization

class VarCNN(AndrewCNN):
    def forward(self, x):
        yhat = super().forward(x)
        return functional.softplus(yhat)

class MeanVarModel(Parameterization):
    '''
    Model predicting pointwise conditional mean and variance
    https://arxiv.org/pdf/1811.05910.pdf
    '''
    def __init__(self):
        super().__init__()
        # Input 2 layers of q, 
        # output 2 layers of q_forcing_advection
        self.net_mean = AndrewCNN(2,2)
        self.net_var = VarCNN(2,2)

        self.load_model()

    def fit(self, ds_train, ds_test, num_epochs=50, 
        batch_size=64, learning_rate=0.001):

        X_train, Y_train, X_test, Y_test, self.x_scale, self.y_scale = \
            prepare_PV_data(ds_train, ds_test)
        
        train(self.net_mean,
            X_train, Y_train,
            X_test, Y_test,
            num_epochs, batch_size, learning_rate)

        Yhat_train = apply_function(self.net_mean, X_train)
        Yhat_test = apply_function(self.net_mean, X_test)
        
        rsq_train = (Y_train - Yhat_train)**2
        rsq_test = (Y_test - Yhat_test)**2

        train(self.net_var, 
            X_train, rsq_train,
            X_test,  rsq_test,
            num_epochs, batch_size, learning_rate)

        self.save_model()

    def save_model(self):
        os.system('mkdir -p model')
        torch.save(self.net_mean.state_dict(), 'model/net_mean.pt')
        torch.save(self.net_var.state_dict(), 'model/net_var.pt')
        self.x_scale.write('x_scale.json')
        self.y_scale.write('y_scale.json')
        save_model_args('MeanVarModel')
        log_to_xarray(self.net_mean.log_dict).to_netcdf('model/stats_mean.nc')
        log_to_xarray(self.net_var.log_dict).to_netcdf('model/stats_var.nc')

    def load_model(self):
        if exists('model/net_mean.pt'):
            print(f'reading MeanVarModel')
            self.net_mean.load_state_dict(
                torch.load('model/net_mean.pt', map_location='cpu'))
            self.net_var.load_state_dict(
                torch.load('model/net_var.pt', map_location='cpu'))
            self.x_scale = ChannelwiseScaler().read('x_scale.json')
            self.y_scale = ChannelwiseScaler().read('y_scale.json')

    def generate_latent_noise(self, ny, nx):
        return np.random.randn(2, ny, nx)

    def predict_snapshot(self, m, noise):
        X = self.x_scale.normalize(m.q.astype('float32'))
        return self.y_scale.denormalize(
                apply_function(self.net_mean, X) + noise * (apply_function(self.net_var, X))**0.5
                ).squeeze().astype('float64')
          
    def predict(self, ds):
        X = self.x_scale.normalize(extract(ds, 'q'))
        mean = xr.DataArray(
            self.y_scale.denormalize(
                apply_function(self.net_mean, X)
                ).reshape(ds.q.shape),
                dims=['run', 'time', 'lev', 'y', 'x'])
        
        var = xr.DataArray(
            self.y_scale.denormalize_var(
                apply_function(self.net_var, X)
                ).reshape(ds.q.shape),
                dims=['run', 'time', 'lev', 'y', 'x'])

        Y = mean + np.sqrt(var) * np.random.randn(*var.shape)
        
        return xr.Dataset({'q_forcing_advection': Y, 
            'q_forcing_advection_mean': mean, 'q_forcing_advection_var': var})