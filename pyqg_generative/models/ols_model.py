import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from os.path import exists
import os

from pyqg_generative.tools.cnn_tools import AndrewCNN, ChannelwiseScaler, log_to_xarray, train, \
    apply_function, extract, prepare_PV_data, save_model_args
from pyqg_generative.models.parameterization import Parameterization

class OLSModel(Parameterization):
    def __init__(self):
        # Input 2 layers of q, 
        # output 2 layers of q_forcing_advection
        self.net = AndrewCNN(2,2)

        self.load_model()

    def fit(self, ds_train, ds_test, num_epochs=50, 
        batch_size=64, learning_rate=0.001):

        X_train, Y_train, X_test, Y_test, self.x_scale, self.y_scale = \
            prepare_PV_data(ds_train, ds_test)

        train(self.net,
            X_train, Y_train,
            X_test, Y_test,
            num_epochs, batch_size, learning_rate)
        
        self.save_model()

    def save_model(self):
        os.system('mkdir -p model')
        torch.save(self.net.state_dict(), 'model/net.pt')
        self.x_scale.write('x_scale.json')
        self.y_scale.write('y_scale.json')
        save_model_args('OLSModel')
        log_to_xarray(self.net.log_dict).to_netcdf('model/stats.nc')

    def load_model(self):
        if exists('model/net.pt'):
            print(f'reading OLSModel')
            self.net.load_state_dict(
                torch.load('model/net.pt', map_location='cpu')
            )
            self.x_scale = ChannelwiseScaler().read('x_scale.json')
            self.y_scale = ChannelwiseScaler().read('y_scale.json')

    def generate_latent_noise(self, ny, nx):
        return 0

    def predict_snapshot(self, m, noise):
        X = self.x_scale.normalize(m.q.astype('float32'))
        return self.y_scale.denormalize(
            apply_function(self.net, X)
            ).squeeze().astype('float64')
    
    def predict(self, ds):
        '''
        ds - standard dataset of
        run x time x nlev x ny x nx

        Output: dataset with three variables:
        q_forcing_advection
        q_forcing_advection_mean
        q_forcing_advection_var
        '''
        X = self.x_scale.normalize(extract(ds, 'q'))
        Y = xr.DataArray(
            self.y_scale.denormalize(
                apply_function(self.net, X)
                ).reshape(ds.q.shape),
                dims=['run', 'time', 'lev', 'y', 'x'])
        
        return xr.Dataset({'q_forcing_advection': Y, 
            'q_forcing_advection_mean': Y, 'q_forcing_advection_var': Y*0})