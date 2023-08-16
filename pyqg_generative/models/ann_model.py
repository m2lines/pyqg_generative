import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from os.path import exists
import os
import json

from pyqg_generative.models.parameterization import Parameterization
from pyqg_generative.tools.cnn_tools import ANN, train, save_model_args, \
    log_to_xarray, prepare_data_ANN, xarray_to_stencil, apply_function, \
    stencil_to_numpy, stack_images

class ANNModel(Parameterization):
    def __init__(self, stencil_size=3, hidden_channels = [24, 24], folder='model', read=True):
        super().__init__()
        self.folder = folder
        os.system(f'mkdir -p {folder}')

        self.stencil_size = stencil_size
        self.hidden_channels = hidden_channels

        # input is the PV field on stencil_size x stencil_size grid
        # output is a single point of dq/dt

        self.net = ANN(stencil_size**2, 1, hidden_channels)
        if read:
            self.load_model(folder)

    def fit(self, ds_train, ds_test, num_epochs=50, 
        batch_size=2**16, learning_rate=0.001):

        X_train, Y_train, self.x_scale, self.y_scale = prepare_data_ANN(ds_train, self.stencil_size)
        X_test, Y_test, _, _ = prepare_data_ANN(ds_test, self.stencil_size)

        X_train = X_train / self.x_scale
        X_test = X_test / self.x_scale
        Y_train = Y_train / self.y_scale
        Y_test = Y_test / self.y_scale

        train(self.net,
            X_train, Y_train,
            X_test, Y_test,
            num_epochs, batch_size, learning_rate)
        
        self.save_model()

    def save_model(self):
        os.system('mkdir -p model')
        torch.save(self.net.state_dict(), f'{self.folder}/net.pt')
        
        with open(f'{self.folder}/scale.json', 'w') as file:
            json.dump({'x_scale': self.x_scale, 'y_scale': self.y_scale}, file)

        save_model_args('ANNModel', folder=self.folder, 
                        stencil_size=self.stencil_size,
                        hidden_channels=self.hidden_channels)

        log_to_xarray(self.net.log_dict).to_netcdf(f'{self.folder}/stats.nc')        

    def load_model(self, folder):
        if exists(f'{folder}/net.pt'):
            print(f'reading OLSModel from {folder}')
            self.net.load_state_dict(
                torch.load(f'{folder}/net.pt', map_location='cpu')
            )
            with open(f'{folder}/scale.json', 'r') as file:
                scale = json.load(file)
                self.x_scale = scale['x_scale']
                self.y_scale = scale['y_scale']

    def generate_latent_noise(self, ny, nx):
        return 0
    
    def predict_snapshot(self, m, noise):
        '''
        m is the pyqg model
        '''
        q = xr.DataArray(m.q, dims=['lev', 'y', 'x']).astype('float32')

        x = xarray_to_stencil(q, self.stencil_size) / self.x_scale

        y = self.y_scale * apply_function(self.net, x, batch_size=2**16)
        y = stencil_to_numpy(y, q.shape[-2], q.shape[-1])

        return y.astype('float64')
    
    def predict(self, ds, M=1000):
        '''
        ds - standard dataset of
        run x time x nlev x ny x nx

        Output: dataset with three variables:
        q_forcing_advection
        q_forcing_advection_mean
        q_forcing_advection_var
        '''
        X = stack_images(ds.q)
        
        XX = xarray_to_stencil(X, self.stencil_size) / self.x_scale

        Y = self.y_scale * apply_function(self.net, XX, batch_size=2**16)
        Y = stencil_to_numpy(Y, X.shape[-2], X.shape[-1]) + 0*X
        Y = Y.unstack().transpose('run', 'time', 'lev', 'y', 'x')
        
        return xr.Dataset({'q_forcing_advection': Y, 
                           'q_forcing_advection_mean': Y, 'q_forcing_advection_var': Y*0})