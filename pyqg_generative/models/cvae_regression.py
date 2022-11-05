import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import torch.optim as optim
from time import time
from os.path import exists
import os

from pyqg_generative.tools.cnn_tools import AndrewCNN, ChannelwiseScaler, log_to_xarray, save_model_args, \
    train, apply_function, extract, prepare_PV_data, minibatch, \
    AverageLoss
from pyqg_generative.tools.computational_tools import subgrid_scores
from pyqg_generative.models.parameterization import Parameterization
from pyqg_generative.tools.operators import coord

class CVAERegression(Parameterization):
    def __init__(self, regression='full_loss', decoder_var = 'adaptive', folder='model', div=False):
        '''
        Regression parameter:
        'None': predict full subgrid forcing
        'full_loss': predict residual of subgrid forcing, 
        but loss function is the same as in initial problem
        '''
        # 2 Input layers of q
        n_in = 2
        # 2 Input layers of noise
        self.n_latent = 2
        # 2 Output layers of q_forcing_advection
        n_out = 2

        self.regression = regression
        self.decoder_var = decoder_var
        self.div = div

        # Decoder (x,z -> y) is identical to the Generator of GAN model
        self.decoder = AndrewCNN(n_in+self.n_latent,n_out, div=div)

        # Encoder (x,y -> z)
        self.encoder = AndrewCNN(n_in+n_out, 2*self.n_latent)
        if regression != 'None':
            self.net_mean = AndrewCNN(n_in, n_out, div=div)

        self.load_model(folder)
    
    def fit(self, ds_train, ds_test, num_epochs=50, num_epochs_regression=50, 
        batch_size=64, learning_rate=2e-4, nruns=5):

        X_train, Y_train, X_test, Y_test, self.x_scale, self.y_scale = \
            prepare_PV_data(ds_train, ds_test)
        
        if self.regression != 'None':
            train(self.net_mean,
                X_train, Y_train,
                X_test, Y_test,
                num_epochs_regression, batch_size, 0.001)
        
        self.save_model(
            *train_CVAE(self, ds_train, ds_test,
            X_train, Y_train, num_epochs, batch_size, learning_rate, 
            nruns)
        )
    
    def save_model(self, optim_loss, log_train, log_test):
        stats, epoch = loss_to_xarray(optim_loss, log_train, log_test)
        stats.to_netcdf('model/stats.nc')
        if self.regression != 'None':
            log_to_xarray(self.net_mean.log_dict).to_netcdf('model/stats_mean.nc')
        print('Optimal epoch:', epoch)
        print('The Last epoch is used for prediction')

        torch.save(self.encoder.state_dict(), 'model/encoder.pt')
        torch.save(self.decoder.state_dict(), 'model/decoder.pt')
        if self.regression != 'None':
            torch.save(self.net_mean.state_dict(), 'model/net_mean.pt')
        self.x_scale.write('x_scale.json')
        self.y_scale.write('y_scale.json')
        save_model_args('CVAERegression', regression=self.regression, div=self.div, decoder_var=self.decoder_var)

    def load_model(self, folder):
        if exists(f'{folder}/encoder.pt'):
            print(f'reading CVAERegression from {folder}')
            self.encoder.load_state_dict(
                torch.load(f'{folder}/encoder.pt', map_location='cpu'))
            self.decoder.load_state_dict(
                torch.load(f'{folder}/decoder.pt', map_location='cpu'))
            if self.regression != 'None':
                self.net_mean.load_state_dict(
                    torch.load(f'{folder}/net_mean.pt', map_location='cpu'))
            self.x_scale = ChannelwiseScaler().read('x_scale.json', folder)
            self.y_scale = ChannelwiseScaler().read('y_scale.json', folder)

    def encode(self, x, y):
        # stack input-output and encode
        result = self.encoder(torch.cat([x,y], dim=1))

        # half of channels for mean, and half for pseudo-var
        mu = result[:,:self.n_latent,:,:]
        logvar = result[:,self.n_latent:,:,:]

        return mu, logvar

    def generate(self, x, z=None):
        dims = (x.shape[0], self.n_latent, x.shape[2], x.shape[3])
        if z is None:
            z = torch.randn(dims, device=x.device)
        return self.decoder(torch.cat([x,z], dim=1))

    def generate_mean_var(self, x, M):
        y = []
        for m in range(M):
            y.append(self.generate(x))
        yfake = torch.stack(y, dim=0)

        return yfake[0], yfake.mean(dim=0), yfake.var(dim=0)

    def generate_latent_noise(self, ny, nx):
        return np.random.randn(1, self.n_latent, ny, nx).astype('float32')
    
    def predict_snapshot(self, m, noise):
        X = self.x_scale.normalize(m.q.astype('float32'))
        Y = apply_function(self.decoder, X, noise, fun=self.generate)
        if self.regression != 'None':
            Y += apply_function(self.net_mean, X)
        return self.y_scale.denormalize(Y).squeeze().astype('float64')
    
    def predict(self, ds, M=1000):
        X = self.x_scale.normalize(extract(ds, 'q'))
        Y, mean, var = apply_function(self.decoder, X, fun=self.generate_mean_var, M=M)
        if self.regression != 'None':
            mean_correction = apply_function(self.net_mean, X)
            Y += mean_correction
            mean += mean_correction

        Y = xr.DataArray(self.y_scale.denormalize(Y).reshape(ds.q.shape),
            dims=['run', 'time', 'lev', 'y', 'x'])
        mean = xr.DataArray(self.y_scale.denormalize(mean).reshape(ds.q.shape),
            dims=['run', 'time', 'lev', 'y', 'x'])
        var = xr.DataArray(self.y_scale.denormalize_var(var).reshape(ds.q.shape),
            dims=['run', 'time', 'lev', 'y', 'x'])

        return xr.Dataset({'q_forcing_advection': Y, 
            'q_forcing_advection_mean': mean, 'q_forcing_advection_var': var})

    def forward(self, x, y):
        '''
        x - inputs, i.e. PV
        y - outputs, i.e. subgrid forcing
        z - latent variable (image)
        mu, std, var, logvar are of latent variable
        '''
        mu, logvar = self.encode(x, y)
        std = torch.exp(0.5 * logvar)
        var = torch.square(std)
        eps = torch.randn_like(std)
        z = eps * std + mu
        yhat = self.generate(x, z)
        return yhat, mu, var, logvar

    def compute_loss(self, x, ytrue, ymean):
        '''
        ymean - correction given by deterministic model
        Loss = - log(P(y|z)) + D_KL(Q(z|y), P(z))
        P(y|z) - multivariate gaussian, with mean predicted by
        decoder f(z).
        Its variance is var_p, does not depend on pixel and channel
        -log(P(y|z)) = 1/(2*var_p) * sum((y_i - f_i(z))^2),
        sum over pixels and channels.
        var_p, i.e. uncertainty in prediction of pixel values
        is chosen as 
        var_p = mean((y_i-f_i(z))^2) following
        http://proceedings.mlr.press/v139/rybkin21a/rybkin21a.pdf.
        P(z) is standard normal multivariate.
        Q(z|y) is given by mu and var.
        Formula for KL divergence:
        https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/.
        D_KL(Q(z|y), P(z)) = 
        0.5 * sum(mu^2 + var - 1 - logvar),
        sum over all pixels and channels.
        '''
        yhat, mu, var, logvar = self.forward(x,ytrue)
        if self.regression != 'None':
            yhat += ymean

        # no reduction here
        KL_pointwise = 0.5 * (torch.square(mu) + var - 1 - logvar)
        MSE_pointwise = torch.square(yhat - ytrue)

        if self.decoder_var == 'adaptive':
            var_p = MSE_pointwise.mean().item()
        elif self.decoder_var == 'fixed':
            var_p = 1.
        else:
            # just specify value :) In most cases 0.1 roughly
            # corresponds to the mean MSE for subgrid forcing
            var_p = self.decoder_var

        # sum over pixels and channels and average over batch
        loss_recon = 1 / (2.*var_p) * MSE_pointwise.sum(dim=(1,2,3)).mean()
        loss_KL = KL_pointwise.sum(dim=(1,2,3)).mean()

        loss = loss_recon + loss_KL

        # additional metrics
        with torch.no_grad():
            MSE = MSE_pointwise.mean()
            var_latent = var.mean()
            var_aggr = mu.var() + var_latent

        return {'loss': loss, 'loss_recon': loss_recon, 'loss_KL': loss_KL, 
            'MSE': MSE, 'var_latent': var_latent, 'var_aggr': var_aggr}

def evaluate_prediction(net, ds, nruns=None, M=16):
    idx=np.arange(ds.run.size)
    if nruns is not None and nruns<len(idx):
        idx=np.random.choice(idx, nruns, replace=False)
    ds = ds.isel(run=idx)

    preds = net.predict(ds, M=M)
    return subgrid_scores(ds['q_forcing_advection'], 
                   preds['q_forcing_advection_mean'],
                   preds['q_forcing_advection']) \
                   [['L2_mean', 'L2_total', 'L2_residual','var_ratio']]

def loss_to_xarray(optim_loss, log_train, log_test):
    ds = log_to_xarray(optim_loss)
    ds.update(xr.concat(log_train, dim='epoch'))
    ds.update(xr.concat(log_test, dim='epoch').rename(
        dict(L2_mean='L2_mean_test', L2_total='L2_total_test', 
        L2_residual='L2_residual_test')))
    ds['L2_loss'] = ds.L2_total_test + ds.L2_residual_test
    Epoch_opt = (ds.L2_loss).idxmin()
    ds['Epoch_opt'] = Epoch_opt
    return ds, int(Epoch_opt)

def train_CVAE(net, ds_train, ds_test,
        X_train, Y_train,
        num_epochs, batch_size, learning_rate, nruns=5):
    '''
    net - an instance of class CVAERegression
    nruns - number of runs used in test and train
    datasets to evaluate prediction on the fly
    '''
    from itertools import chain # to concatenate generators

    os.system('mkdir -p model')

    if net.regression != 'None':
        Y_mean = apply_function(net.net_mean, X_train)
    else:
        Y_mean = 0*Y_train

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.decoder.to(device)
    net.encoder.to(device)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Training starts on device {device_name}, number of samples {len(X_train)}")

    net.encoder.train()
    net.decoder.train()
    
    optimizer = optim.Adam(chain(net.encoder.parameters(), net.decoder.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)  

    optim_loss = {}
    log_train = []
    log_test = []
        
    t_s = time()
    for epoch in range(0,num_epochs):
        t_e = time()
        logger = AverageLoss(optim_loss)
        for x, y, ymean in minibatch(X_train, Y_train, Y_mean, batch_size=batch_size):
            optimizer.zero_grad()
            losses = net.compute_loss(x.to(device),y.to(device),ymean.to(device))
            losses['loss'].backward() # optimize over the 'loss' value
            optimizer.step()
            logger.accumulate(optim_loss, losses, len(x))
        scheduler.step()

        logger.average(optim_loss)
        
        log_train.append(evaluate_prediction(net, ds_train, nruns))
        log_test.append(evaluate_prediction(net, ds_test, nruns))

        #torch.save(net.encoder.state_dict(),f'model/encoder_{epoch+1}.pt')
        #torch.save(net.decoder.state_dict(),f'model/decoder_{epoch+1}.pt')
        
        t = time()
        print('[%d/%d] [%.2f/%.2f] MSE/KL: [%.3f, %.3f] Var: [%.3f,%.3f] L2_mean: [%.3f,%.3f] L2_total: [%.3f,%.3f] L2_res: [%.3f,%.3f] Var_ratio: [%.3f, %.3f]' 
            % (epoch+1, num_epochs,
             t-t_e, (t-t_s)*(num_epochs/(epoch+1)-1),
             optim_loss['MSE'][-1], optim_loss['loss_KL'][-1],
             optim_loss['var_latent'][-1], optim_loss['var_aggr'][-1],
             log_train[-1]['L2_mean'], log_test[-1]['L2_mean'],
             log_train[-1]['L2_total'], log_test[-1]['L2_total'],
             log_train[-1]['L2_residual'], log_test[-1]['L2_residual'],
             log_train[-1]['var_ratio'][0], log_train[-1]['var_ratio'][1]
            ))

    return optim_loss, log_train, log_test