import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import torch.optim as optim
from time import time
from os.path import exists
import os

from pyqg_generative.tools.cnn_tools import AndrewCNN, ChannelwiseScaler, log_to_xarray, save_model_args, \
    DCGAN_discriminator, train, apply_function, extract, prepare_PV_data, weights_init, minibatch, \
    AverageLoss
from pyqg_generative.tools.deep_inversion import DeepInversionGenerator
from pyqg_generative.tools.computational_tools import subgrid_scores
from pyqg_generative.models.parameterization import Parameterization
from pyqg_generative.tools.operators import coord

LAMBDA_DRIFT = 1e-3
LAMBDA_GP = 10

class CGANRegression(Parameterization):
    def __init__(self, regression='full_loss', nx=64, generator='Andrew', folder='model'):
        '''
        Regression parameter:
        'None': predict full subgrid forcing
        'full_loss': predict residual of subgrid forcing, 
        but loss function is the same as in initial problem
        'residual_loss': predict residual of subgrid forcing, 
        but loss function is for residual
        '''
        # 2 Input layers of q
        n_in = 2
        # 2 Input layers of noise
        self.n_latent = 2
        # 2 Output layers of q_forcing_advection
        n_out = 2

        self.regression = regression
        self.generator = generator
        self.nx = nx

        if generator == 'Andrew':
            self.G = AndrewCNN(n_in+self.n_latent,n_out)
        elif generator == 'DeepInversion':
            self.G = DeepInversionGenerator(n_in+self.n_latent,n_out)
        else:
            raise ValueError('generator not implemented')
        # Note minibatch discrimination (2*n_out)
        self.D = DCGAN_discriminator(n_in+2*n_out, bn='None', nx=nx)

        if regression != 'None':
            self.net_mean = AndrewCNN(n_in, n_out)

        self.G.apply(weights_init)
        self.D.apply(weights_init)

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
            *train_CGAN(self, ds_train, ds_test,
            X_train, Y_train, num_epochs, batch_size, learning_rate, 
            nruns)
        )
    
    def save_model(self, optim_loss, log_train, log_test):
        stats, epoch = loss_to_xarray(optim_loss, log_train, log_test)
        stats.to_netcdf('model/stats.nc')
        if self.regression != 'None':
            log_to_xarray(self.net_mean.log_dict).to_netcdf('model/stats_mean.nc')
            print('Read generator/discriminator from optimal epoch ', epoch)
            self.G.load_state_dict(torch.load(f'model/G_{epoch}.pt', map_location='cpu'))
            self.D.load_state_dict(torch.load(f'model/D_{epoch}.pt', map_location='cpu'))
        else:
            print('The Last epoch is used for prediction')

        #os.system('rm model/G_*.pt')
        #os.system('rm model/D_*.pt')

        torch.save(self.G.state_dict(), 'model/G.pt')
        torch.save(self.D.state_dict(), 'model/D.pt')
        if self.regression != 'None':
            torch.save(self.net_mean.state_dict(), 'model/net_mean.pt')
        self.x_scale.write('x_scale.json')
        self.y_scale.write('y_scale.json')
        save_model_args('CGANRegression', regression=self.regression, nx=self.nx, generator=self.generator)

    def load_model(self, folder):
        if exists(f'{folder}/G.pt'):
            print(f'reading CGANRegression from {folder}')
            self.G.load_state_dict(
                torch.load(f'{folder}/G.pt', map_location='cpu'))
            self.D.load_state_dict(
                torch.load(f'{folder}/D.pt', map_location='cpu'))
            if self.regression != 'None':
                self.net_mean.load_state_dict(
                    torch.load(f'{folder}/net_mean.pt', map_location='cpu'))
            self.x_scale = ChannelwiseScaler().read('x_scale.json', folder)
            self.y_scale = ChannelwiseScaler().read('y_scale.json', folder)

    def generate(self, x, z=None):
        dims = (x.shape[0], self.n_latent, x.shape[2], x.shape[3])
        if z is None:
            z = torch.randn(dims, device=x.device)
        return self.G(torch.cat([x,z], dim=1))

    def generate_mean_var(self, x, M):
        y = []
        for m in range(M):
            y.append(self.generate(x))
        yfake = torch.stack(y, dim=0)

        return yfake[0], yfake.mean(dim=0), yfake.var(dim=0)

    def generate_ensemble(self, x, M):
        y = []
        for m in range(M):
            y.append(self.generate(x))
        yfake = torch.stack(y, dim=0)
        return yfake
    
    def generate_latent_noise(self, ny, nx):
        return np.random.randn(1, self.n_latent, ny, nx).astype('float32')
    
    def predict_snapshot(self, m, noise):
        X = self.x_scale.normalize(m.q.astype('float32'))
        Y = apply_function(self.G, X, noise, fun=self.generate)
        if self.regression != 'None':
            Y += apply_function(self.net_mean, X)
        return self.y_scale.denormalize(Y).squeeze().astype('float64')
    
    def predict(self, ds, M=1000):
        X = self.x_scale.normalize(extract(ds, 'q'))
        Y, mean, var = apply_function(self.G, X, fun=self.generate_mean_var, M=M)
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
    
    def predict_ensemble(self, ds, M=1000):
        X = self.x_scale.normalize(extract(ds, 'q'))
        Y = apply_function(self.G, X, fun=self.generate_ensemble, M=M)

        return Y
        
def gradient_penalty(net, xtrue, ytrue, yfake1, 
    yfake2):
    batch_size = xtrue.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=xtrue.device)

    rand_num = np.random.randint(0, 2, 1)
    if rand_num==0:
        ytrue_cat = torch.cat((ytrue, yfake2.detach()), dim=1)
    elif rand_num==1:
        ytrue_cat = torch.cat((yfake1.detach(), ytrue), dim=1)
    yfake_cat = torch.cat((yfake1.detach(), yfake2.detach()), dim=1)

    yinterp = epsilon * ytrue_cat + (1-epsilon) * yfake_cat
    yinterp.requires_grad = True

    D = net.D(torch.cat((xtrue, yinterp), dim=1))

    dDdy = torch.autograd.grad(
        outputs=D, inputs=yinterp,
        grad_outputs=torch.ones_like(D),
        retain_graph=True, create_graph=True)[0]

    dDdy = dDdy.view(batch_size, -1) # Nbatch x All other dims
    # This norm implements sqrt(sum(x**2))
    grad_loss = LAMBDA_GP * torch.mean((torch.linalg.norm(dDdy,2,dim=1)-1)**2)
    return grad_loss

def evaluate_prediction(net, ds, nruns=None, M=16):
    idx=np.arange(ds.run.size)
    if nruns is not None and nruns<len(idx):
        idx=np.random.choice(idx, nruns, replace=False)
    ds = ds.isel(run=idx)

    preds = net.predict(ds, M=M)
    return subgrid_scores(ds['q_forcing_advection'], 
                   preds['q_forcing_advection_mean'],
                   preds['q_forcing_advection']) \
                   [['L2_mean', 'L2_total', 'L2_residual']]

def loss_to_xarray(optim_loss, log_train, log_test):
    ds = log_to_xarray(optim_loss)
    ds.update(xr.concat(log_train, dim='epoch'))
    ds.update(xr.concat(log_test, dim='epoch').rename(
        dict(L2_mean='L2_mean_test', L2_total='L2_total_test', 
        L2_residual='L2_residual_test')))
    ds['loss'] = ds.L2_total_test + ds.L2_residual_test
    Epoch_opt = (ds.loss).idxmin()
    ds['Epoch_opt'] = Epoch_opt
    return ds, int(Epoch_opt)

def train_CGAN(net, ds_train, ds_test,
    X_train, Y_train,
    num_epochs, batch_size, learning_rate, nruns=5):
    '''
    net - an instance of class CGANModel
    nruns - number of runs used in test and train
    datasets to evaluate prediction on the fly
    '''
    os.system('mkdir -p model')

    if net.regression != 'None':
        Y_mean = apply_function(net.net_mean, X_train)
    else:
        Y_mean = 0*Y_train

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.G.to(device); net.D.to(device)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Training starts on device {device_name}, number of samples {len(X_train)}")

    net.G.train(); net.D.train()

    optimizerD = optim.Adam(net.D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(net.G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.5)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.5)

    optim_loss = {}
    log_train = []
    log_test = []

    # Training Loop
    t_s = time()
    for epoch in range(num_epochs):
        t_e = time()
        logger = AverageLoss(optim_loss)
        for i, (x,y,ymean) in enumerate(minibatch(X_train, Y_train, Y_mean, batch_size=batch_size)):
            xtrue = x.to(device)
            ytrue = y.to(device)
            if net.regression == 'residual_loss':
                ytrue -= ymean.to(device)

            net.D.zero_grad()
            yfake1 = net.generate(xtrue)
            yfake2 = net.generate(xtrue)

            if net.regression == 'full_loss':
                yfake1 += ymean.to(device)
                yfake2 += ymean.to(device)

            Dtrue1 = net.D(torch.cat([xtrue, ytrue, yfake2.detach()], dim=1))
            Dtrue2 = net.D(torch.cat([xtrue, yfake1.detach(), ytrue], dim=1))
            Dfake  = net.D(torch.cat([xtrue, yfake1.detach(), yfake2.detach()], dim=1))
            D_loss = - 0.5 * (Dtrue1.mean() + Dtrue2.mean()) + Dfake.mean()
            D_drift = LAMBDA_DRIFT * (Dtrue1**2).mean()
            D_grad = gradient_penalty(net, xtrue, ytrue, yfake1, yfake2)

            error = D_loss + D_grad + D_drift
            error.backward()
            optimizerD.step()

            if i % 5 == 0:
                net.G.zero_grad()
                G_loss = - (net.D(torch.cat([xtrue, yfake1, yfake2], dim=1))).mean()
                G_loss.backward()

                optimizerG.step()

            logger.accumulate(optim_loss, 
                {
                    'D_loss': D_loss,
                    'D_grad': D_grad,
                    'D_drift': D_drift,
                    'G_loss': G_loss,
                }, len(x))
            
        schedulerD.step()
        schedulerG.step()
        logger.average(optim_loss)

        log_train.append(evaluate_prediction(net, ds_train, nruns))
        log_test.append(evaluate_prediction(net, ds_test, nruns))
        
        torch.save(net.G.state_dict(),f'model/G_{epoch+1}.pt')
        torch.save(net.D.state_dict(),f'model/D_{epoch+1}.pt')
        
        t = time()
        
        print('[%d/%d] [%.2f/%.2f] D_loss: %.2f G_loss: %.2f L2_mean: [%.3f,%.3f] L2_total: [%.3f,%.3f] L2_res: [%.3f,%.3f]'
                  % (epoch+1, num_epochs,
                     t-t_e, (t-t_s)*(num_epochs/(epoch+1)-1),
                     optim_loss['D_loss'][-1], 
                     optim_loss['G_loss'][-1],
                     log_train[-1]['L2_mean'], log_test[-1]['L2_mean'],
                     log_train[-1]['L2_total'], log_test[-1]['L2_total'],
                     log_train[-1]['L2_residual'], log_test[-1]['L2_residual']
                     ))
    return optim_loss, log_train, log_test