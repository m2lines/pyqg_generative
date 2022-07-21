import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import torch.optim as optim
from time import time
import os

from pyqg_generative.tools.cnn_tools import AndrewCNN, DCGAN_discriminator, \
    train, apply_function, extract, prepare_PV_data, weights_init, minibatch, \
    AverageLoss
from pyqg_generative.tools.computational_tools import subgrid_scores
from pyqg_generative.models.parameterization import Parameterization
from pyqg_generative.tools.operators import coord

LAMBDA_DRIFT = 1e-3
LAMBDA_GP = 10

class CGANRegression(Parameterization):
    def __init__(self):
        # 2 Input layers of q
        n_in = 2
        # 2 Input layers of noise
        self.n_latent = 2
        # 2 Output layers of q_forcing_advection
        n_out = 2

        self.G = AndrewCNN(n_in+self.n_latent,n_out)
        # Note minibatch discrimination (2*n_out)
        self.D = DCGAN_discriminator(n_in+2*n_out, bn='None')
        self.mean = AndrewCNN(n_in, n_out)

        self.G.apply(weights_init)
        self.D.apply(weights_init)
    
    def fit(self, ds_train, ds_test, num_epochs=50, 
        batch_size=64, learning_rate=2e-4, nruns=5):

        X_train, Y_train, X_test, Y_test, self.x_scale, self.y_scale = \
            prepare_PV_data(ds_train, ds_test)
        
        train(self.mean,
            X_train, Y_train,
            X_test, Y_test,
            num_epochs, 64, 0.001)
        
        Y_mean = apply_function(self.mean, X_train)
        
        train_CGAN(self, ds_train, ds_test,
            X_train, Y_train, Y_mean, num_epochs, batch_size, learning_rate, 
            nruns)

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
    
    def generate_latent_noise(self, ny, nx):
        return np.random.randn(1, self.n_latent, ny, nx).astype('float32')
    
    def predict_snapshot(self, m, noise):
        X = self.x_scale.normalize(m.q.astype('float32'))
        return self.y_scale.denormalize(
            apply_function(self.G, X, noise, fun=self.generate) +
            apply_function(self.mean, X)
        ).squeeze().astype('float64')
    
    def predict(self, ds, M=100):
        X = self.x_scale.normalize(extract(ds, 'q'))
        Y, mean, var = apply_function(self.G, X, fun=self.generate_mean_var, M=M)
        Y_mean = apply_function(self.mean, X)

        Y = xr.DataArray(self.y_scale.denormalize(Y+Y_mean).reshape(ds.q.shape),
            dims=['run', 'time', 'lev', 'y', 'x'])
        mean = xr.DataArray(self.y_scale.denormalize(mean+Y_mean).reshape(ds.q.shape),
            dims=['run', 'time', 'lev', 'y', 'x'])
        var = xr.DataArray(self.y_scale.denormalize_var(var).reshape(ds.q.shape),
            dims=['run', 'time', 'lev', 'y', 'x'])

        return xr.Dataset({'q_forcing_advection': Y, 
            'q_forcing_advection_mean': mean, 'q_forcing_advection_var': var})

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
    num_epochs = len(optim_loss['G_loss'])
    epoch = coord(np.arange(1, num_epochs+1), 'epoch')
    for key in optim_loss.keys():
        optim_loss[key] = xr.DataArray(optim_loss[key], dims='epoch', coords=[epoch])
    
    ds = xr.Dataset(optim_loss)
    ds.update(xr.concat(log_train, dim='epoch'))
    ds.update(xr.concat(log_test, dim='epoch').rename(
        dict(L2_mean='L2_mean_test', L2_total='L2_total_test', 
        L2_residual='L2_residual_test')))
    loss = ds.L2_mean_test + ds.L2_total_test + ds.L2_residual_test
    ds['loss'] = loss
    Epoch_opt = loss.idxmin()
    return ds, int(Epoch_opt)

def train_CGAN(net, ds_train, ds_test,
    X_train, Y_train, Y_mean,
    num_epochs, batch_size, learning_rate, nruns=5):
    '''
    net - an instance of class CGANModel
    nruns - number of runs used in test and train
    datasets to evaluate prediction on the fly
    '''
    os.system('mkdir -p checkpoints')

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

            net.D.zero_grad()
            yfake1 = net.generate(xtrue) + ymean.to(device)
            yfake2 = net.generate(xtrue) + ymean.to(device)

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
        
        torch.save(net.G.state_dict(),f'checkpoints/G_{epoch+1}.pt')
        
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
    stats, epoch = loss_to_xarray(optim_loss, log_train, log_test)
    print('Optimal epoch is', epoch)
    file = f'checkpoints/G_{epoch}.pt'
    print(file,' is loaded')
    net.G.load_state_dict(torch.load(file, map_location='cpu'))
    os.system('rm checkpoints/*.pt')
    torch.save(net.G.state_dict(), 'checkpoints/G_final.pt')
    stats.to_netcdf('checkpoints/stats.nc')