import sys; sys.path.insert(0, '../')

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from time import time

from tools.cnn_tools import *
from tools.deep_inversion import DeepInversionGenerator, DeepInversionDiscriminator
from tools.computational_tools import *

class CGAN(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_latent: int, 
            minibatch_discrimination, deterministic, loss_type, 
            lambda_MSE_mean, lambda_MSE_sample, ncritic, training,
            generator, discriminator, bn, GP_shuffle, regression):
        '''
        n_in - number of channels in input image
        n_out - number of channels in output
        n_latent - number of channels in latent space
        in terms of images.
        There is no coarsening/refining of the resolution in generator
        minibatch discrimination:
            https://github.com/birajaghoshal/posterior-sampling-UQ
            https://arxiv.org/pdf/1811.05910.pdf
        loss_type in ('GAN', 'WGAN')
        lambda_MSE_mean - regularization on sampled conditional meanЖ
            https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Ohayon_High_Perceptual_Quality_Image_Denoising_With_a_Posterior_Sampling_CGAN_ICCVW_2021_paper.pdf
        lambda_MSE_sample - regularization on individual samples
            https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf
        ncritic - number of iterations of discriminator before 1 iteration of generator
        training = {'DCGAN', 'DeepInversion'} choose parameters of Adam optimizer
        generator = {'Andrew', 'DeepInversion'}
        discriminator = {'DCGAN', 'DeepInversion'}
        bn = {'BatchNorm', 'LayerNorm', 'None'} - batch normalization option for discriminator
        GP_shuffle = {True/False} If True, compute GP as in code https://github.com/birajaghoshal/posterior-sampling-UQ/blob/master/train.py#L40
        otherwise compute GP as in original paper
        regression = {True/False}. If True, deterministic part is predicted with additional CNN. 
        '''
        super().__init__()

        self.n_latent = n_latent
        self.minibatch_discrimination = minibatch_discrimination
        self.deterministic = deterministic
        self.loss_type = loss_type
        self.lambda_MSE_mean = lambda_MSE_mean
        self.lambda_MSE_sample = lambda_MSE_sample
        self.GP_shuffle = GP_shuffle
        self.regression = regression

        if training == 'DCGAN':
            # see also https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/blob/master/train.py#L104
            self.adam_kw = {'betas': (0.5, 0.999)}
        elif training == 'DeepInversion':
            self.adam_kw = {'betas': (0.5, 0.9), 'weight_decay': 0.0001}
        else:
            print('Wrong training parameter')

        if ncritic is None:
            ncritic = 1 if loss_type == 'GAN' else 5
        self.ncritic = ncritic

        if generator == 'Andrew':
            self.G = AndrewCNN(n_in+n_latent, n_out)
        elif generator == 'DeepInversion':
            self.G = DeepInversionGenerator(n_in+n_latent, n_out)
        else:
            print('Wrong generator parameter')

        D_channels = n_in+2*n_out if minibatch_discrimination else n_in+n_out

        if discriminator == 'DCGAN':
            self.D = DCGAN_discriminator(D_channels, bn=bn)
        elif discriminator == 'DeepInversion':
            self.D = DeepInversionDiscriminator(D_channels, bn=bn)
        else:
            print('Wrong discriminator parameter')

        if regression:
            self.reg_net = AndrewCNN(n_in, n_out)
        
        self.G.apply(weights_init)
        self.D.apply(weights_init)
    
    def generate(self, x: torch.tensor) -> torch.tensor:
        '''
        mapping (x,z) -> y
        '''
        dims = (x.shape[0], self.n_latent, x.shape[2], x.shape[3])
        fun = torch.zeros if self.deterministic else torch.randn
        z = fun(dims, device=x.device)
        return self.G(torch.cat([x,z], dim=1))
    
    def generate_full(self, x):
        y = self.generate(x)
        if self.regression:
            y += self.reg_net(x)
        return y

    def discriminate(self, *x: torch.tensor) -> torch.tensor:
        '''
        Takes arbitrary number of fields and returns "critic",
        i.e. value before sigmoid is applied
        '''
        return self.D(torch.cat([*x], dim=1))

    def f_transform(self, D):
        '''
        See see https://arxiv.org/pdf/1901.08753.pdf for
        definition of f, g, h function
        Note that for f and g we change sign as we want to minimize
        D - output of discriminator before sigmoid
        Returns single value for minibatch - averaged loss
        For stability, can be further improved by combining 
        both operations into one, see:
        https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/blob/master/torchprob/gan/loss.py#L8
        '''
        if self.loss_type=='GAN':
            criterion = nn.BCELoss()
            activation = nn.Sigmoid()
            return criterion(activation(D), torch.ones_like(D))
        if self.loss_type=='WGAN':
            return -D.mean()
    
    def g_transform(self, D):
        if self.loss_type=='GAN':
            criterion = nn.BCELoss()
            activation = nn.Sigmoid()
            return criterion(activation(D), torch.zeros_like(D))
        if self.loss_type=='WGAN':
            return D.mean()

    def h_transform(self, D):
        if self.loss_type=='GAN':
            criterion = nn.BCELoss()
            activation = nn.Sigmoid()
            # Generator tries to fool discriminator, i.e.
            # the loss is the loss for "true" images
            return criterion(activation(D), torch.ones_like(D))
        if self.loss_type=='WGAN':
            return -D.mean()

    def generator_loss(self, xtrue, yfake):
        '''
        here xtrue is torch tensor as usual,
        yfake is tensor or list of tensors (in case of minibatch discrimination)
        '''
        if isinstance(yfake,list):
            Dfake = self.discriminate(xtrue, *yfake)
        else:
            Dfake = self.discriminate(xtrue, yfake)

        return self.h_transform(Dfake)

    def discriminator_loss(self, xtrue, ytrue):
        if self.minibatch_discrimination is False:
            yfake = self.generate(xtrue)

            Dtrue = self.discriminate(xtrue, ytrue)
            Dfake = self.discriminate(xtrue, yfake.detach())
            
            loss = self.f_transform(Dtrue) + self.g_transform(Dfake)
            
            return loss, yfake, Dtrue
        else:
            yfake1 = self.generate(xtrue)
            yfake2 = self.generate(xtrue)

            Dtrue1 = self.discriminate(xtrue, ytrue, yfake2.detach())
            Dtrue2 = self.discriminate(xtrue, yfake1.detach(), ytrue)
            Dfake  = self.discriminate(xtrue, yfake1.detach(), yfake2.detach())

            loss = 0.5 * self.f_transform(Dtrue1) + 0.5 * self.f_transform(Dtrue2) + \
                         self.g_transform(Dfake)

            return loss, [yfake1, yfake2], [Dtrue1, Dtrue2]

    def drift_penalty(self, Dtrue, lambda_drift=1e-3):
        '''
        https://github.com/birajaghoshal/posterior-sampling-UQ/blob/master/train.py#L53
        lambda_drift see https://arxiv.org/pdf/1811.05910.pdf appendix D.3
        '''
        if self.loss_type=='GAN':
            return torch.zeros(1,requires_grad=True).mean()
        
        if self.loss_type=='WGAN':
            if isinstance(Dtrue, list):
                Dtrue = 0.5 * (Dtrue[0] + Dtrue[1])
            
            return lambda_drift * (Dtrue**2).mean()

    def gradient_penalty(self, xtrue, ytrue, yfake, lambda_gp=10):
        '''
        https://github.com/birajaghoshal/posterior-sampling-UQ/blob/master/utils.py#L4
        https://github.com/EmilienDupont/wgan-gp/blob/master/training.py#L73
        https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L137,
        
        lambda_gp see https://arxiv.org/pdf/1811.05910.pdf appendix D.3

        Discriminator should not contain batch norm:
        https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
        https://github.com/EmilienDupont/wgan-gp/blob/master/models.py#L47
        https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L68
        https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L74
        Instead Layernorm is used:
        https://stackoverflow.com/questions/69177155/the-images-generated-by-wgan-gp-looks-very-gray
        https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/blob/master/v0/models_64x64.py#L98
        https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py#L88
        https://github.com/igul222/improved_wgan_training/blob/master/tflib/ops/layernorm.py#L19
        '''
        if self.loss_type=='GAN':
            return torch.zeros(1,requires_grad=True).mean()

        if self.loss_type=='WGAN':
            batch_size = xtrue.shape[0]
            epsilon = torch.rand(batch_size, 1, 1, 1, device=xtrue.device)

            if self.minibatch_discrimination is False:
                yinterp = epsilon * ytrue + (1-epsilon) * yfake.detach() # detach because we want to compute grad later
                yinterp.requires_grad = True
                
                # No f_transform here to avoid mean over batch
                # No minus here as it does not matter for norm
                D = self.D(torch.cat((xtrue, yinterp), dim=1))

                dDdy = torch.autograd.grad(outputs=D, inputs=yinterp, 
                                           grad_outputs=torch.ones_like(D), # As D is not scalar, see https://github.com/birajaghoshal/posterior-sampling-UQ/blob/master/utils.py#L16
                                           retain_graph=True, create_graph=True)[0]
            else:
                #https://github.com/birajaghoshal/posterior-sampling-UQ/blob/master/train.py#L40
                if self.GP_shuffle:
                    rand_num = np.random.randint(0, 2, 1)
                    if rand_num==0:
                        ytrue_cat = torch.cat((ytrue, yfake[1].detach()), dim=1)
                    elif rand_num==1:
                        ytrue_cat = torch.cat((yfake[0].detach(), ytrue), dim=1)

                    yfake_cat = torch.cat((yfake[0].detach(), yfake[1].detach()), dim=1)

                    yinterp = epsilon * ytrue_cat + (1-epsilon) * yfake_cat
                    yinterp.requires_grad = True

                    D = self.D(torch.cat((xtrue, yinterp), dim=1))

                    dDdy = torch.autograd.grad(outputs=D, inputs=yinterp,
                                            grad_outputs=torch.ones_like(D),
                                            retain_graph=True, create_graph=True)[0]
                    
                else:
                    ytrue1 = torch.cat((ytrue, yfake[1].detach()), dim=1)
                    ytrue2 = torch.cat((yfake[0].detach(), ytrue), dim=1)
                    yfake_cat = torch.cat((yfake[0].detach(), yfake[1].detach()), dim=1)
                    
                    yinterp1 = epsilon * ytrue1 + (1-epsilon) * yfake_cat
                    yinterp2 = epsilon * ytrue2 + (1-epsilon) * yfake_cat

                    yinterp1.requires_grad = True
                    yinterp2.requires_grad = True

                    D1 = 0.5 * self.D(torch.cat((xtrue, yinterp1), dim=1))
                    D2 = 0.5 * self.D(torch.cat((xtrue, yinterp2), dim=1))

                    dDdy = torch.autograd.grad(outputs=D1, inputs=yinterp1,
                                            grad_outputs=torch.ones_like(D1),
                                            retain_graph=True, create_graph=True)[0] \
                        + torch.autograd.grad(outputs=D2, inputs=yinterp2,
                                            grad_outputs=torch.ones_like(D2),
                                            retain_graph=True, create_graph=True)[0]
            
            dDdy = dDdy.view(batch_size, -1) # Nbatch x All other dims

            # This norm implements sqrt(sum(x**2))
            grad_loss = lambda_gp * torch.mean((torch.linalg.norm(dDdy,2,dim=1)-1)**2)
            return grad_loss
            
    def MSE_mean_loss(self, xtrue, ytrue, PB=8, M=16):
        '''
        Add MSE loss for sampled conditional mean
        PB - penalty batch size, i.e. number of x-y pairs to add penalty
        M - number of samples for a fixed input x
        In total PB*M=64 calls of generator (for batch size of 64 it should not significantly increase cost)
        alpha - regularization parameter
        Sample images from generator and discriminator loss are not used here
        # https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Ohayon_High_Perceptual_Quality_Image_Denoising_With_a_Posterior_Sampling_CGAN_ICCVW_2021_paper.pdf
        '''
        if self.lambda_MSE_mean==0:
            return torch.zeros(1,requires_grad=True).mean() # return zero but backward() still can be applied

        yfake = []
        for m in range(M):
            yfake.append(self.generate(xtrue[:PB]))

        yfake_mean = torch.stack(yfake,dim=0).mean(dim=0)

        criterion = nn.MSELoss()

        return self.lambda_MSE_mean*criterion(yfake_mean, ytrue[:PB])

    def MSE_sample_loss(self, ytrue, yfake):
        if self.lambda_MSE_sample==0:
            return torch.zeros(1,requires_grad=True).mean() # return zero but backward() still can be applied

        criterion = nn.MSELoss()

        if self.minibatch_discrimination:
            error1 = criterion(yfake[0], ytrue)
            error2 = criterion(yfake[1], ytrue)
            return 0.5*self.lambda_MSE_sample*(error1+error2)
        else:
            return self.lambda_MSE_sample*criterion(yfake, ytrue)

    def evaluate_test(self, X: np.array, Y: np.array, batch_size=64, postfix='', M=16, log_dict=None):
        '''
        3 Metrics in total are supposed to track convergence of GAN:
        - MSE (of mean prediction)
        - Error in spectral density of total forcing
        - Error in spectral density of residual w.r.t. mean prediction
        '''
        def batch_statistics(x, y):
            '''
            
            '''
            if self.deterministic:
                yfake = [self.generate_full(x)]
                yfake_mean = yfake[0]
            else:
                yfake = []
                for _ in range(M):
                    yfake.append(self.generate_full(x))
                yfake_mean = torch.stack(yfake, dim=0).mean(dim=0)
            
            MSE = ((yfake_mean-y)**2).mean(dim=(0,2,3))
            var = (y**2).mean(dim=(0,2,3))
            
            # Spectral analysis
            def fft_covariance(_y):
                '''
                Computes joint target-target spectral covariance density
                '''
                # Concatenate input-target matrix
                Afft = torch.fft.rfftn(_y, dim=(2,3))
                n = Afft.shape[1]
                C = torch.ones(n,n,*Afft.shape[-2:], device=_y.device)
                C = float('nan') * C
                alpha = 1. / (_y.shape[-2]*_y.shape[-1])
                for i in range(n):
                    for j in range(i,n):
                        C[i,j,:,:] = torch.mean(alpha*torch.real(Afft[:,i] * torch.conj(Afft[:,j])), dim=0)
                return C

            C_target = fft_covariance(y)
            C_fake = fft_covariance(yfake[0])
            C_target_res = fft_covariance(y-yfake_mean)
            C_fake_res = fft_covariance(yfake[0]-yfake_mean)
            return {'MSE': MSE, 'var': var, 'C_target': C_target, 'C_fake': C_fake, 'C_target_res': C_target_res, 'C_fake_res': C_fake_res}

        self.eval()
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        metrics = {}
        for x, y in minibatch(X, Y, batch_size=batch_size, shuffle=False):
            with torch.no_grad():
                out = batch_statistics(x.to(dev), y.to(dev))
                n = len(x)
                for key in out.keys():
                    val = n * out[key].cpu().numpy()
                    try:
                        metrics[key] += val
                    except:
                        metrics[key] = val
        self.train()
        for key in out.keys():
            metrics[key] /= len(X)

        sqr_mean = lambda x: np.mean(x**2, axis=(2,3))
        L2_error = lambda x,y: np.nanmean((sqr_mean(x-y)/sqr_mean(y))**0.5)

        def push(key, val):
            _key = key+postfix
            if _key not in log_dict.keys():
                log_dict[_key] = []
            log_dict[_key].append(val)
            
        # Each metric first normalized and then averaged over layers
        push('L2', np.mean((metrics['MSE'] / metrics['var'])**0.5))
        push('L2_q', L2_error(metrics['C_fake'], metrics['C_target']))
        push('L2_dq', L2_error(metrics['C_fake_res'], metrics['C_target_res']))
        
class CGANModel:
    def __init__(self, inputs: list[tuple[str,int]], targets: list[tuple[str,int]],
            n_latent=2, minibatch_discrimination=True, 
            deterministic=False, loss_type='WGAN', 
            lambda_MSE_mean=0, lambda_MSE_sample=0,
            ncritic=5, training='DCGAN',
            generator='Andrew', discriminator='DCGAN', bn='None',
            GP_shuffle=True, regression=False):
        '''
        inputs - What use to condition our CGAN
        targets - What we want to generate with our CGAN
        target_mean - ground truth for conditional mean
        target_var - ground truth for conditional variance
        '''
        n_in = len(inputs)
        n_out = len(targets)
        self.inputs = inputs
        self.targets = targets
        self.net = CGAN(n_in, n_out, n_latent,
            minibatch_discrimination, deterministic, loss_type,
            lambda_MSE_mean, lambda_MSE_sample, ncritic, training,
            generator, discriminator, bn, GP_shuffle, regression)

    def fit(self, ds_train: xr.DataArray, ds_test: xr.DataArray, 
            num_epochs=50, batch_size=64, lr=2e-4):

        X_train = dataset_to_array(ds_train, self.inputs)
        Y_train = dataset_to_array(ds_train, self.targets)

        X_test = dataset_to_array(ds_test, self.inputs)
        Y_test = dataset_to_array(ds_test, self.targets)

        x_scale = ChannelwiseScaler(X_train)
        y_scale = ChannelwiseScaler(Y_train)

        self.x_scale = x_scale
        self.y_scale = y_scale

        return train_CGAN(self.net, 
            x_scale.direct(X_train), y_scale.normalize(Y_train),
            x_scale.direct(X_test),  y_scale.normalize(Y_test),
            num_epochs, batch_size, lr)

    def predict(self, ds: xr.Dataset, ensemble_size=10, stats=None):
        '''
        ds - dataset with inputs
        returns dataset with inputs and predictions
        '''
        if stats is not None:
            Epoch = int(stats.Epoch_opt)
            try:
                self.net.load_state_dict(torch.load(f'checkpoints/net_{Epoch}.pt'))
            except:
                self.net.load_state_dict(torch.load(f'checkpoints/net_{Epoch}.pt', map_location=torch.device('cpu')))
            print(f'Prediction is given for Epoch={Epoch}')
            print('where optimal losses on validation set are:')
            print('L2+L2_q+L2_dq: %.3f, L2: %.3f L2_q: %.3f L2_dq: %.3f' %
                (stats.loss_opt, stats.L2_opt, stats.L2_q_opt, stats.L2_dq_opt))
            
        # form array of predictions
        X = self.x_scale.direct(dataset_to_array(ds, self.inputs))
        Y = self.y_scale.normalize(dataset_to_array(ds, self.targets))

        gen_ensemble = []
        for i in range(ensemble_size):
            gen_ensemble.append(self.y_scale.denormalize(
                apply_function(self.net, self.net.generate_full, X)
            ))

        gen_ensemble = np.stack(gen_ensemble, axis=0).astype('float64')
        gen_var = gen_ensemble.var(axis=0)
        gen_std = gen_ensemble.std(axis=0)
        gen_mean = gen_ensemble.mean(axis=0)
        gen = gen_ensemble[0,:]
        gen_res = gen_ensemble[0,:] - gen_mean
        true_res = dataset_to_array(ds, self.targets) - gen_mean

        del gen_ensemble

        ds_predict = extract_tuples(ds, self.targets)
        ds_predict = array_to_dataset(ds, gen.astype('float32'), self.targets, '_gen', ds_predict)
        ds_predict = array_to_dataset(ds, gen_var.astype('float32'), self.targets, '_gen_var', ds_predict)
        ds_predict = array_to_dataset(ds, gen_std.astype('float32'), self.targets, '_gen_std', ds_predict)
        ds_predict = array_to_dataset(ds, gen_mean.astype('float32'), self.targets, '_gen_mean', ds_predict)
        ds_predict = array_to_dataset(ds, gen_res.astype('float32'), self.targets, '_gen_res', ds_predict)
        ds_predict = array_to_dataset(ds, true_res.astype('float32'), self.targets, '_res', ds_predict)
        
        # Metrics used for Epoch selection
        log_dict = {}
        self.net.evaluate_test(X, Y, M=ensemble_size, log_dict=log_dict)
        ds_predict['L2_opt'] = log_dict['L2'][-1]
        ds_predict['L2_q_opt'] = log_dict['L2_q'][-1]
        ds_predict['L2_dq_opt'] = log_dict['L2_dq'][-1]
        if stats is not None:
            ds_predict.attrs = stats.attrs

        # MSE metrics
        def dims_except(*dims):
            return [d for d in ds_predict['q_forcing_advection'].dims if d not in dims]
        time = dims_except('x','y','lev')
        space = dims_except('time','lev')
        both = dims_except('lev')

        truth, pred = ds_predict.q_forcing_advection.astype('float64'), ds_predict.q_forcing_advection_gen_mean.astype('float64')
        error = (truth - pred)**2
        den = ds_predict.q_forcing_advection**2
        ds_predict['spatial_mse'] = error.mean(dim=time)
        ds_predict['temporal_mse'] = error.mean(dim=space)
        ds_predict['mse'] = error.mean(dim=both)

        ds_predict['spatial_mse_norm'] = ds_predict['spatial_mse'] / den.mean(dim=time)
        ds_predict['temporal_mse_norm'] = ds_predict['temporal_mse'] / den.mean(dim=space)
        ds_predict['mse_norm'] = ds_predict['mse'] / den.mean(dim=both)

        ds_predict['spatial_correlation'] = xr.corr(truth, pred, dim=time)
        ds_predict['temporal_correlation'] = xr.corr(truth, pred, dim=space)
        ds_predict['correlation'] = xr.corr(truth, pred, dim=both)
        
        # PDF and their metrics
        time = slice(44,None)
        Nbins = 50
        for lev in [0,1]:
            array = ds_predict['q_forcing_advection'].isel(time=time, lev=lev).values.ravel()
            m = array.mean()
            sigma = array.std()
            xmin = m-4*sigma; xmax = m+4*sigma
            density, points = PDF_histogram(array, xmin, xmax, Nbins=Nbins)
            ds_predict['PDF'+str(lev)] = xr.DataArray(density, dims='q_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                attrs={'description': f'PDF of true subgrid forcing in {lev} level', 'long_name': 'subgrid forcing PDF'})
            
            array = ds_predict['q_forcing_advection_gen'].isel(time=time, lev=lev).values.ravel()
            density, points = PDF_histogram(array, xmin, xmax, Nbins=Nbins)
            ds_predict['PDF_gen'+str(lev)] = xr.DataArray(density, dims='q_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                attrs={'description': f'PDF of generated subgrid forcing in {lev} level', 'long_name': 'subgrid forcing PDF'})
            
            array = ds_predict['q_forcing_advection_gen_mean'].isel(time=time, lev=lev).values.ravel()
            density, points = PDF_histogram(array, xmin, xmax, Nbins=Nbins)
            ds_predict['PDF_gen_mean'+str(lev)] = xr.DataArray(density, dims='q_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                attrs={'description': f'PDF of generated mean subgrid forcing in {lev} level', 'long_name': 'subgrid forcing PDF'})

        for lev in [0,1]:
            array = ds_predict['q_forcing_advection_res'].isel(time=time, lev=lev).values.ravel()
            m = array.mean()
            sigma = array.std()
            xmin = m-4*sigma; xmax = m+4*sigma
            density, points = PDF_histogram(array, xmin, xmax, Nbins=Nbins)
            ds_predict['PDF_res'+str(lev)] = xr.DataArray(density, dims='dq_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                attrs={'description': f'PDF of true subgrid forcing residual in {lev} level', 'long_name': 'subgrid forcing residual PDF'})
            
            array = ds_predict['q_forcing_advection_gen_res'].isel(time=time, lev=lev).values.ravel()
            density, points = PDF_histogram(array, xmin, xmax, Nbins=Nbins)
            ds_predict['PDF_gen_res'+str(lev)] = xr.DataArray(density, dims='dq_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                attrs={'description': f'PDF of generated subgrid forcing in {lev} level', 'long_name': 'subgrid forcing residual PDF'})
        
        # spectrum computations
        ds_predict['PSD'] = isotropic_spectrum_compute(ds, ds_predict.q_forcing_advection, 
            name='Power spectral density of $dq/dt, m/s^4$', description='Power spectrum of truth subgrid forcing')
        ds_predict['PSD_gen'] = isotropic_spectrum_compute(ds, ds_predict.q_forcing_advection_gen, 
            name='Power spectral density of $dq/dt, m/s^4$', description='Power spectrum of generated subgrid forcing')
        ds_predict['PSD_gen_mean'] = isotropic_spectrum_compute(ds, ds_predict.q_forcing_advection_gen_mean, 
            name='Power spectral density of $dq/dt, m/s^4$', description='Power spectrum of generated mean subgrid forcing')
        ds_predict['PSD_res'] = isotropic_spectrum_compute(ds, ds_predict.q_forcing_advection_res, 
            name='Power spectral density of $dq/dt, m/s^4$', description='Power spectrum of truth residual of subgrid forcing')
        ds_predict['PSD_gen_res'] = isotropic_spectrum_compute(ds, ds_predict.q_forcing_advection_gen_res, 
            name='Power spectral density of $dq/dt, m/s^4$', description='Power spectrum of generated residual of subgrid forcing')
        
        # Cross spectrum between layers
        ds_predict['CSD'] = isotropic_cross_layer_spectrum(ds, ds_predict.q_forcing_advection, 
            name='Cross layer Spectral Density of $dq/dt, m/s^4$', description='CSD of truth subgrid forcing')
        ds_predict['CSD_gen'] = isotropic_cross_layer_spectrum(ds, ds_predict.q_forcing_advection_gen, 
            name='Cross layer Spectral Density of $dq/dt, m/s^4$', description='CSD of generated subgrid forcing')
        ds_predict['CSD_gen_mean'] = isotropic_cross_layer_spectrum(ds, ds_predict.q_forcing_advection_gen_mean, 
            name='Cross layer Spectral Density of $dq/dt, m/s^4$', description='CSD of generated mean subgrid forcing')
        ds_predict['CSD_res'] = isotropic_cross_layer_spectrum(ds, ds_predict.q_forcing_advection_res, 
            name='Cross layer Spectral Density of $dq/dt, m/s^4$', description='CSD of truth residual of subgrid forcing')
        ds_predict['CSD_gen_res'] = isotropic_cross_layer_spectrum(ds, ds_predict.q_forcing_advection_gen_res, 
            name='Cross layer Spectral Density of $dq/dt, m/s^4$', description='CSD of generated residual of subgrid forcing')

        # cospectrum of subgrid forces
        ds_predict['Zflux'] = isotropic_cross_spectrum_compute(ds, ds.q, ds_predict.q_forcing_advection,
            name='Enstrophy contribution, $m/s^3$', description='Potential enstrophy rate of change spectrum for truth subgrid forcing')
        ds_predict['Zflux_gen'] = isotropic_cross_spectrum_compute(ds, ds.q, ds_predict.q_forcing_advection_gen,
            name='Enstrophy contribution, $m/s^3$', description='Potential enstrophy rate of change spectrum for generated subgrid forcing')
        ds_predict['Zflux_gen_mean'] = isotropic_cross_spectrum_compute(ds, ds.q, ds_predict.q_forcing_advection_gen_mean,
            name='Enstrophy contribution, $m/s^3$', description='Potential enstrophy rate of change spectrum for generated mean subgrid forcing')
        ds_predict['Zflux_res'] = isotropic_cross_spectrum_compute(ds, ds.q, ds_predict.q_forcing_advection_res,
            name='Enstrophy contribution, $m/s^3$', description='Potential enstrophy rate of change spectrum for truth residual of subgrid forcing')
        ds_predict['Zflux_gen_res'] = isotropic_cross_spectrum_compute(ds, ds.q, ds_predict.q_forcing_advection_gen_res,
            name='Enstrophy contribution, $m/s^3$', description='Potential enstrophy rate of change spectrum for generated residual of subgrid forcing')

        return ds_predict

def train_CGAN(net, X_train, Y_train, X_test, Y_test, 
        num_epochs, batch_size, lr):
    if net.regression:
        train(net.reg_net, X_train, Y_train, 
            X_test, Y_test, 50, batch_size, 0.001)

    # Trainer from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"Training starts on device {device}, number of samples {len(X_train)}")

    # Switch batchnorm2d layer to training mode
    net.train()

    optimizerD = optim.Adam(net.D.parameters(), lr=lr, **net.adam_kw)
    optimizerG = optim.Adam(net.G.parameters(), lr=lr, **net.adam_kw)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.5)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.5)
    
    optim_loss = {}
    log_dict = {}

    # Training Loop
    t_s = time()
    for epoch in range(num_epochs):
        t_e = time()
        logger = AverageLoss(optim_loss)
        for i, (x,y) in enumerate(minibatch(X_train, Y_train, batch_size=batch_size)):
            xtrue = x.to(device)
            ytrue = y.to(device)

            if net.regression:
                ytrue -= net.reg_net(xtrue).detach()
            
            net.D.zero_grad()
            D_loss, yfake, Dtrue = net.discriminator_loss(xtrue, ytrue)
            D_grad = net.gradient_penalty(xtrue, ytrue, yfake)
            D_drift = net.drift_penalty(Dtrue)
            error = D_loss + D_grad + D_drift
            error.backward()
            optimizerD.step()

            if i % net.ncritic == 0:
                net.G.zero_grad()
                G_loss = net.generator_loss(xtrue, yfake)
                MSE_mean = net.MSE_mean_loss(xtrue, ytrue)
                MSE_sample = net.MSE_sample_loss(ytrue, yfake)

                error = G_loss + MSE_mean + MSE_sample
                error.backward()

                optimizerG.step()

            logger.accumulate(optim_loss, 
                {
                    'D_loss': D_loss,
                    'D_grad': D_grad,
                    'D_drift': D_drift,
                    'G_loss': G_loss,
                    'MSE_mean': MSE_mean, 
                    'MSE_sample': MSE_sample
                }, len(x))
            
        schedulerD.step()
        schedulerG.step()
        logger.average(optim_loss)

        n = len(X_test) # reduce cost of evaluation on training set
        net.evaluate_test(X_test, Y_test, batch_size, postfix='_test', log_dict=log_dict)
        net.evaluate_test(X_train[:n], Y_train[:n], batch_size, postfix='', log_dict=log_dict)

        torch.save(net.state_dict(),f'checkpoints/net_{epoch+1}.pt')
        
        t = time()
        
        print('[%d/%d] [%.2f/%.2f] D_loss: %.2f G_loss: %.2f L2: [%.3f,%.3f] L2_q: [%.3f,%.3f] L2_dq: [%.3f,%.3f]'
                  % (epoch+1, num_epochs,
                     t-t_e, (t-t_s)*(num_epochs/(epoch+1)-1),
                     optim_loss['D_loss'][-1], 
                     optim_loss['G_loss'][-1],
                     log_dict['L2'][-1], log_dict['L2_test'][-1],
                     log_dict['L2_q'][-1], log_dict['L2_q_test'][-1],
                     log_dict['L2_dq'][-1], log_dict['L2_dq_test'][-1],
                     ))
    d = log_dict.copy()
    d.update(optim_loss)
    return dict_to_xarray(d)

def dict_to_xarray(d):
    ds = xr.Dataset()
    N = len(d['L2'])
    ds['L2'] = xr.DataArray(d['L2'], dims='Epoch', coords=[np.arange(1,N+1)],
        attrs={'description': 'L2 loss for conditional mean, i.e. square root of MSE', 'long_name': 'L2 loss'})
    ds['L2_test'] = xr.DataArray(d['L2_test'], dims='Epoch', 
        attrs={'description': 'L2 loss for conditional mean, i.e. square root of MSE', 'long_name': 'L2 loss'})
    ds['L2_q'] = xr.DataArray(d['L2_q'], dims='Epoch', coords=[np.arange(1,N+1)],
        attrs={'description': 'L2 loss in 2d power spectrum of subgrid forces', 'long_name': 'L2 loss in spectrum'})
    ds['L2_q_test'] = xr.DataArray(d['L2_q_test'], dims='Epoch', coords=[np.arange(1,N+1)],
        attrs={'description': 'L2 loss in 2d power spectrum of subgrid forces', 'long_name': 'L2 loss in spectrum'})
    ds['L2_dq'] = xr.DataArray(d['L2_dq'], dims='Epoch', coords=[np.arange(1,N+1)],
        attrs={'description': 'L2 loss in 2d power spectrum of subgrid forces residuals', 'long_name': 'L2 loss in spectrum residuals'})
    ds['L2_dq_test'] = xr.DataArray(d['L2_dq_test'], dims='Epoch', coords=[np.arange(1,N+1)],
        attrs={'description': 'L2 loss in 2d power spectrum of subgrid forces residuals', 'long_name': 'L2 loss in spectrum residuals'})
    for key in ['D_loss', 'G_loss', 'D_grad', 'D_drift', 'MSE_sample', 'MSE_mean']:
        ds[key] = xr.DataArray(d[key], dims='Epoch', coords=[np.arange(1,N+1)])

    loss = ds.L2_test + ds.L2_q_test + ds.L2_dq_test
    Epoch_opt = loss.idxmin()
    ds['Epoch_opt'] = Epoch_opt
    ds['loss'] = loss
    ds['loss_opt'] = loss.sel(Epoch=Epoch_opt)
    ds['L2_opt'] = ds.L2_test.sel(Epoch=Epoch_opt)
    ds['L2_q_opt'] = ds.L2_q_test.sel(Epoch=Epoch_opt)
    ds['L2_dq_opt'] = ds.L2_dq_test.sel(Epoch=Epoch_opt)
    return ds