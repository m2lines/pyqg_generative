import torch
import torch.nn as nn
import numpy as np
import pyqg
import xarray as xr

from pyqg_generative.tools.stochastic_pyqg import stochastic_QGModel
from pyqg_generative.tools.computational_tools import subgrid_scores, PDF_histogram
from pyqg_generative.tools.spectral_tools import spectrum
from pyqg_generative.tools.parameters import AVERAGE_SLICE_ANDREW
from pyqg_generative.tools.operators import coord

class Parameterization(pyqg.QParameterization):
    def generate_latent_noise():
        raise NotImplementedError
    def predict_snapshot():
        raise NotImplementedError
    def predict():
        raise NotImplementedError

    def __call__(self, m: stochastic_QGModel):
        latent_noise = lambda: self.generate_latent_noise(m.ny, m.nx)
        demean = lambda x: x - x.mean(axis=(1,2), keepdims=True)

        if m.noise_sampler.update(latent_noise):
            noise = m.noise_sampler.noise
            m.PV_forcing = demean(self.predict_snapshot(m, noise))
        
        return m.PV_forcing
    
    def test_offline(self, ds: xr.Dataset, ensemble_size=10):
        preds = self.predict(ds) # return sample, mean and var
        preds.attrs = ds.attrs
        
        target = 'q_forcing_advection'

        # shuffle true and prediction
        preds[target+'_gen'] = preds[target].copy(deep=True)
        preds[target] = ds[target].copy(deep=True)
        # var -> std
        preds[target+'_std'] = preds[target+'_var']**0.5
        # residuals
        preds[target+'_res'] = preds[target] - preds[target+'_mean']
        preds[target+'_gen_res'] = preds[target+'_gen'] - preds[target+'_mean']

        # subgrid scores
        keys = ['R2_mean', 'R2_total', 'R2_residual', \
            'L2_mean', 'L2_total', 'L2_residual']
        preds.update(
            subgrid_scores(preds[target], preds[target+'_mean'], 
                preds[target+'_gen'])[keys])

        # Andrew metrics
        def dims_except(*dims):
            return [d for d in preds[target].dims if d not in dims]
        time = dims_except('x','y','lev')
        space = dims_except('time','lev')
        both = dims_except('lev')

        true = preds[target].astype('float64')
        pred = preds[target+'_mean'].astype('float64')
        error = (true - pred)**2
        preds['spatial_mse'] = error.mean(dim=time)
        preds['temporal_mse'] = error.mean(dim=space)
        preds['mse'] = error.mean(dim=both)

        def limits(x):
            return np.minimum(np.maximum(x, -10), 1)

        preds['spatial_skill'] = limits(1 - preds['spatial_mse'] / true.var(dim=time))
        preds['temporal_skill'] = limits(1 - preds['temporal_mse'] / true.var(dim=space))
        preds['skill'] = limits(1 - preds['mse'] / true.var(dim=both))

        preds['spatial_correlation'] = xr.corr(true, pred, dim=time)
        preds['temporal_correlation'] = xr.corr(true, pred, dim=space)
        preds['correlation'] = xr.corr(true, pred, dim=both)

        preds['temporal_var_ratio'] = \
            (preds[target+'_gen_res']**2).mean(dim=space) / \
            (preds[target+'_res']**2).mean(dim=space)
        preds['var_ratio'] = \
            (preds[target+'_gen_res']**2).mean(dim=both) / \
            (preds[target+'_res']**2).mean(dim=both)

        # Spectral characteristics
        sp = spectrum()
        def sp_save(arr):
            return sp(arr,
                name='Power spectral density of $dq/dt$', units='$m/s^4$',
                description='Power spectrum of subgrid forcing'    
                )
        preds['PSD'] = sp_save(preds[target])
        preds['PSD_gen'] = sp_save(preds[target+'_gen'])
        preds['PSD_res'] = sp_save(preds[target+'_res'])
        preds['PSD_gen_res'] = sp_save(preds[target+'_gen_res'])
        preds['PSD_mean'] = sp_save(preds[target+'_mean'])

        # Cospectrum
        sp = spectrum(type='cospectrum')
        def sp_save(arr1, arr2):
            return - sp(arr1, arr2,
                name='Energy contribution', units='$m^3/s^3$',
                description='Energy contribution of subgrid forcing')
        psi = ds['psi']
        preds['Eflux'] = sp_save(psi, preds[target])
        preds['Eflux_gen'] = sp_save(psi, preds[target+'_gen'])
        preds['Eflux_res'] = sp_save(psi, preds[target+'_res'])
        preds['Eflux_gen_res'] = sp_save(psi, preds[target+'_gen_res'])
        preds['Eflux_mean'] = sp_save(psi, preds[target+'_mean'])

        # Cross layer covariances
        sp = spectrum(type='cross_layer')
        def sp_save(arr):
            return sp(arr,
                name='Cross layer covariance', units='$m/s^4$',
                description='Cross layer covariance of subgrid forcing')
        preds['CSD_res'] = sp_save(preds[target+'_res'])
        preds['CSD_gen_res'] = sp_save(preds[target+'_gen_res'])

        # PDF computations
        time = AVERAGE_SLICE_ANDREW
        Nbins = 50
        for lev in [0,1]:
            arr = preds[target].isel(time=time, lev=lev)
            mean, std = arr.mean(), arr.std()
            xmin = float(mean - 4*std); xmax = float(mean + 4*std)
            for suffix in ['', '_gen', '_mean']:
                array = preds[target+suffix].isel(time=time, lev=lev).values.ravel()
                points, density = PDF_histogram(array, xmin = xmin, xmax=xmax, Nbins=Nbins)
                preds['PDF'+suffix+str(lev)] = xr.DataArray(density, dims='q_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                    attrs={'long_name': 'subgrid forcing PDF'})

        for lev in [0,1]:
            arr = preds[target+'_res'].isel(time=time, lev=lev)
            mean, std = arr.mean(), arr.std()
            xmin = float(mean - 4*std); xmax = float(mean + 4*std)
            for suffix in ['_res', '_gen_res']:
                array = preds[target+suffix].isel(time=time, lev=lev).values.ravel()
                points, density = PDF_histogram(array, xmin = xmin, xmax=xmax, Nbins=Nbins)
                preds['PDF'+suffix+str(lev)] = xr.DataArray(density, dims='q_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                    attrs={'long_name': 'subgrid forcing residual PDF'})

        return preds.astype('float32')