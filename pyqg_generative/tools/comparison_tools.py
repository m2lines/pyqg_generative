import glob
import os
import sys
import xarray as xr
import numpy as np
import pyqg
import json
from scipy.stats import wasserstein_distance

import pyqg_generative.tools.operators as op
from pyqg_generative.tools.computational_tools import PDF_histogram
from pyqg_generative.tools.parameters import AVERAGE_SLICE_ANDREW
from pyqg_generative.tools.spectral_tools import calc_ispec, coord, spectrum
from pyqg_parameterization_benchmarks.utils import FeatureExtractor

DISTRIB_KEYS = [
    'distrib_diff_q1',
    'distrib_diff_q2',
    'distrib_diff_u1',
    'distrib_diff_u2',
    'distrib_diff_v1',
    'distrib_diff_v2',
    'distrib_diff_KE1',
    'distrib_diff_KE2',
    'distrib_diff_Ens1',
    'distrib_diff_Ens2'
]

SPECTRAL_KEYS = [
    'spectral_diff_KEspec1',
    'spectral_diff_KEspec2',
    'spectral_diff_KEflux',
    'spectral_diff_APEflux',
    'spectral_diff_APEgenspec',
    'spectral_diff_KEfrictionspec',
    'spectral_diff_Eflux'
]

def distrib_score(similarity_instance):
    l = list({k:v for k,v in similarity_instance.items() if k in DISTRIB_KEYS}.values())
    if len(l) > 0:
        return np.mean(l)
    else:
        return np.nan

def spectral_score(similarity_instance):
    l = list({k:v for k,v in similarity_instance.items() if k in SPECTRAL_KEYS}.values())
    if len(l) > 0:
        return np.mean(l)
    else:
        return np.nan

def coarsegrain_reference_dataset(ds, resolution, operator):
    '''
    Reference dataset contains statistics at resolution 256^2,
    but need to be coarsegrained before comparison. 
    Coarsegraining is applied to:
    - Snapshots of the solution
    - Spectral fluxes

    Any total spectral flux (APE gen, APE flux, KE flux, Bottom drag),
    i.e. rate of change of filtered energy due to resolved and
    unresolved interactions,
    is pointwise (in Fourier space) quadratic function of filtered 
    Fourier coefficients. Therefore, spectral flux
    need to be multiplied by the square of filter 
    transmission function.

    operator: 'Operator1' or 'Operator2'
    resolution: 32, 48, 64, 96
    '''
    if operator == 'Operator1':
        operator = op.Operator1
    elif operator == 'Operator2':
        operator = op.Operator2
    elif operator == 'Operator4':
        operator = op.Operator4
    elif operator == 'Operator5':
        operator = op.Operator5
    else:
        raise ValueError('operator must be Operator1 or Operator2')

    dsf = xr.Dataset()
    for var in ['q', 'u', 'v', 'psi']:
        print(f'var={var}')
        dsf[var] = operator(ds[var], resolution)

    # Coarsegrain spectral flux
    n = resolution // 2
    ratio = ds.x.size / resolution
    for var in ['KEspec', 'KEflux', 'APEflux', 'APEgenspec', 'KEfrictionspec']:
        if len(ds[var].shape) == 3:
            array = np.concatenate((ds[var][:,:n,:n+1], ds[var][:,-n:,:n+1]), axis=-2)
            dsf[var] = xr.DataArray(array, dims=['run', 'l', 'k'])
        elif len(ds[var].shape) == 4:
            array = np.concatenate((ds[var][:,:,:n,:n+1], ds[var][:,:,-n:,:n+1]), axis=-2)
            dsf[var] = xr.DataArray(array, dims=['run', 'lev', 'l', 'k'])
        else:
            raise ValueError('var must be 3 or 4 dimensional')
    m = pyqg.QGModel(nx=resolution, log_level=0)

    dsf['k'] = m.to_dataset().k
    dsf['l'] = m.to_dataset().l

    # Apply filter
    for var in ['KEspec', 'KEflux', 'APEflux', 'APEgenspec', 'KEfrictionspec']:
        if operator == op.Operator1:
            dsf[var] = dsf[var] * m.filtr * m.filtr
        elif operator == op.Operator2:
            k2 = dsf['k']**2 + dsf['l']**2
            dx = m.dx
            filtr = np.exp(-k2 * (2*dx)**2 / 24)
            dsf[var] = dsf[var] * filtr * filtr
    return dsf

def diagnostic_differences_Perezhogin(ds1, ds2, T=128): 
    '''
    Here it is assumed that ds2 is a target. It is used for 
    normalization
    Output: differences dictionary and scales dictionary
    '''  
    if 'run' not in ds1.dims: ds1 = ds1.expand_dims('run')
    if 'run' not in ds2.dims: ds2 = ds2.expand_dims('run')

    from time import time

    distribution_quantities = dict(
        q='q',
        u='u',
        v='v',
        KE='add(pow(u,2),pow(v,2))', # u^2 + v^2
        Ens='pow(curl(u,v),2)', # (u_y - v_x)^2
    )
    
    differences = {}
    scales = {} # normalization factors
    
    for label, expr in distribution_quantities.items():
        for z in [0,1]:
            # Flatten over space and the last T timesteps
            ts = slice(-T,None)
            q1 = FeatureExtractor(ds1.isel(lev=z,time=ts))(expr).ravel()
            q2 = FeatureExtractor(ds2.isel(lev=z,time=ts))(expr).ravel()
            # Compute the empirical wasserstein distance
            differences[f"distrib_diff_{label}{z+1}"] = wasserstein_distance(q1, q2)
            scales[f"distrib_diff_{label}{z+1}"] = float(np.sqrt(np.mean(q2**2)))
            
    def twothirds_nyquist(m):
        return m.k[0][np.argwhere(np.array(m.filtr)[0]<1)[0,0]]
        
    def spectral_rmse(spec1, spec2):
        # Initialize pyqg models so we can use pyqg's calc_ispec helper
        m1 = pyqg.QGModel(nx=spec1.data.shape[-2], log_level=0)
        m2 = pyqg.QGModel(nx=spec2.data.shape[-2], log_level=0)
        # Compute isotropic spectra
        kr1, ispec1 = calc_ispec(m1, spec1.values)
        kr2, ispec2 = calc_ispec(m2, spec2.values)
        # Take error over wavenumbers below 2/3 of both models' Nyquist freqs
        kmax = min(twothirds_nyquist(m1), twothirds_nyquist(m2))
        nk = (kr1 < kmax).sum()

        return np.sqrt(np.mean((ispec1[:nk].astype('float64')-ispec2[:nk].astype('float64'))**2)), np.sqrt(np.mean((ispec2[:nk].astype('float64'))**2))
        
    for spec in ['KEspec']:
        for z in [0,1]:
            tt = time()
            spec1 = ds1[spec].isel(lev=z).mean('run')
            spec2 = ds2[spec].isel(lev=z).mean('run')
            differences[f"spectral_diff_{spec}{z+1}"], scales[f"spectral_diff_{spec}{z+1}"] \
                 = spectral_rmse(spec1, spec2)

    def compute_Eflux(ds):
        out = 0
        for spec in ['KEflux', 'APEflux', 'paramspec_KEflux', 'paramspec_APEflux']:
            if spec in ds.data_vars:
                out = out + ds[spec].mean('run')
        return out

    for spec in ['Eflux']:
        spec1 = compute_Eflux(ds1)
        spec2 = compute_Eflux(ds2)
        differences[f"spectral_diff_{spec}"], scales[f"spectral_diff_{spec}"] \
            = spectral_rmse(spec1, spec2)

    for spec in ['APEgenspec']:
        spec1 = ds1[spec].mean('run')
        spec2 = ds2[spec].mean('run')
        differences[f"spectral_diff_{spec}"], scales[f"spectral_diff_{spec}"] \
            = spectral_rmse(spec1, spec2)

    normalized_differences = {}
    for key in differences.keys():
        normalized_differences[key] = differences[key]/scales[key]
        
    return normalized_differences, differences, scales

def dataset_statistics(ds, delta=0.25, **kw_ispec):
    '''
    If path is given, the dataset is returned as is
    If dataset is given, statistics are computed
    '''
    if isinstance(ds, str):
        ds = xr.open_mfdataset(ds, combine='nested', concat_dim='run', decode_times=False, chunks={'time':1, 'run':1})
        if 'years' not in ds['time'].attrs:
            ds['time'] = ds['time'] / 360
            ds['time'].attrs = {'long_name':'time [$years$]'}
        return ds

    def KE(ds):
        return (ds.u**2 + ds.v**2) * 0.5
    
    def KE_time(ds):
        if 'run' in ds.dims:
            dims = ['run', 'x', 'y']
        else:
            dims = ['x', 'y']
        return op.ave_lev(KE(ds), delta=delta).mean(dims)
    
    stats = xr.Dataset()

    m = pyqg.QGModel(nx=len(ds.x), log_level=0)
    for key in ['APEflux', 'APEgenspec', 'Dissspec', 'ENSDissspec', 
        'ENSflux', 'ENSfrictionspec', 'ENSgenspec', 'ENSparamspec', 
        'Ensspec', 'KEflux', 'KEfrictionspec', 'KEspec', 'entspec', 
        'paramspec', 'paramspec_APEflux', 'paramspec_KEflux']:
        if key not in ds.keys():
            continue
        var = ds[key]    
        if 'run' in ds.dims:
            var = var.mean(dim='run')
        if 'lev' in var.dims:
            sps = []
            for z in [0,1]:
                k, sp = calc_ispec(m, var.isel(lev=z).values, **kw_ispec)
                sps.append(sp)
            sp = np.stack(sps, axis=0)
            stats[key+'r'] = \
                xr.DataArray(sp, dims=['lev', 'kr'],
                coords=[[1,2], coord(k, 'wavenumber, $m^{-1}$')])

            var_mean = op.ave_lev(var, delta)
            k, sp = calc_ispec(m, var_mean.values, **kw_ispec)
            stats[key+'r_mean'] = \
                xr.DataArray(sp, dims=['kr'],
                coords=[coord(k, 'wavenumber, $m^{-1}$')])
        else:
            k, sp = calc_ispec(m, var.values, **kw_ispec)
            stats[key+'r'] = \
                xr.DataArray(sp, dims=['kr'],
                coords=[coord(k, 'wavenumber, $m^{-1}$')])

    budget_sum = 0
    for key in ['KEfluxr', 'APEfluxr', 'APEgenspecr', 'KEfrictionspecr', 
        'paramspec_APEfluxr', 'paramspec_KEfluxr']:
        if key in stats.keys():
            budget_sum += stats[key]
    stats['Energysumr'] = budget_sum

    Eflux = 0
    for key in ['KEfluxr', 'APEfluxr', 'paramspec_KEfluxr', 'paramspec_APEfluxr']:
        if key in stats.keys():
            Eflux = Eflux + stats[key]
    stats['Efluxr'] = Eflux

    stats['KE_time'] = KE_time(ds)

    if 'years' not in stats['time'].attrs:
        stats['time'] = stats['time'] / 360
        stats['time'].attrs = {'long_name':'time [$years$]'}

    return stats

def cache_path(path):
    dir = os.path.dirname(path)
    files = os.path.basename(path)
    #https://www.delftstack.com/howto/python/str-to-hex-python/
    cachename = files.encode('utf-8').hex() + '.cache_netcdf'
    return os.path.join(dir,cachename)

def dataset_smart_read(path, delta=0.25, read_cache=True, compute_all=True):
    #print(path)
    nfiles = len(glob.glob(path))
    #if nfiles < 10:
        #print('Warning! Computations are unstable. Number of files is less than 10 and equal to', nfiles)
    cache = cache_path(path)
    if os.path.exists(cache) and read_cache:
        #print('Read cache ' + cache)
        ds1 = xr.open_mfdataset(path, combine='nested', concat_dim='run', decode_times=False, chunks={'time':1, 'run':1})
        ds2 = xr.open_dataset(cache)
        ds1['time'] = ds1['time'] / 360
        ds1['time'].attrs = {'long_name':'time [$years$]'}
        ds2['time'] = ds1['time'] # make sure time is the same
        return xr.merge([ds1, ds2])
    if os.path.exists(cache) and not read_cache:
        #print('Delete cache ' + cache)
        os.remove(cache)

    ds = xr.open_mfdataset(path, combine='nested', concat_dim='run', decode_times=False, chunks={'time':1, 'run':1})
    ds['time'] = ds['time'] / 360
    ds['time'].attrs = {'long_name':'time [$years$]'}

    #print('Compute statistics')
    stats = xr.Dataset()

    def KE(ds):
        return (ds.u**2 + ds.v**2) * 0.5
    
    def relative_vorticity(ds):
        return xr.DataArray(FeatureExtractor(ds)('curl(u,v)').compute(), dims=ds.q.dims)
    
    def Ens(ds):
        return 0.5 * (relative_vorticity(ds))**2
    
    def KE_time(ds):
        return op.ave_lev(KE(ds), delta=delta).mean(('run', 'x', 'y'))
    
    def Vabs(ds):
        return np.sqrt(2*KE(ds))

    if compute_all:
        stats['omega'] = relative_vorticity(ds)
        stats['KE'] = KE(ds)
        stats['Ens'] = Ens(ds)
        stats['Vabs'] = Vabs(ds)
    
    def PDF_var(ds, var, lev):
        if compute_all:
            time = AVERAGE_SLICE_ANDREW
        else:
            time = slice(-1, None)
        ds_ = ds.isel(time=time).isel(lev=lev)
        if var == 'KE':
            values = KE(ds_)
        elif var == 'Ens':
            values = Ens(ds_)
        else:
            values = ds_[var]
        values = values.values.ravel()

        xmin = 0 if var in ['KE', 'Ens'] else None
        if var=='Ens' and lev==0:
            xmax = 1e-10
        elif var=='Ens' and lev==1:
            xmax = 1.5e-12
        elif var=='KE' and lev==0:
            xmax = 1.5e-2
        elif var=='KE' and lev==1:
            xmax = 5e-4
        else:
            xmax = None

        points, density = PDF_histogram(values, xmin=xmin, xmax=xmax)
        return xr.DataArray(density, dims=f'{var}_{lev}', coords=[points])

    if compute_all:
        variables = ['q', 'u', 'v', 'KE', 'Ens']
    else:
        variables = ['q', 'u', 'v', 'KE']
    
    for var in variables:
        for lev in [0,1]:
            stats[f'PDF_{var}{lev+1}'] = PDF_var(ds, var, lev)

    m = pyqg.QGModel(nx=len(ds.x), log_level=0)
    for key in ['APEflux', 'APEgenspec', 'Dissspec', 'ENSDissspec', 
        'ENSflux', 'ENSfrictionspec', 'ENSgenspec', 'ENSparamspec', 
        'Ensspec', 'KEflux', 'KEfrictionspec', 'KEspec', 'entspec', 
        'paramspec', 'paramspec_APEflux', 'paramspec_KEflux']:
        if key not in ds.keys():
            continue
        var = ds[key].mean(dim='run')
        if 'lev' in var.dims:
            sps = []
            for z in [0,1]:
                k, sp = calc_ispec(m, var.isel(lev=z).values)
                sps.append(sp)
            sp = np.stack(sps, axis=0)
            stats[key+'r'] = \
                xr.DataArray(sp, dims=['lev', 'kr'],
                coords=[[1,2], coord(k, 'wavenumber, $m^{-1}$')])

            var_mean = op.ave_lev(var, delta)
            k, sp = calc_ispec(m, var_mean.values)
            stats[key+'r_mean'] = \
                xr.DataArray(sp, dims=['kr'],
                coords=[coord(k, 'wavenumber, $m^{-1}$')])
        else:
            k, sp = calc_ispec(m, var.values)
            stats[key+'r'] = \
                xr.DataArray(sp, dims=['kr'],
                coords=[coord(k, 'wavenumber, $m^{-1}$')])

    budget_sum = 0
    for key in ['KEfluxr', 'APEfluxr', 'APEgenspecr', 'KEfrictionspecr', 
        'paramspec_APEfluxr', 'paramspec_KEfluxr']:
        if key in stats.keys():
            budget_sum += stats[key]
    stats['Energysumr'] = budget_sum

    Eflux = 0
    for key in ['KEfluxr', 'APEfluxr', 'paramspec_KEfluxr', 'paramspec_APEfluxr']:
        if key in stats.keys():
            Eflux = Eflux + stats[key]
    stats['Efluxr'] = Eflux

    stats['KE_time'] = KE_time(ds)

    stats.to_netcdf(cache)

    return xr.merge([ds, stats])

if __name__ ==  '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--save_file', type=str)
    parser.add_argument('--key', type=str)
    args = parser.parse_args()

    print(args)

    model = xr.open_mfdataset(args.model_path, combine='nested', concat_dim='run')
    print('model loaded')
    target = xr.open_dataset(args.target_path)
    print('target loaded')
    
    difference,_,_ = diagnostic_differences_Perezhogin(model, target, T=128)
    print('difference calculated')
    difference['key'] = args.key
    print('key added')

    with open(args.save_file, 'w') as file:
        print('json file opened')
        json.dump(difference, file)
        print('json dump done')
    print('json file closed')
    del model, target