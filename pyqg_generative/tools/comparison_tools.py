import glob
import os
import sys
import xarray as xr
import numpy as np
import pyqg
import json

import pyqg_generative.tools.operators as op
from pyqg_generative.tools.computational_tools import PDF_histogram
from pyqg_generative.tools.parameters import AVERAGE_SLICE_ANDREW
from pyqg_generative.tools.spectral_tools import calc_ispec, coord
import pyqg_subgrid_experiments as pse
import pyqg_parameterization_benchmarks as ppb

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

def folder_iterator(
    return_blowup=False, return_reference=False,
    base_folder='/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/', 
    Resolution = [32, 48, 64, 96],
    Operator = ['Operator1', 'Operator2'],
    Model = ['OLSModel', 'MeanVarModel', 'CGANRegression', 'CGANRegressionxy-full'],
    Sampling = ['AR1', 'constant'],
    Decorrelation = [0, 12, 24, 36, 48],
    Configuration = ['eddy']
    ):

    for resolution in Resolution:
        for operator in Operator:
            for model in Model:
                _operator = operator+'-'+str(resolution)
                for sampling in Sampling:
                    for decorrelation in Decorrelation: # in hours; 0 means tau=dt
                        if model=='OLSModel' and sampling=='AR1':
                            continue
                        if sampling=='AR1' and decorrelation==0:
                            continue
                        if decorrelation>0 and sampling=='constant':
                            continue
                        for configuration in Configuration:
                            folder = base_folder + _operator + '/' + model
                            subfolder = configuration + '-' + sampling + '-' + str(decorrelation)
                            folder = folder + '/' + subfolder
                            if not os.path.exists(folder):
                                continue
                            nfiles = len(glob.glob(os.path.join(folder, '*.nc')))
                            if not return_blowup:
                                if nfiles != 10:
                                    continue
                            
                            if return_reference:
                                reference = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_256/'+operator+'-'+str(resolution)+'.nc'
                                key = _operator + '/' + model + '/' + subfolder
                                baseline = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_'+str(resolution) + '/*.nc'
                                yield folder, reference, baseline, key
                            else:
                                yield folder

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
    else:
        raise ValueError('operator must be Operator1 or Operator2')

    dsf = xr.Dataset()
    for var in ['q', 'u', 'v', 'psi']:
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

def total_score(similarity_instance):
    '''
    To be used as loss_function in finding best
    models
    '''
    return 0.5 * (distrib_score(similarity_instance) 
        + spectral_score(similarity_instance))
        
def KEspec_score(similarity_instance):
    return (similarity_instance['spectral_diff_KEspec1'] + similarity_instance['spectral_diff_KEspec2']) * 0.5

def score_for_model(similarity, loss_function, resolution, operator, model,
    Sampling=['AR1', 'constant'], Decorrelation = [0, 12, 24, 36, 48], 
    configuration='eddy'):
    '''
    similarity - dict with keys of format 'Operator1-64/MeanVarModel/eddy-AR1-12',
    containing dictionary of distributional and spectral keys (DISTRIB_KEY and SPECTRAL_KEY)

    Returns dict of time sampling schemes with computed score
    '''
    keys = []
    for sampling in Sampling:
        for decorrelation in Decorrelation:
            if sampling=='constant' and decorrelation>0:
                continue
            keys.append(operator + '-' + str(resolution) + '/' + model + '/' + configuration + '-' + sampling + '-' + str(decorrelation))
    
    return {key:loss_function(similarity[key]) for key in keys if key in similarity.keys()}


def best_time_sampling(similarity, loss_function,
    Resolution = [48, 64, 96],
    Operator = ['Operator1', 'Operator2'],
    Model = ['OLSModel', 'MeanVarModel', 'CGANRegression'],
    Sampling = ['AR1', 'constant'],
    Decorrelation = [0, 12, 24, 36, 48],
    configuration = 'eddy'):
    '''
    loss_function is a scalar-valued function
    of similarity dictionary
    '''

    loss_similarity = {key:loss_function(similarity[key]) for key in similarity.keys()}

    for resolution in Resolution:
        for operator in Operator:
            for model in Model:
                keys = [
                    operator + '-' + str(resolution) + '/' + model + '/' + configuration + '-' + sampling + '-' + str(decorrelation)
                    for sampling in Sampling
                    for decorrelation in Decorrelation
                ]
                model_dict = {key:loss_similarity[key] for key in keys if key in similarity.keys()}
                #https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
                return model_dict#max(model_dict, key=model_dict.get)

def cache_path(path):
    dir = os.path.dirname(path)
    files = os.path.basename(path)
    #https://www.delftstack.com/howto/python/str-to-hex-python/
    cachename = files.encode('utf-8').hex() + '.cache_netcdf'
    return os.path.join(dir,cachename)

def dataset_smart_read(path, delta=0.25, read_cache=True):
    cache = cache_path(path)
    if os.path.exists(cache) and read_cache:
        print('Read cache ' + cache)
        ds1 = xr.open_mfdataset(path, combine='nested', concat_dim='run', decode_times=False)
        ds2 = xr.open_dataset(cache)
        ds1['time'] = ds1['time'] / 360
        ds1['time'].attrs = {'long_name':'time [$years$]'}
        ds2['time'] = ds1['time'] # make sure time is the same
        return xr.merge([ds1, ds2])
    if os.path.exists(cache) and not read_cache:
        print('Delete cache ' + cache)
        os.remove(cache)

    ds = pse.Dataset(path)

    print('Compute statistics')
    stats = xr.Dataset()

    def KE(ds):
        return (ds.u**2 + ds.v**2) * 0.5
    
    def Ens(ds):
        return 0.5 * (ds.relative_vorticity)**2
    
    def KE_time(ds):
        return op.ave_lev(KE(ds), delta=delta).mean(('run', 'x', 'y'))
    
    def PDF_var(ds, var, lev):
        ds_ = ds.isel(time=AVERAGE_SLICE_ANDREW).isel(lev=lev)
        if var == 'KE':
            values = KE(ds_)
        elif var == 'Ens':
            values = Ens(ds_)
        else:
            values = ds_[var]
        values = values.values.ravel()

        xmin = 0 if var in ['KE', 'Ens'] else None
        xmax = 1e-10 if var=='Ens' else None
        points, density = PDF_histogram(values, xmin=xmin, xmax=xmax)
        return xr.DataArray(density, dims=f'{var}_{lev}', coords=[points])

    for var in ['q', 'u', 'v', 'KE', 'Ens']:
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

    return xr.merge([ds.ds, stats])

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
    
    difference,_,_ = ppb.diagnostic_differences_Perezhogin(model, target, T=128)
    print('difference calculated')
    difference['key'] = args.key
    print('key added')

    with open(args.save_file, 'w') as file:
        print('json file opened')
        json.dump(difference, file)
        print('json dump done')
    print('json file closed')