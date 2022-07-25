import glob
import os
import xarray as xr
import numpy as np
import pyqg

import pyqg_generative.tools.operators as op

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
    'spectral_diff_KEfrictionspec'
]

def folder_iterator(
    return_blowup=False, return_reference=False,
    base_folder='/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/', 
    Resolution = [48, 64, 96],
    Operator = ['Operator1', 'Operator2'],
    Model = ['OLSModel', 'MeanVarModel', 'CGANRegression'],
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
                        for configuration in Configuration:
                            folder = base_folder + _operator + '/' + model
                            subfolder = configuration + '-' + sampling + '-' + str(decorrelation)
                            folder = folder + '/' + subfolder
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
    return np.mean(list({k:v for k,v in similarity_instance.items() if k in DISTRIB_KEYS}.values()))

def spectral_score(similarity_instance):
    return np.mean(list({k:v for k,v in similarity_instance.items() if k in SPECTRAL_KEYS}.values()))

def total_score(similarity_instance):
    '''
    To be used as loss_function in finding best
    models
    '''
    return 0.5 * (distrib_score(similarity_instance) 
        + spectral_score(similarity_instance))

def score_for_model(similarity, loss_function, resolution, operator, model,
    Sampling=['AR1', 'constant'], Decorrelation = [0, 12, 24, 36, 48], 
    configuration='eddy'):
    '''
    similarity - dict with keys of format 'Operator1-64/MeanVarModel/eddy-AR1-12',
    containing dictionary of distributional and spectral keys (DISTRIB_KEY and SPECTRAL_KEY)

    Returns dict of time sampling schemes with computed score
    '''

    keys = [
        operator + '-' + str(resolution) + '/' + model + '/' + configuration + '-' + sampling + '-' + str(decorrelation)
        for sampling in Sampling
        for decorrelation in Decorrelation
    ]

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