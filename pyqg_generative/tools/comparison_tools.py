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
import pyqg_subgrid_experiments as pse
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

def cache_path(path):
    dir = os.path.dirname(path)
    files = os.path.basename(path)
    #https://www.delftstack.com/howto/python/str-to-hex-python/
    cachename = files.encode('utf-8').hex() + '.cache_netcdf'
    return os.path.join(dir,cachename)

def dataset_smart_read(path, delta=0.25, read_cache=True):
    print(path)
    nfiles = len(glob.glob(path))
    if nfiles < 10:
        print('Warning! Computations are unstable. Number of files is less than 10 and equal to', nfiles)
    cache = cache_path(path)
    if os.path.exists(cache) and read_cache:
        #print('Read cache ' + cache)
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
        xmax = 1e-10 if var=='Ens' and lev==0 else None
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

def ensemble_dataset_read(model_path, target_path):
    dir = os.path.dirname(model_path)
    cache = dir+'.nc'
    if os.path.exists(cache):
        return xr.open_dataset(cache)

    model = xr.open_mfdataset(model_path, combine='nested', concat_dim='run', decode_times=False).astype('float64')
    target = xr.open_mfdataset(target_path, combine='nested', concat_dim='run', decode_times=False).astype('float64')
    model['x'] = target['x']
    model['y'] = target['y']

    du_err = target['u'] - model['u_mean']
    dv_err = target['v'] - model['v_mean']
    du_res = model['u'] - model['u_mean']
    dv_res = model['v'] - model['v_mean']

    def time_mean(var):
        return var.isel(time=slice(5,31)).mean(dim='time')

    var_res = (du_res**2 + dv_res**2).mean(dim=('run', 'x', 'y'))
    var_err = (du_err**2 + dv_err**2).mean(dim=('run', 'x', 'y'))
    ensemble_spread = (time_mean(var_res) / time_mean(var_err)).mean(dim='lev')

    def normalize(var):
        return var / var.std(dim=('x', 'y'))

    def psd(var):
        M = len(var.x) * len(var.y)
        varf = np.fft.rfftn(var.values, axes=(-2, -1)) / M
        af2 = (np.abs(varf)**2).mean(axis=(0)) # averaging over ensemble members
        m = pyqg.QGModel(nx=len(var.x), log_level=0)
        out = []
        for t in range(af2.shape[0]):
            k, sp0 = calc_ispec(m, af2[t,0], averaging=True, truncate=True)
            k, sp1 = calc_ispec(m, af2[t,1], averaging=True, truncate=True)
            out.append(np.stack(
                [
                sp0,
                sp1
                ]
            ))
        out = xr.DataArray(np.stack(out), dims=['time', 'lev', 'kr'])
        out['kr'] = k
        return out
    
    KE_res = psd(du_res) + psd(dv_res)
    KE_err = psd(du_err) + psd(dv_err)

    du_err = normalize(du_err)
    dv_err = normalize(dv_err)
    du_res = normalize(du_res)
    dv_res = normalize(dv_res)

    KE_res_normalized = psd(du_res) + psd(dv_res)
    KE_err_normalized = psd(du_err) + psd(dv_err)

    dx = model.x[2] - model.x[1]
    kmax = 0.65 * np.pi / dx

    def L2(sp1, sp2):
        mask = sp1.kr <= kmax
        return ((mask*(sp1-sp2)**2).sum('kr'))**0.5
    
    def time_mean_L2(sp1, sp2):
        # 5 to 30 days fo prediction
        return time_mean(L2(sp1, sp2))

    ensemble_shape = (time_mean_L2(KE_res_normalized, KE_err_normalized) \
        / time_mean_L2(0*KE_res_normalized, KE_err_normalized)).mean(dim='lev')

    ds = xr.Dataset()
    for key in ['var_res', 'var_err', 'KE_res', 'KE_err', 'ensemble_spread', 'ensemble_shape', 'kmax']:
        ds[key] = eval(key)

    ds.to_netcdf(cache)
    return ds

def plot_panel_figure(operator='Operator1', resolution=48,
    models_folders = ['OLSModel', 'MeanVarModel', 'CGANRegression'],
    configuration = 'eddy', density='PDF_Ens1',
    models_weights = [],
    samplings = [], lss = [], lws = [], colors = [],
    markers = [], markersizes = [], markerfillstyles = [],
    markeredgewidths = [], labels = [], alphas = [],
    read_cache=True):
    '''
    Each style parameter may be specified for a few experiments
    or not defined. Autocompletion with default values will be
    performed
    '''
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(2,3, figsize=(14,6))
    plt.subplots_adjust(wspace=0.3, hspace=0.65)
    
    xlim=[1e-5, 2e-4]; ylim1=[1e-1,2e+2]; ylim2=[1e-2,1e+1]
    if resolution==64:
        ylim2=[5e-3,1e+1]
    elif resolution==96:
        xlim=[1e-5, 3e-4]
        ylim1=[1e-2, 2e+2]
        ylim2=[1e-4,1e+1]

    delta = 0.25 if configuration=='eddy' else 0.1

    def ave_lev(arr):
        return op.ave_lev(arr, delta)

    def dataset_read(path, read_cache=read_cache):
        return dataset_smart_read(path, delta, read_cache=read_cache)
    
    def style_complete(property, default, nm):
        '''
        If property is defined not for every model, 
        complete with default value
        '''
        if not isinstance(property, list):
            property = nm*[property]
        np = len(property)
        if np < nm:
            if isinstance(default, list):
                property += default[np:nm]
            else:
                property += [default]*(nm-np)
        return property

    nmodels = len(models_folders)
    samplings = style_complete(samplings, 'constant-0', nmodels)
    model_weights = style_complete(models_weights, '', nmodels)
    model_weights = ['' if w=='' or w==1 else str(w)+'-' for w in model_weights]
    samplings = [w+configuration+'-'+s for s,w in zip(samplings,model_weights)]

    lss = style_complete(lss, '-', nmodels)
    lws = style_complete(lws, 1, nmodels)
    colors = style_complete(colors, [f'C{j}' for j in range(nmodels)], nmodels) # just default colors
    markers = style_complete(markers, ['o', 's', '>', '<', 'X', 'D', '*', 'v', 'p', 'P', 'd'], nmodels)
    markersizes = style_complete(markersizes, 4, nmodels)
    markerfillstyles = style_complete(markerfillstyles, 'full', nmodels)
    markeredgewidths = style_complete(markeredgewidths, 1, nmodels)
    labels = style_complete(labels, models_folders, nmodels)
    alphas = style_complete(alphas, 1, nmodels)
    
    print(nmodels, labels)

    dns_line = {'color': 'k', 'ls': '-', 'lw': 1, 'label': 'DNS'}
    target_line = {'color': 'k', 'ls': '--', 'lw': 2, 'label': 'fDNS'}
    lores_line = {'color': 'gray', 'ls': '-', 'lw': 2, 'label': 'lores'}
    lores_3600_line = {'color': 'tab:purple', 'ls': '--', 'lw': 2, 'label': 'lores, $1h$'}

    lines = []
    for j in range(nmodels):
        lines.append({'ls': lss[j], 'lw': lws[j], 'color': colors[j], 
            'marker': markers[j], 'markersize': markersizes[j],
            'fillstyle': markerfillstyles[j], 'markeredgewidth': markeredgewidths[j],
            'label': labels[j], 'alpha': alphas[j]})
    
    hires = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_256/[0-9].nc', read_cache=read_cache)
    target = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_256/{operator}-{str(resolution)}.nc', read_cache=read_cache)
    lores = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_{str(resolution)}/[0-9].nc', read_cache=read_cache)
    lores_3600 = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_3600_{str(resolution)}/[0-9].nc', read_cache=read_cache)
    
    models = []
    for folder, sampling in zip(models_folders, samplings):
        try:
            ds = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/{operator}-{str(resolution)}/{folder}/{sampling}/[0-9].nc', read_cache=read_cache)
        except:
            ds = None
        models.append(ds) 

    offline = xr.open_dataset(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/{operator}-{str(resolution)}/0.nc')
    offline['paramspec_KEfluxr'] = ave_lev(spectrum(type='cospectrum', averaging=True, truncate=True)(-offline.psi, offline.q_forcing_advection))
    offline['paramspec_APEfluxr'] = 0*offline['paramspec_KEfluxr']

    ax = axs[0][0]
    for model, line in zip([hires, target, lores, lores_3600, *models], 
        [dns_line, target_line, lores_line, lores_3600_line, *lines]):
        try:
            model.KEspecr.isel(lev=0).plot(ax=ax, **line)
        except:
            pass
    for model, line in zip([hires, target, lores, lores_3600, *models], 
        [dns_line, target_line, lores_line, lores_3600_line, *lines]):
        try:
            model.KEspecr.isel(lev=1).plot(ax=ax, **line)
        except:
            pass
        
    ax.set_ylim([min(ylim2), max(ylim1)])
    ax.set_xlim(xlim)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('KE spectrum')
    ax.set_ylabel('Energy spectrum [$m^3/s^2$]', fontsize=11)
    ax.text(1.3e-4, 1.5e-1, 'lower')
    ax.text(1.2e-4, 1.5e+1, 'upper')
    
    ax = axs[1][0]
    for model, line in zip([hires, target, lores, lores_3600, *models], 
        [dns_line, target_line, lores_line, lores_3600_line, *lines]):
        try:
            model.Efluxr.plot(ax=ax, **line)
        except:
            pass
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim([-1.2e-6, 1.2e-6])
    ax.set_title('Energy transfer \n (resolved+subgrid)')
    ax.set_ylabel('Energy change [$m^3/s^3$]', fontsize=11)
    
    ax=axs[1][1]
    for model, line in zip([offline, *models],
        [target_line, *lines]):
        try:
            (model.paramspec_APEfluxr + model.paramspec_KEfluxr).plot(ax=ax, **line)
        except:
            pass
    ax.set_xlim(xlim)
    ax.set_xscale('log')
    ax.set_ylabel('Energy change [$m^3/s^3$]', fontsize=11)
    ax.set_title('Energy transfer \n (subgrid)')
    
    ax = axs[0][1]
    for model, line in zip([hires, target, lores, lores_3600, *models], 
        [dns_line, target_line, lores_line, lores_3600_line, *lines]):
        try:
            model.APEgenspecr.plot(ax=ax, **line)
        except:
            pass
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim([-2e-7, 2.5e-6])
    ax.set_title('Energy source')
    ax.set_ylabel('Energy change [$m^3/s^3$]', fontsize=11)
    
    ax = axs[0][2]
    for model, line in zip([hires, target, lores, lores_3600, *models], 
        [dns_line, target_line, lores_line, lores_3600_line, *lines]):
        try:
            model.KE_time.plot(ax=ax, markevery=6, **line)
        except:
            pass
    ax.set_title('KE per unit mass')
    ax.set_ylabel('Kinetic energy [$10^{-4} m^2/s^2$]', fontsize=11)
    ax.set_ylim([0, 8e-4])
    ax.set_yticks([0, 2e-4, 4e-4, 6e-4, 8e-4])
    ax.set_yticklabels([0,2,4,6,8])
    
    
    ax = axs[1][2]
    for model, line in zip([hires, target, lores, lores_3600, *models], 
        [dns_line, target_line, lores_line, lores_3600_line, *lines]):
        try:
            model[density].plot(ax=ax, markevery=2, **line)
        except:
            pass
    ax.set_yscale('log')
    ax.set_ylabel('Probability density', fontsize=11)
    if density == 'PDF_Ens1':
        ax.set_title('Upper enstrophy PDF')
        ax.set_xlabel('relative enstrophy [$10^{-11}s^{-2}$]')
        ax.set_xticks([0, 2e-11, 4e-11, 6e-11, 8e-11, 10e-11])
        ax.set_xticklabels([0, 2, 4, 6, 8, 10])
        ax.set_ylim([5e+7, 1e+12])
    elif density == 'PDF_KE1':
        ax.set_title('Upper KE PDF')
        ax.set_xlabel('kinetic energy [$m^2s^{-2}$]')
    
    fig.align_ylabels()
    fig.align_xlabels()
    axs[1][2].legend(frameon=False, ncol=2, fontsize=9)

    return axs

def plot_PDFs(operator='Operator1', resolution=48,
    models_folders = ['OLSModel', 'MeanVarModel', 'CGANRegression'],
    configuration = 'eddy',
    models_weights = [],
    samplings = [], lss = [], lws = [], colors = [],
    markers = [], markersizes = [], markerfillstyles = [],
    markeredgewidths = [], labels = [], alphas = [],
    read_cache=True):
    '''
    Each style parameter may be specified for a few experiments
    or not defined. Autocompletion with default values will be
    performed
    '''
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(2,3, figsize=(14,6))
    plt.subplots_adjust(wspace=0.3, hspace=0.65)
    
    xlim=[1e-5, 2e-4]; ylim1=[1e-1,2e+2]; ylim2=[1e-2,1e+1]
    if resolution==64:
        ylim2=[5e-3,1e+1]
    elif resolution==96:
        xlim=[1e-5, 3e-4]
        ylim1=[1e-2, 2e+2]
        ylim2=[1e-4,1e+1]

    delta = 0.25 if configuration=='eddy' else 0.1

    def ave_lev(arr):
        return op.ave_lev(arr, delta)

    def dataset_read(path, read_cache=read_cache):
        return dataset_smart_read(path, delta, read_cache=read_cache)
    
    def style_complete(property, default, nm):
        '''
        If property is defined not for every model, 
        complete with default value
        '''
        if not isinstance(property, list):
            property = nm*[property]
        np = len(property)
        if np < nm:
            if isinstance(default, list):
                property += default[np:nm]
            else:
                property += [default]*(nm-np)
        return property

    nmodels = len(models_folders)
    samplings = style_complete(samplings, 'constant-0', nmodels)
    model_weights = style_complete(models_weights, '', nmodels)
    model_weights = ['' if w=='' or w==1 else str(w)+'-' for w in model_weights]
    samplings = [w+configuration+'-'+s for s,w in zip(samplings,model_weights)]

    lss = style_complete(lss, '-', nmodels)
    lws = style_complete(lws, 1, nmodels)
    colors = style_complete(colors, [f'C{j}' for j in range(nmodels)], nmodels) # just default colors
    markers = style_complete(markers, ['o', 's', '>', '<', 'X', 'D', '*', 'v', 'p', 'P', 'd'], nmodels)
    markersizes = style_complete(markersizes, 4, nmodels)
    markerfillstyles = style_complete(markerfillstyles, 'full', nmodels)
    markeredgewidths = style_complete(markeredgewidths, 1, nmodels)
    labels = style_complete(labels, models_folders, nmodels)
    alphas = style_complete(alphas, 1, nmodels)
    
    print(nmodels, labels)

    dns_line = {'color': 'k', 'ls': '-', 'lw': 1, 'label': 'DNS'}
    target_line = {'color': 'k', 'ls': '--', 'lw': 2, 'label': 'fDNS'}
    lores_line = {'color': 'gray', 'ls': '-', 'lw': 2, 'label': 'lores'}
    lores_3600_line = {'color': 'tab:purple', 'ls': '--', 'lw': 2, 'label': 'lores, $1h$'}

    lines = []
    for j in range(nmodels):
        lines.append({'ls': lss[j], 'lw': lws[j], 'color': colors[j], 
            'marker': markers[j], 'markersize': markersizes[j],
            'fillstyle': markerfillstyles[j], 'markeredgewidth': markeredgewidths[j],
            'label': labels[j], 'alpha': alphas[j]})
    
    hires = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_256/[0-9].nc', read_cache=read_cache)
    target = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_256/{operator}-{str(resolution)}.nc', read_cache=read_cache)
    lores = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_{str(resolution)}/[0-9].nc', read_cache=read_cache)
    lores_3600 = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_3600_{str(resolution)}/[0-9].nc', read_cache=read_cache)
    
    models = []
    for folder, sampling in zip(models_folders, samplings):
        try:
            ds = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/{operator}-{str(resolution)}/{folder}/{sampling}/[0-9].nc', read_cache=read_cache)
        except:
            ds = None
        models.append(ds) 

    offline = xr.open_dataset(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/{operator}-{str(resolution)}/0.nc')
    offline['paramspec_KEfluxr'] = ave_lev(spectrum(type='cospectrum', averaging=True, truncate=True)(-offline.psi, offline.q_forcing_advection))
    offline['paramspec_APEfluxr'] = 0*offline['paramspec_KEfluxr']

    for row, layer in zip([0,1], ['Upper', 'Lower']):
        for col, xlabel, dens in zip([0,1,2], ['potential vorticity [$1/s$]', 'relative enstrophy [$1/s^2$]', 'kinetic energy [$m^2/s^2$]'], ['PDF_q', 'PDF_Ens', 'PDF_KE']):
            density = dens+str(row+1)
            ax = axs[row][col]
            for model, line in zip([hires, target, lores, *models], 
                [dns_line, target_line, lores_line, *lines]):
                try:
                    model[density].plot(ax=ax, markevery=2, **line)
                except:
                    pass
            ax.set_yscale('log')
            ax.set_ylabel('Probability density', fontsize=11)
            title = dict(PDF_q='PV $q$', PDF_Ens='$1/2|curl(\mathbf{v})|^2$', PDF_KE='$1/2|\mathbf{v}|^2$')[dens]
            ax.set_title(layer+' '+title)
            ax.set_xlabel(xlabel)
            if dens=='PDF_KE' and layer=='Lower':
                ax.set_xticks([0, 0.00025, 0.0005])

    fig.align_ylabels()
    fig.align_xlabels()
    axs[1][2].legend(frameon=False, ncol=2, fontsize=9)

    return axs

def plot_solution(operator='Operator1', resolution=48, 
    models_folders = ['OLSModel', 'MeanVarModel', 'CGANRegression'],
    configuration = 'eddy', samplings=None, labels=None,
    cbar_label='Kinetic \n energy [$m^2/s^2$]', fun = lambda ds: 0.5 * (ds.u**2 + ds.v**2)):

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(2,3+len(models_folders), figsize=(12,4))
    plt.subplots_adjust(wspace=0.0, hspace=0.1)

    delta = 0.25 if configuration=='eddy' else 0.1
    def dataset_read(path):
        return dataset_smart_read(path, delta)

    if samplings is None:
        samplings = ['constant-0'] * len(models_folders)
    
    if labels is None:
        labels = models_folders

    idx = dict(time=-1, run=0)

    for lev in [0,1]:
        row = lev
        hires = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_256/[0-9].nc').isel(idx).isel(lev=lev)
        target = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_256/{operator}-{str(resolution)}.nc').isel(idx).isel(lev=lev)
        lores = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{configuration}/reference_{str(resolution)}/[0-9].nc').isel(idx).isel(lev=lev)
        
        models = []
        for folder, sampling in zip(models_folders, samplings):
            try:
                ds = dataset_read(f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/{operator}-{str(resolution)}/{folder}/{configuration}-{sampling}/[0-9].nc').isel(idx).isel(lev=lev)
            except:
                ds = None
            models.append(ds) 
        
        var = fun(hires)
        vmax = np.percentile(var, 99)
        if var.min() < 0 and vmax > 0:
            vmin = -vmax
            cmap = 'bwr'
        elif var.min() < 0 and vmax < 0:
            vmin = np.percentile(var, 1)
            cmap = 'inferno'
        else:
            vmin = 0
            cmap = 'inferno'
            
        print(vmin, vmax)
        
        for col, exp in enumerate([hires, target, lores, *models]):
            ax = axs[row][col]
            var = fun(exp)
            im = ax.imshow(var, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        
        if row==0:
            plt.colorbar(im, cax=fig.add_axes([0.9,0.52,0.01,0.36]), label=cbar_label)
        if row==1:
            plt.colorbar(im, cax=fig.add_axes([0.9,0.125,0.01,0.36]), label=cbar_label)

    for col, title in enumerate(['DNS', 'fDNS', 'lores', *labels]):
        axs[0][col].set_title(title)
    axs[0][0].set_ylabel('upper layer')
    axs[1][0].set_ylabel('lower layer')

def plot_difference(
    configuration='eddy',
    sampling='constant-0',
    timestep='',
    model_weight = 1,
    models=['Reference', 'OLSModel', 'MeanVarModel', 'CGANRegression'], 
    labels=['lores', 'MSE', 'GZ', 'GAN'], 
    markers = ['', 'o', 's', '>', '<', 'X', 'd'],
    colors = ['k'] + [f'C{j}' for j in range(0,10)],
    normalize=False):
    import json
    with open('difference.json', 'r') as file:
        difference = json.load(file)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(2,2, figsize=(10,5))
    plt.subplots_adjust(wspace=0.2)

    keys = []
    for model in models:
        postfix = '' if model.find('Reference')>-1 else '-' + str(sampling)
        prefix = str(model_weight) + '-' if model_weight != 1 and model.find('Reference')==-1 else ''
        keys.append(f'{model}/{prefix}{configuration}{timestep}{postfix}')

    def resolution_slice(operator, model_discriptor):
        resolutions=[48, 64, 96]
        keys = [f'{operator}-{str(res)}/{model_discriptor}' for res in resolutions]
        dist = [distrib_score(difference.get(key, {})) for key in keys]
        spec = [spectral_score(difference.get(key, {})) for key in keys]
        return xr.DataArray(dist, coords=[resolutions]), xr.DataArray(spec, coords=[resolutions])

    lss = ['--', '-', '-', '-', '-', '-', '-']

    for row, operator in enumerate(['Operator1', 'Operator2']):
        for key, label, marker, color, ls in zip(
                            keys, labels, markers, colors, lss):
            dist, spec = resolution_slice(operator, key)
            if normalize:
                dist0, spec0 = resolution_slice(operator, keys[0])
                dist = dist / dist0
                spec = spec / spec0
            dist.plot(ax=axs[row][0], label=label, marker=marker, color=color, lw=1.5, ls=ls)
            spec.plot(ax=axs[row][1], label=label, marker=marker, color=color, lw=1.5, ls=ls)
    
    axs[0][0].set_title('Normalized distribution error')
    axs[0][1].set_title('Normalized spectral error')
    axs[0][0].set_ylabel('mean $\widetilde{W}_1(F_1,F_2)$')
    axs[1][0].set_ylabel('mean $\widetilde{W}_1(F_1,F_2)$')
    axs[0][1].set_ylabel('mean $\widetilde{L}_2(\mathcal{E}_1,\mathcal{E}_2)$')
    axs[1][1].set_ylabel('mean $\widetilde{L}_2(\mathcal{E}_1,\mathcal{E}_2)$')

    fig.text(0.91,0.54,'cut-off + exponential', rotation=-90)
    fig.text(0.91,0.15,'cut-off + Gaussian', rotation=-90)

    axs[0][1].legend(frameon=False, ncol=2)
    for i in [0,1]:
        for j in [0,1]:
            if i==0:
                axs[i][j].set_xticklabels([])
                axs[i][j].set_xlabel('')
            else:
                axs[i][j].set_xlabel('resolution')
            axs[i][j].set_xticks([48, 64, 96])
    
    if normalize:
        for j in [0,1]:
            axs[0][j].set_ylim(0,1.2)
        for j in [0,1]:
            axs[1][j].set_ylim(0,2)
    else:
        axs[0][0].set_ylim([0, 0.22])
        axs[1][0].set_ylim([0, 0.22])
        axs[0][1].set_ylim([0,0.6])
        axs[1][1].set_ylim([0,0.6])

    return axs   

def plot_offline_metrics(models=['OLSModel', 'MeanVarModel', 'CGANRegression', 'CGANRegression-recompute', 'CGANRegression-None-recompute', 'CGANRegression-Unet'],
    labels=['MSE', 'GZ', 'GAN', 'GAN', 'GAN-full', 'GAN-Unet'], dataset='offline_test.nc'):
    
    def L2(type='mean', model='OLSModel', operator='Operator1', resolution=64):
        folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/'
        file = os.path.join(folder, operator+'-'+str(resolution),model,dataset)
        if os.path.exists(file):
            ds = xr.open_dataset(file)
            if type != 'var_ratio':
                return float(ds['L2_'+type])
            else:
                return float(ds[type].mean())
        else:
            print('Wrong path', file)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    markers = ['o', 's', '>', '<', 'X', 'd', 'v', '^', '1', '2', '3', '4', '*', 'p', 'h']
    colors = [f'C{j}' for j in range(0,20)]
    lss = ['-'] * 20
    fig, axs = plt.subplots(2,4,figsize=(16,5))
    plt.subplots_adjust(hspace=0.08, wspace=0.2)
    for i, operator in enumerate(['Operator1', 'Operator2']):
        for j, (model, label) in enumerate(zip(models, labels)):
            res = [32, 48, 64, 96]
            L2_mean     = [L2('mean', model, operator, resolution) for resolution in res]
            L2_total    = [L2('total', model, operator, resolution) for resolution in res]
            L2_residual = [L2('residual', model, operator, resolution) for resolution in res]
            var_ratio = [L2('var_ratio', model, operator, resolution) for resolution in res]
            
            line = dict(marker=markers[j], color=colors[j], ls=lss[j])
            
            ax = axs[i][0]
            ax.plot(res, L2_mean, label=label, **line)
            
            ax = axs[i][1]
            ax.plot(res, L2_total, label=label, **line)
            
            ax = axs[i][2]
            ax.plot(res, L2_residual, label=label, **line)
            
            ax = axs[i][3]
            ax.plot(res, var_ratio, label=label, **line)
    
    if dataset == 'offline_test.nc':
        for i in [0,1]:
            axs[i][0].set_ylim([-0.05, 0.905])
            axs[i][1].set_ylim([-0.02, 0.275])
            axs[i][2].set_ylim([-0.2, 1.2])
            axs[i][3].set_ylim([-0.2, 1.2])
    else:
        pass

    for j in range(4):
        axs[1][j].set_xlabel('resolution')
        axs[1][j].set_xticks([32, 48, 64, 96])
        axs[0][j].set_xticks([32, 48, 64, 96])
        axs[0][j].set_xticklabels([])
    axs[0][0].set_title('$L_2^{mean}$')
    axs[0][1].set_title('$L_2^{sample}$')
    axs[0][2].set_title('$L_2^{residual}$')
    axs[0][3].set_title('$spread$')

    axs[0][0].set_ylabel('cut-off + exponential')
    axs[1][0].set_ylabel('cut-off + Gaussian')

    axs[0][3].legend(bbox_to_anchor=(1,1))

    return axs    

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