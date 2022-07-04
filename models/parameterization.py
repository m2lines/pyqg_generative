from math import trunc
from tools.computational_tools import PDF_histogram
from tools.spectral_tools import calc_ispec, xarray_to_model, coord, spectrum, ave_lev
import xarray as xr
import pyqg
import numpy as np

DAY = 86400
DEFAULT_PYQG_PARAMS = dict(nx=64, dt=3600., tmax=90*DAY, tavestart=45*DAY)
SAMPLE_SLICE = slice(-20, None) # in indices
AVERAGE_SLICE = slice(360*5*DAY,None) # in seconds
AVERAGE_SLICE_ANDREW = slice(44,None) # in indices

def sample(ds, time=SAMPLE_SLICE, variable='q'):
    '''
    Recieves xr.Dataset and returns
    initial condition for QGModel as xarray
    '''
    q = ds[variable]
    if 'time' in q.dims:
        q = q.isel(time=time)
    
    for dim in ['run', 'time']:
        if dim in q.dims:
            coordinate = {dim: np.random.choice(q[dim])}
            q = q.sel(coordinate)
    return q

def concat_in_time(datasets):
    '''
    Concatenation of snapshots in time:
    - Concatenate everything
    - Store averaged statistics
    - Discard complex vars
    - Reduce precision
    '''
    # Concatenate datasets along the time dimension
    ds = xr.concat(datasets, dim='time')
    
    # Diagnostics get dropped by this procedure since they're only present for
    # part of the timeseries; resolve this by saving the most recent
    # diagnostics (they're already time-averaged so this is ok)
    for key,var in datasets[-1].variables.items():
        if key not in ds:
            ds[key] = var.isel(time=-1)

    # To save on storage, reduce double -> single
    # And discard complex vars
    for key,var in ds.variables.items():
        if var.dtype == np.float64:
            ds[key] = var.astype(np.float32)
        elif var.dtype == np.complex128:
            ds = ds.drop_vars(key)

    return ds

def concat_in_run(datasets, delta, time=AVERAGE_SLICE):
    '''
    Concatenation of runs:
    - Computes 1D spectra 
    - Computes PDFs
    - Average statistics over runs

    delta - H1/H2 layers height ratio, needed 
    for vertical averaging
    '''
    ds = xr.concat(datasets, dim='run')

    # Compute 1D spectra
    m = xarray_to_model(ds)
    for key in ['APEflux', 'APEgenspec', 'Dissspec', 'ENSDissspec', 
        'ENSflux', 'ENSfrictionspec', 'ENSgenspec', 'ENSparamspec', 
        'Ensspec', 'KEflux', 'KEfrictionspec', 'KEspec', 'entspec', 
        'paramspec', 'paramspec_APEflux', 'paramspec_KEflux']:
        var = ave_lev(ds[key].mean(dim='run'), delta)

        k, sp = calc_ispec(m, var.values, averaging=False, truncate=False)
        sp = xr.DataArray(sp, dims=['kr'],
            coords=[coord(k, 'isotropic wavenumber, $m^{-1}$')],
            attrs=dict(long_name=ds[key].attrs['long_name'], units=ds[key].attrs['units']+' * m'))
        ds[key+'r'] = sp

    # Check that selector defined in SECONDS (but not indices) works
    assert len(ds.time.sel(time=time)) < len(ds.time)
    # There are snapshots for PDF
    assert len(ds.time.sel(time=time)) > 0

    # Compute PDFs
    x = ds.sel(time=time).isel(lev=0)['q'].values.ravel()
    points, density = PDF_histogram(x, Nbins=100, xmin=-3e-5, xmax=3e-5)
    ds['pdf_pv'] = xr.DataArray(density, dims=['pv'],
        coords=[coord(points, 'potential vorticity, $m^{-1}$')],
        attrs=dict(long_name='PDF of upper level PV'))
    
    KE = ave_lev(0.5*(ds.u**2 + ds.v**2), delta)
    x = KE.sel(time=time).values.ravel()
    points, density = PDF_histogram(x, Nbins=50, xmin=0, xmax=0.005)
    ds['pdf_ke'] = xr.DataArray(density, dims=['ke'],
        coords=[coord(points, 'kinetic energy, $m^2/s^2$')],
        attrs=dict(long_name='PDF of depth-averaged KE'))

    return ds

def run_simulation(pyqg_params=DEFAULT_PYQG_PARAMS, parameterization = None, sample_interval = 30):
    '''
    Run model m with parameters pyqg_params
    and saves snapshots every sample_interval days
    Returns xarray.Dataset with snapshots and 
    averaged statistics
    '''
    params = pyqg_params.copy()
    if parameterization is not None:
        params.update(parameterization = parameterization)
    m = pyqg.QGModel(**params)
    
    snapshots = []
    for t in m.run_with_snapshots(tsnapint = sample_interval*86400):
        snapshots.append(m.to_dataset().copy(deep=True))
    
    return concat_in_time(snapshots)

def subgrid_scores(true, mean, gen):
    '''
    Compute scalar metrics for three components of subgrid forcing:
    - Mean subgrid forcing      ~ close to true forcing in MSE
    - Generated subgrid forcing ~ close to true forcing in spectrum
    - Genereted residual        ~ close to true residual in spectrum 
    true - xarray with true forcing
    mean - mean prediction
    gen  - generated prediction

    Result is score, i.e. 1-mse/normalization

    Here we assuma that dataset has dimensions run x time x lev x Ny x Nx
    '''
    def R2(x, x_true):
        dims = [d for d in x.dims if d != 'lev']
        return float((1 - ((x-x_true)**2).mean(dims) / (x_true**2).mean(dims)).mean())
    
    # first compute R2 for each layer, and after that normalize
    R2_mean = R2(mean, true)

    sp = spectrum(time=slice(None,None)) # power spectrum for full time slice

    sp_true = sp(true)
    sp_gen = sp(gen)
    R2_total = R2(sp_gen, sp_true)
    
    sp_true_res = sp(true-mean)
    sp_gen_res = sp(gen-mean)

    R2_residual = R2(sp_gen_res, sp_true_res)

    return R2_mean, R2_total, R2_residual

class Parameterization(pyqg.QParameterization):
    def predict(self, ds):
        '''
        ds - dataset having dimensions (lev, y, x);
        Additional dimensions run and time are arbitrary
        ds should contain input fields: self.inputs
        Output: dataset with fields: self.targets
        Additionally: mean and var are returned!
        '''
        raise NotImplementedError
    
    def __call__(self, m):
        '''
        m - instance of pyqg.QGModel
        return numpy array with prediction of 
        PV forcing
        '''
        ds = xr.Dataset()
        for var in self.inputs:
            ds[var] = xr.DataArray(m.__getattribute__(var), dims=('lev', 'y', 'x'))

        # Here I assume that there is only one target
        return self.predict(ds)[self.targets[0]].values

    def test_online(self, pyqg_params=DEFAULT_PYQG_PARAMS, nruns=10):
        '''
        Run ensemble of simulations with parameterization
        and save statistics 
        '''

        delta = pyqg.QGModel(**pyqg_params).delta
            
        datasets = []    
        for run in range(nruns):
            datasets.append(run_simulation(pyqg_params, 
                parameterization = self))
        return concat_in_run(datasets, delta=delta)

    def test_ensemble(self, ds: xr.Dataset, params_coarse, params_fine={}):
        '''
        Sample initial conditions from ds and 
        run ensemble of simulations
        '''
        m_param = pyqg.QGModel(**params_coarse, parameterization=self)
        m_coarse = pyqg.QGModel(**params_coarse)
        m_fine = pyqg.QGModel(**params_fine)

        q = sample(ds).values
        m_param.set_q1q2(*q.astype('float64'))
        m_param.run()

    def test_offline(self, ds: xr.Dataset, ensemble_size=10):
        '''
        Compute predictions of subgrid forces 
        on dataset ds and save dataset with statistics:
        Andrew metrics, spetral characteristics and PDFs
        '''        
        preds = self.predict(ds) # return sample, mean and var
        preds.attrs = ds.attrs
        
        target = 'q_forcing_advection'
        if target != self.targets[0]:
            raise ValueError('Wrong target!')

        # shuffle true and prediction
        preds[target+'_gen'] = preds[target].copy(deep=True)
        preds[target] = ds[target].copy(deep=True)
        # var -> std
        preds[target+'_std'] = preds[target+'_var']**0.5
        # residuals
        preds[target+'_res'] = preds[target] - preds[target+'_mean']
        preds[target+'_gen_res'] = preds[target+'_gen'] - preds[target+'_mean']

        # subgrid scores
        preds['R2_mean'], preds['R2_total'], preds['R2_residual'] = \
            subgrid_scores(ds[target], preds[target+'_mean'], preds[target+'_gen'])

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
                array = preds[target].isel(time=time, lev=lev).values.ravel()
                points, density = PDF_histogram(array, xmin = xmin, xmax=xmax, Nbins=Nbins)
                preds['PDF'+suffix+str(lev)] = xr.DataArray(density, dims='q_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                    attrs={'long_name': 'subgrid forcing PDF'})

        for lev in [0,1]:
            arr = preds[target+'_res'].isel(time=time, lev=lev)
            mean, std = arr.mean(), arr.std()
            xmin = float(mean - 4*std); xmax = float(mean + 4*std)
            for suffix in ['_res', '_gen_res']:
                array = preds[target].isel(time=time, lev=lev).values.ravel()
                points, density = PDF_histogram(array, xmin = xmin, xmax=xmax, Nbins=Nbins)
                preds['PDF'+suffix+str(lev)] = xr.DataArray(density, dims='q_'+str(lev), coords=[coord(points, '$dq/dt, s^{-2}$')],
                    attrs={'long_name': 'subgrid forcing residual PDF'})

        return preds

class ReferenceModel(Parameterization):
    def __init_(self):
        super().__init__()
    def __call__(self, m):
        return m.q*0