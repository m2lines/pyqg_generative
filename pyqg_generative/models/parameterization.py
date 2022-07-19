from pyqg_generative.tools.computational_tools import PDF_histogram
from pyqg_generative.tools.spectral_tools import calc_ispec, spectrum
from pyqg_generative.tools.cnn_tools import array_to_dataset, timer, init_seeds
from pyqg_generative.tools.operators import coord, ave_lev
import pyqg_parameterization_benchmarks as ppb
from torch.multiprocessing import Pool, set_start_method
import xarray as xr
import pyqg
import numpy as np

def xarray_to_model(arr):
    nx = len(arr.x)
    return pyqg.QGModel(nx=nx, log_level=0)

SAMPLE_SLICE = slice(-40, None) # in indices
AVERAGE_SLICE = slice(360*5*DAY,None) # in seconds
AVERAGE_SLICE_ANDREW = slice(44,None) # in indices
ANDREW_1000_STEPS = 3600000

class noise_time_sampler():
    '''
    Class which have memory and implements 
    time smoothing of noise
    '''
    def __init__(self, sampling_type='AR1', nsteps=1):
        self.sampling_type = sampling_type
        self.nsteps = nsteps

    def __call__(self, noise):
        if self.sampling_type == 'AR1':
            return self.AR1_sampling(noise, self.nsteps)
        elif self.sampling_type == 'constant':
            return self.constant_sampling(noise, self.nsteps)

    def AR1_sampling(self, noise, nsteps=1):
        '''
        Sampling from work https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1995.0126
        Predicts noise using AR1 model
        at the next time step.
        nsteps - decorrelation time in time steps
        If nsteps = 1, AR1 is equal to white noise in time sampling
        in case of first call, self.noise is initialized with input noise
        '''
        a = 1 - 1/nsteps
        b = (1/nsteps * (2 - 1/nsteps))**0.5
        if hasattr(self, 'noise'):
            self.noise = a * self.noise + b * noise
        else:
            self.noise = noise
        return self.noise, True
    
    def constant_sampling(self, noise, nsteps):
        '''
        Sampling from work https://www.sciencedirect.com/science/article/pii/S1463500317300100
        '''
        update = True
        if hasattr(self, 'noise'):
            if self.counter % nsteps == 0:
                self.noise = noise
                self.counter = 1
            else:
                self.counter += 1
                update = False
        else:
            self.noise = noise
            self.counter = 1
        return self.noise, update

class stochastic_QGModel(pyqg.QGModel):
    '''
    Accumulate noise for correlated SGS closures
    '''
    def __init__(self, pyqg_params, sampling_type='AR1', nsteps=1):
        super().__init__(**pyqg_params, log_level=0)
        self.noise_sampler = noise_time_sampler(sampling_type, nsteps)

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
    assert len(ds.time.sel(time=time)) < 3/4 * len(ds.time)
    # There are snapshots for PDF
    assert len(ds.time.sel(time=time)) > 1/4 * len(ds.time)

    # Compute PDFs
    x = ds.sel(time=time).isel(lev=0)['q'].values.ravel()
    points, density = PDF_histogram(x, Nbins=100, xmin=-3e-5, xmax=3e-5)
    ds['pdf_pv'] = xr.DataArray(density, dims=['pv'],
        coords=[coord(points, 'potential vorticity, $s^{-1}$')],
        attrs=dict(long_name='PDF of upper level PV'))
    
    KE = ave_lev(0.5*(ds.u**2 + ds.v**2), delta)
    x = KE.sel(time=time).values.ravel()
    points, density = PDF_histogram(x, Nbins=50, xmin=0, xmax=0.005)
    ds['pdf_ke'] = xr.DataArray(density, dims=['ke'],
        coords=[coord(points, 'kinetic energy, $m^2/s^2$')],
        attrs=dict(long_name='PDF of depth-averaged KE'))

    # Compute time-series
    ds['KE'] = xr.DataArray(KE.mean(dim=('run', 'x', 'y')),
        attrs=dict(long_name='depth-averaged kinetic energy, $m^2/s^2$'))

    return ds

def compute_highres_trajectories(ds, pyqg_params, Tmax=90*DAY, output_sampling=1*DAY, nruns=15):
    '''
    Computes trajectory of highres model, and save
    coarsegrained result. Will be used as initial
    condition and target trajectory in esemble forecasting

    ds - dataset with highres data
    '''
    pyqg_high = pyqg_params.copy()
    pyqg_high['nx'] = ds.x.size
    pyqg_high['dt'] = 3600
    pyqg_high['tmax'] = Tmax
    
    # Filter to coarse resolution
    def filter(model):
        op = ppb.coarsening_ops.Operator1(model, pyqg_params['nx'])              
        return op.m2.to_dataset().q
    
    individual_runs = []
    for run in range(nruns):
        q = sample(ds)
        highres = pyqg.QGModel(**pyqg_high, log_level=0)
        highres.set_q1q2(*q.values.astype('float64'))
        snapshots = []
        time = []
        for t in highres.run_with_snapshots(tsnapint = output_sampling):
            snapshots.append(filter(highres))
            time.append(t)
        individual_runs.append(xr.concat(snapshots, dim='time'))
    
    out = xr.Dataset()
    out['q'] = xr.concat(individual_runs, dim='run')
    out['time'] = xr.DataArray(time, dims=['time'])
    out.attrs['pyqg_params'] = str(pyqg_params)
    return out.astype('float32')

def run_simulation(pyqg_params, sampling_type, nsteps,
    sample_interval):
    '''
    Run model m with parameters pyqg_params
    and saves snapshots every sample_interval seconds
    Returns xarray.Dataset with snapshots and 
    averaged statistics
    '''
    m = stochastic_QGModel(pyqg_params, sampling_type, nsteps)
    
    snapshots = []
    for t in m.run_with_snapshots(tsnapint = sample_interval):
        snapshots.append(m.to_dataset().copy(deep=True))

    return concat_in_time(snapshots)

def run_simulation_with_two_models(q, pyqg_low, pyqg_high, 
    sampling_type, nsteps,
    output_sampling=1*DAY, ensemble_size=16):
    '''
    q - initial condition of PV for highres model in form of 
    xarray
    pyqg_low - low resolution model parameters
    pyqg_high - high resolution model parameters
    Tmax - integration time
    Output sampling - how often save snapshots
    ensemble_size - number of stochastic runs from the same initial condition
    '''
    assert q.x.size == pyqg_high['nx']

    def filter(model):
        op = ppb.coarsening_ops.Operator1(model, pyqg_low['nx'])        
        return xr.DataArray(op.m2.q, dims=['lev', 'y', 'x'] )

    lowres_models = [stochastic_QGModel(pyqg_low, sampling_type, nsteps) for _ in range(ensemble_size)]
    highres = pyqg.QGModel(**pyqg_high, log_level=0)

    highres.set_q1q2(*q.values.astype('float64'))
    for model in lowres_models:
        model.set_q1q2(*filter(highres).values)

    lowres_snapshots = []
    highres_snapshots = []

    for t in highres.run_with_snapshots(tsnapint = output_sampling):
        highres_snapshots.append(filter(highres))
    q_highres = xr.concat(highres_snapshots, dim='time')

    model_snapshots = []
    for model in lowres_models:
        lowres_snapshots = []
        for t in model.run_with_snapshots(tsnapint = output_sampling):
            lowres_snapshots.append(model.to_dataset().q.copy(deep=True))
        model_snapshots.append(xr.concat(lowres_snapshots, dim='time'))
    q_lowres = xr.concat(model_snapshots, dim='ensemble')

    out = xr.Dataset()
    out['q_mean'] = q_lowres.mean(dim='ensemble').copy()
    out['q_gen'] = q_lowres.isel(ensemble=0).copy()
    out['q_true'] = q_highres.copy()

    return out

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

    Here we assume that dataset has dimensions run x time x lev x Ny x Nx
    '''
    def R2(x, x_true):
        dims = [d for d in x.dims if d != 'lev']
        return float((1 - ((x-x_true)**2).mean(dims) / (x_true).var(dims)).mean())
    
    ds = xr.Dataset()
    # first compute R2 for each layer, and after that normalize
    ds['R2_mean'] = R2(mean, true)

    sp = spectrum(time=slice(None,None)) # power spectrum for full time slice

    ds['sp_true'] = sp(true).astype('float64')
    ds['sp_gen'] = sp(gen).astype('float64')
    ds['R2_total'] = R2(ds.sp_gen, ds.sp_true)
    
    ds['sp_true_res'] = sp(true-mean).astype('float64')
    ds['sp_gen_res'] = sp(gen-mean).astype('float64')
    ds['R2_residual'] = R2(ds.sp_gen_res, ds.sp_true_res)

    return ds

class Parameterization(pyqg.QParameterization):
    def predict(self, ds, noise):
        '''
        ds - dataset having dimensions (lev, y, x);
        Additional dimensions run and time are arbitrary
        noise - input numpy array with noise of pattern:
        n_stoch_channels x Ny x Nx
        ds should contain input fields: self.inputs
        Output: dataset with fields: self.targets
        Additionally: mean and var are returned!
        '''
        raise NotImplementedError
    
    def nst_ch(self):
        '''
        Defines the number of stochastic channels
        used in online simulation
        '''
        raise NotImplementedError
    
    def __call__(self, m: stochastic_QGModel):
        '''
        m - instance of pyqg.QGModel
        return numpy array with prediction of 
        PV forcing
        '''
        # generate new portion of noise
        noise = np.random.randn(self.nst_ch(), m.ny, m.nx)[np.newaxis, :]
        noise, update = m.noise_sampler(noise)

        demean = lambda x: x - x.mean(axis=(1,2), keepdims=True)

        if update:
            ds = xr.Dataset()
            for var in self.inputs:
                ds[var] = xr.DataArray(m.__getattribute__(var), dims=('lev', 'y', 'x'))
                m.return_data = demean(self.predict(ds, noise)[self.targets[0]].values)
        
        return m.return_data

    @timer
    def test_online(self, pyqg_params=EDDY_PARAMS, sampling_type='AR1', nsteps=1, 
        nruns=5, sample_interval=ANDREW_1000_STEPS):
        '''
        Run ensemble of simulations with parameterization
        and save statistics 
        '''
        delta = pyqg.QGModel(**pyqg_params).delta

        params = pyqg_params.copy()
        params['parameterization'] = self
        
        set_start_method('spawn', force=True)
        with Pool(5) as pool:
            pool.starmap(init_seeds, [()]*5)
            datasets = pool.starmap(run_simulation, [(params, sampling_type, nsteps, sample_interval)]*nruns)
        
        out = concat_in_run(datasets, delta=delta, 
            time=slice(params['tavestart'],None)).astype('float32')
        out.attrs['pyqg_params'] = str(pyqg_params)
        return out

    @timer
    def test_ensemble(self, ds: xr.Dataset, pyqg_params=EDDY_PARAMS, sampling_type='AR1', nsteps=1, 
        Tmax=90*DAY, output_sampling=1*DAY, ensemble_size=15, nruns=15):
        '''
        ds - dataset with high-res fields
        Initial conditions are sampled from this dataset.
        pyqg_params - low resolution models parameters
        Tmax - integration time
        Output sampling - how often save snapshots
        ensemble_size - number of stochastic runs from the same initial condition
        nruns - number of different initial conditions
        Total computational cost for lowres model in days:
        Tmax * ensemble_size * nruns.
        For default parameters, slightly more complex than
        test_online.
        '''
        pyqg_low = pyqg_params.copy()
        pyqg_low['parameterization'] = self
        pyqg_low['tmax'] = Tmax

        pyqg_high = pyqg_params.copy()
        pyqg_high['nx'] = ds.x.size
        pyqg_high['dt'] = 3600
        pyqg_high['tmax'] = Tmax

        set_start_method('spawn', force=True)
        q_init = [sample(ds) for run in range(nruns)]

        with Pool(5) as pool:
            pool.starmap(init_seeds, [()]*5)
            snapshots = pool.starmap(run_simulation_with_two_models, 
                zip(q_init, [pyqg_low]*nruns, [pyqg_high]*nruns, [sampling_type]*nruns, 
                [nsteps]*nruns, [output_sampling]*nruns, [ensemble_size]*nruns))

        out = xr.concat(snapshots, dim='run')
        
        s_scores = []
        for t in out.time:
            d = out.sel(time=slice(t,t))
            s_scores.append(subgrid_scores(d.q_true, d.q_mean, d.q_gen))
            
        out.update(xr.concat(s_scores, dim='time'))
        out.attrs['pyqg_params'] = str(pyqg_params)
        
        return out.astype('float32')

    @timer
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
        preds.update(
            subgrid_scores(preds[target], preds[target+'_mean'], 
                preds[target+'_gen'])[['R2_mean', 'R2_total', 'R2_residual']])

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

class ReferenceModel(Parameterization):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs = inputs
        self.targets = targets
    
    def __call__(self, m):
        return m.q*0
    
    def predict(self, ds, noise=None):
        out = xr.Dataset()
        out['q_forcing_advection'] = ds['q'] * 0
        out['q_forcing_advection_mean'] = ds['q'] * 0
        out['q_forcing_advection_var'] = ds['q'] * 0
        return out

    def nst_ch(self):
            return 2    

    def fit(*args, **kw):
        pass

class TrivialStochastic(Parameterization):
    def __init__(self, inputs, targets, amp=1):
        super().__init__()
        self.amp = amp
        self.inputs = inputs
        self.targets = targets

    def predict(self, ds, noise=None):
        if noise is None:
            noise = np.random.randn(ds.q.shape[0]*ds.q.shape[1], *ds.q.shape[2:])
        std = 3e-12
        Y = self.amp * std * noise
        mean = 0*Y
        var = 0*Y
        return xr.merge((
            array_to_dataset(ds, Y, self.targets),
            array_to_dataset(ds, mean, self.targets, postfix='_mean'),
            array_to_dataset(ds, var, self.targets, postfix='_var')
        ))
        
    def fit(*args, **kw):
        pass
        
    def nst_ch(self):
        return 2