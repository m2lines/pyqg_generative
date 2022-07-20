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
    m = pyqg.QGModel(nx=len(ds.x), log_level=0)
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