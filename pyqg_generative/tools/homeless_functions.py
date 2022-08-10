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