import xarray as xr
import numpy as np
import pyqg
import json 

from pyqg_generative.tools.cnn_tools import timer
from pyqg_generative.tools.operators import Operator1, Operator2, Operator4, PV_subgrid_forcing
from pyqg_generative.tools.parameters import ANDREW_1000_STEPS, DAY
from pyqg_generative.tools.stochastic_pyqg import stochastic_QGModel
from pyqg_generative.models.ols_model import OLSModel
from pyqg_generative.models.mean_var_model import MeanVarModel
from pyqg_generative.models.cgan_regression import CGANRegression
from pyqg_generative.models.cgan_regressionxy import CGANRegressionxy
from pyqg_generative.models.cvae_regression import CVAERegression

@timer
def concat_in_time(datasets):
    '''
    Concatenation of snapshots in time:
    - Concatenate everything
    - Store averaged statistics
    - Discard complex vars
    - Reduce precision
    '''
    from time import time
    # Concatenate datasets along the time dimension
    tt = time()
    ds = xr.concat(datasets, dim='time')
    if float(time() - tt) > 100:
        print('Line 27 took', time() - tt, 'seconds')
        print('Len of the datasets', len(datasets), '\n')
        
        print('Individual coords:')
        for d in datasets:
            print(d.coords)
        print('Total coord:')
        print(ds.coords)

        print('\n')
        print('individual variables:')
        for d in datasets:
            print(d.variables)
        print('total variables:')
        print(ds.variables)

        raise ValueError('Concatenation of datasets took too long time')
    
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

    ds = ds.rename({'p': 'psi'}) # Change for conventional name
    ds['time'] = ds.time.values / 86400
    ds['time'].attrs['units'] = 'days'

    return ds

def generate_subgrid_forcing(Nc, pyqg_params, sampling_freq=ANDREW_1000_STEPS):
    '''
    It is assumed that pyqg_params contains basic
    parameters of the simulation (delta, rek, beta)
    and configures DNS model
    and coarse models' resolutions are given 
    by list of Nc[:]
    sampling_freq - time interval of sampling
    subgrid forces, as default given by 1000 time steps of 3600s each

    Returns dictionary of datasets, where each key corresponds to 
    combination of operator and resolution
    '''
    def key(op, nc):
        '''
        Function providing key for a combination
        of operator and coarse resolution
        '''
        return op.__name__ + '-' + str(nc)

    m = pyqg.QGModel(**pyqg_params)

    out = {}
    for t in m.run_with_snapshots(tsnapint=sampling_freq):
        qdns = m.q
        for op in [Operator1, Operator2, Operator4]:
            for nc in Nc:
                forcing, mf = PV_subgrid_forcing(qdns, nc, op, pyqg_params)
                mf = mf.to_dataset()
                forcing = mf.q*0 + forcing # inherit coordinate information
                ds = xr.Dataset({'q_forcing_advection': forcing, 'q': mf.q, 'u': mf.u, 'v': mf.v, 'psi': mf.p}).astype('float32').squeeze()
                ds['time'] = m.t / 86400
                ds['time'].attrs['units'] = 'days'
                try:
                    out[key(op,nc)].append(ds)
                except KeyError:
                    out[key(op,nc)] = [ds]
    for key in out.keys():
        out[key] = xr.concat(out[key], 'time'). \
            assign_attrs({'pyqg_params': str(pyqg_params)}). \
            assign_attrs(**m.to_dataset().attrs)
    return out

@timer
def run_simulation(pyqg_params, parameterization=None, q_init=None,
    sampling_freq=ANDREW_1000_STEPS):
    '''
    pyqg_params - only str-type parameters
    parameterization - dict
    parameterization['self'] = class instance
    parameterization['sampling'] = 'AR1'
    parameterization['nsteps'] = 1
    q_init - initial conditiona for PV, numpy array nlev*ny*nx
    '''
    if parameterization is None:
        m = pyqg.QGModel(**pyqg_params)
    else:
        params = pyqg_params.copy()
        params['parameterization'] = parameterization['self']
        m = stochastic_QGModel(params, parameterization['sampling'],
            parameterization['nsteps'])

    if q_init is not None:
        m.q = q_init.astype('float64')
        m._invert()
        ds = [m.to_dataset().copy(deep=True)] # convenient to have IC saved
    else:
        ds = [] # for backward compatibility

    for t in m.run_with_snapshots(tsnapint=sampling_freq): 
        ds.append(m.to_dataset().copy(deep=True))
    
    out = concat_in_time(ds).astype('float32')
    out.attrs['pyqg_params'] = str(pyqg_params)
    return out

if __name__ ==  '__main__':
    import argparse
    import os
    from pyqg_generative.tools.parameters import ConfigurationDict

    parser = argparse.ArgumentParser()
    parser.add_argument('--pyqg_params', type=str, default=str({}))
    parser.add_argument('--ensemble_member', type=int, default=0)
    parser.add_argument('--forcing', type=str, default="no")
    parser.add_argument('--reference', type=str, default="no")
    parser.add_argument('--parameterization', type=str, default="no")
    parser.add_argument('--forecast', type=str, default="no")
    parser.add_argument('--subfolder', type=str, default="")
    parser.add_argument('--sampling', type=str, default="AR1")
    parser.add_argument('--nsteps', type=int, default=1)
    parser.add_argument('--initial_condition', type=str, default="no")
    parser.add_argument('--model_weight', type=float, default=1.0)
    args = parser.parse_args()
    print(args)
    
    if args.forcing == "yes":
        Nc = [32, 48, 64, 96]

        datasets = generate_subgrid_forcing(Nc, eval(args.pyqg_params))
        for key in datasets.keys():
            os.system('mkdir -p '+ key)
            datasets[key].to_netcdf(os.path.join(key, f'{args.ensemble_member}.nc'))
    
    if args.reference == "yes":
        print(args.pyqg_params)
        run_simulation(eval(args.pyqg_params)).to_netcdf(
            os.path.join(args.subfolder, f'{args.ensemble_member}.nc')
        )

    if args.parameterization == "yes":
        with open('model/model_args.json') as file:
            model_args = json.load(file)

        model = args.model_weight * eval(model_args.pop('model'))(**model_args)
        parameterization = \
            dict(self=model, sampling=args.sampling, nsteps=args.nsteps)
        
        os.system('mkdir -p '+args.subfolder)
        run_simulation(eval(args.pyqg_params), parameterization).to_netcdf(
            os.path.join(args.subfolder, f'{args.ensemble_member}.nc')
        )

    if args.forecast == 'yes':
        if os.path.exists('model/model_args.json'):
            with open('model/model_args.json') as file:
                model_args = json.load(file)

            model = args.model_weight * eval(model_args.pop('model'))(**model_args)
            parameterization = \
                dict(self=model, sampling=args.sampling, nsteps=args.nsteps)
        else:
            parameterization = None

        initial_condition = eval(args.initial_condition)
        pyqg_params = eval(args.pyqg_params)
        path = initial_condition['path']+str(initial_condition['selector']['run'])+'.nc'
        q_init = xr.open_dataset(path).isel(time=initial_condition['selector']['time']).q.values
        try:
            q_init = eval(initial_condition['operator'])(q_init, pyqg_params['nx'])
            print('Operator is applied')
        except:
            print('Operator is not applied')
        
        print('q_init type = ', type(q_init))
        print('q_init shape = ', q_init.shape)

        ds = []
        for j_ens in range(initial_condition['n_ens']):
            print('Start ensemble member ', j_ens)
            ds.append(run_simulation(pyqg_params, parameterization, q_init, 1*DAY)[['q', 'u', 'v', 'psi']])

        print('Concat in runs ')
        ds = xr.concat(ds, 'run')

        print('Compute mean')
        out = xr.Dataset()
        for var in ['q', 'u', 'v', 'psi']:
            out[var] = ds[var].isel(run=0)
            out[var+'_mean'] = ds[var].mean('run')

        print('Saving to file')
        out.to_netcdf(os.path.join(args.subfolder, f'{initial_condition["number"]}.nc'))