from pyqg_generative.tools.operators import Operator1, Operator2, PV_subgrid_forcing
import xarray as xr
import pyqg

def generate_subgrid_forcing(Ndns, Nc, pyqg_params, sampling_freq=1000*3600):
    '''
    It is assumed that pyqg_params contains basic
    parameters of the simulation (delta, rek, beta),
    while resolution is passed directly:
    DNS has resolution Ndns x Ndns
    and coarse models are given by list of Nc[:]
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

    params = pyqg_params.copy()
    params.update(nx=Ndns, log_level=0)
    m = pyqg.QGModel(**params)

    out = {}
    for t in m.run_with_snapshots(tsnapint=sampling_freq):
        qdns = m.q
        for op in [Operator1, Operator2]:
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
        out[key] = xr.concat(out[key], 'time')
        out[key].attrs['pyqg_params'] = str(pyqg_params)
    return out

if __name__ ==  '__main__':
    import argparse
    import os
    DAY = 86400
    YEAR = 360*DAY
    EDDY_PARAMS = {'nx': 64, 'dt': 3600, 'tmax': 10*YEAR, 'tavestart': 5*YEAR}
    JET_PARAMS = {'nx': 64, 'dt': 3600, 'tmax': 10*YEAR, 'tavestart': 5*YEAR, 'rek': 7e-08, 'delta': 0.1, 'beta': 1e-11}
    CUSTOM_PARAMS = {'nx': 256, 'tmax': 3600*1000*4}

    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', type=str, default="eddy")
    parser.add_argument('--ensemble_member', type=int, default=0)
    args = parser.parse_args()

    Ndns = 256
    Nc = [32, 48, 64, 96]
    
    mapper = dict(eddy=EDDY_PARAMS, jet=JET_PARAMS, custom=CUSTOM_PARAMS)

    datasets = generate_subgrid_forcing(Ndns, Nc, mapper[args.configuration])
    for key in datasets.keys():
        os.system('mkdir -p '+ key)
        datasets[key].to_netcdf(os.path.join(key, f'{args.ensemble_member}.nc'))