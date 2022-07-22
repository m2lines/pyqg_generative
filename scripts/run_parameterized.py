import sys
sys.path.append('../pyqg_generative/tools/')
import os

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR
from slurm_helpers import run_experiment, DEFAULT_HPC

N_ENS = 10 # number of ensemble members

def job_name(*A):
    out = ''
    for a in A:
        out += str(a)[0]
    return out
    
for resolution in [32, 48, 64, 96]:
    for operator in ['Operator1', 'Operator2']:
        for model in ['OLSModel', 'MeanVarModel', 'CGANRegression']:
            _operator = operator+'-'+str(resolution)
            for sampling in ['AR1', 'constant']:
                for decorrelation in [0, 24, 36, 48]: # in hours; 0 means tau=dt
                    if model=='OLSModel' and sampling=='AR1':
                        continue
                    if sampling=='AR1' and decorrelation==0:
                        continue
                    for basic_params, name in zip([EDDY_PARAMS, JET_PARAMS], ['eddy', 'jet']):
                        for ens in range(N_ENS):
                            model_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/' + _operator + '/' + model
                            script_py ='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'
                            hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 'hours': 1, 
                                'job-name': job_name(resolution, operator, model, sampling, decorrelation, ens), 'gres': 'NONE'})

                            pyqg_params = basic_params.nx(resolution)._update({'tmax': 20*YEAR})
                            nsteps = int(decorrelation * 3600 / pyqg_params['dt']) if decorrelation > 0 else 1
                            subfolder = name + '-' + sampling + '-' + str(decorrelation)

                            args = {'parameterization': 'yes', 'ensemble_member': ens, 'pyqg_params': pyqg_params, 
                                'subfolder': subfolder, 'sampling': sampling, 'nsteps': nsteps}

                            run_experiment(model_folder, script_py, hpc, args, delete=False)