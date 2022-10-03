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

def decor_to_nsteps(decorrelation, dt):
    if decorrelation == 0:
        nsteps = 1
    elif decorrelation < 0:
        nsteps = -1
    elif decorrelation > 0:
        nsteps = int(decorrelation * 3600 / dt)
    return nsteps

ntask = 0
for resolution in [48, 64, 96]:
    for operator in ['Operator1', 'Operator2']:
        if ntask>1500:
            input()
        for model in ['OLSModel', 'MeanVarModel', 'CGANRegression', 'CGANRegression-recompute', 'CGANRegression-Unet', 'CGANRegression-None-recompute', 'CVAERegression-None', 'CVAERegression-None-1000', 'OLSModel-div', 'CGANRegression-div']:
            _operator = operator+'-'+str(resolution)
            for sampling in ['AR1', 'constant']:
                for decorrelation in [0]: # in hours; 0 means tau=dt
                    if sampling=='constant' and decorrelation>0:
                        continue
                    if model=='OLSModel' and sampling=='AR1':
                        continue
                    if sampling=='AR1' and decorrelation==0:
                        continue
                    for model_weight in [0.5]:
                        for basic_params, name in zip([EDDY_PARAMS, EDDY_PARAMS, JET_PARAMS, JET_PARAMS], ['eddy', 'eddy-3600', 'jet', 'jet-3600']):
                            for ens in range(N_ENS):
                                model_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/' + _operator + '/' + model
                                script_py ='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'

                                pyqg_params = basic_params.nx(resolution)._update({'tmax': 20*YEAR})
                                if name in ['eddy-3600', 'jet-3600']:
                                    pyqg_params['dt'] = 3600
                                nsteps = decor_to_nsteps(decorrelation, pyqg_params['dt'])
                                
                                if model_weight == 1:
                                    subfolder = name + '-' + sampling + '-' + str(decorrelation)
                                else:
                                    subfolder = str(model_weight) + '-' + name + '-' + sampling + '-' + str(decorrelation)
                                os.system('mkdir -p ' + model_folder + '/' + subfolder)

                                hours = {32:1, 48: 2, 64: 4, 96: 10}[resolution]

                                ntask += 1
                                print(ntask)
                                hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 2, 'hours': hours, 
                                    'job-name': f'j{str(resolution)}-{str(ntask)}',
                                    'gres': 'NONE', 'partition': 'cs',
                                    'output': f'{subfolder}/out-{ens}.txt', 'error': f'{subfolder}/err-{ens}.txt'})

                                args = {'parameterization': 'yes', 'ensemble_member': ens, 'pyqg_params': pyqg_params, 
                                    'subfolder': subfolder, 'sampling': sampling, 'nsteps': nsteps, 'model_weight': model_weight}

                                run_experiment(model_folder, script_py, hpc, args, delete=False)