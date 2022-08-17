import sys
sys.path.append('../pyqg_generative/tools/')
import os

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR, DAY
from slurm_helpers import run_experiment, DEFAULT_HPC

N_IC = 15 # number of initial conditions
N_ENS = 15 # number of ensemble members for a given IC
NDAYS = 90 # number of days to forecast

ntask = 0
for resolution in [256]:#[48, 64, 96]:
    for operator in ['Default']:#['Operator1', 'Operator2']:
        _operator = operator+'-'+str(resolution)
        for model in ['Reference']:#['OLSModel', 'MeanVarModel', 'CGANRegression']:
            sampling = 'AR1'

            model_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/' + _operator + '/' + model
            os.system('rm -rf ' + model_folder + '/' + 'eddy-forecast')
            for decorrelation in [0, 12, 24, 36, 48]: # in hours; 0 means tau=dt
                if model in ['OLSModel', 'Reference'] and decorrelation > 0:
                    continue
                
                for j_ic in range(N_IC):
                    script_py ='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'

                    pyqg_params = EDDY_PARAMS.nx(resolution)._update({'tmax': NDAYS*DAY})
                    nsteps = int(decorrelation * 3600 / pyqg_params['dt']) if decorrelation > 0 else 1

                    subfolder = 'eddy-forecast/' + sampling + '-' + str(decorrelation)
                    os.system('mkdir -p ' + model_folder + '/' + subfolder)

                    hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 'hours': 0, 
                        'job-name': ntask, 
                        'gres': 'NONE', 'output': f'{subfolder}/out-{j_ic}.txt', 'error': f'{subfolder}/err-{j_ic}.txt'})
                    ntask += 1
                    print('ntask=', ntask)

                    initial_condition = dict(
                        path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_256/[0-9].nc',
                        selector = dict(run=j_ic, time=-1) if j_ic<10 else dict(run=j_ic-10, time=-30),
                        operator = operator,
                        number = j_ic,
                        n_ens = 1 if model in ['OLSModel', 'Reference'] else N_ENS
                    )

                    args = {'forecast': 'yes', 'pyqg_params': pyqg_params, 
                        'subfolder': subfolder, 'sampling': sampling, 'nsteps': nsteps, 'initial_condition': initial_condition}

                    run_experiment(model_folder, script_py, hpc, args, delete=False)