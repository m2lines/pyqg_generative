import sys
sys.path.append('../pyqg_generative/tools/')
import os

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR
from slurm_helpers import run_experiment, DEFAULT_HPC

def job_name(model, operator, resolution):
    if 'OLS' in model:
        m_str = 'ls'
    elif 'CGAN' in model:
        m_str = 'gan'
    elif 'CVAE' in model:
        m_str = 'vae'
    elif 'MeanVar' in model:
        m_str = 'gz'
    return  dict(Operator1='1', Operator2='2')[operator] + \
      m_str + \
      str(resolution)

N_ENS = 10 # number of ensemble members

#CONFIGURATION = 'eddy';
#MODELS_FOLDER = 'models_retrain'
#NUM_REALIZATIONS = 5

CONFIGURATION = 'jet_300'; TRANSFER = 'eddy'
MODELS_FOLDER = 'models_jet'
NUM_REALIZATIONS = 3

for resolution in [96]:
    for operator in ['Operator1', 'Operator2']:
        print(operator, resolution, CONFIGURATION)
        input()
        for folder in ['OLSModel', 'CVAERegression-None', 'MeanVarModel', 'CGANRegression']:
            _operator = operator+'-'+str(resolution)
            for realization in range(0,NUM_REALIZATIONS):
                model_folder = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{MODELS_FOLDER}/' + _operator + '/' + folder+'-'+str(realization)
                script_py ='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'
                for basic_params, name in zip([JET_PARAMS], ['jet']):
                    pyqg_params = basic_params.nx(resolution)._update({'tmax': 20*YEAR})
                    
                    # Only white noise samplings
                    nsteps = 1; decorrelation=0; sampling='constant'
                    model_weight=1
                    subfolder = name + '-' + sampling + '-' + str(decorrelation)
                    os.system('mkdir -p ' + model_folder + '/' + subfolder)

                    hours = {32:1, 48: 2, 64: 4, 96: 10}[resolution]
                    hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 2, 'hours': hours, 
                        'job-name': CONFIGURATION[0]+job_name(folder, operator, resolution),
                        'gres': 'NONE', 'partition': 'cs',
                        'output': f'{subfolder}/%a.out', 
                        'error':  f'{subfolder}/%a.err',
                        'array': f'0-{str(N_ENS-1)}',
                        'sbatch_args': '--dependency=afterok:25768057,25768058,25768059,25768060,25768061,25768062'})

                    args = {'parameterization': 'yes', 'ensemble_member': '$SLURM_ARRAY_TASK_ID', 'pyqg_params': pyqg_params, 
                        'subfolder': subfolder, 'sampling': sampling, 'nsteps': nsteps, 'model_weight': model_weight}

                    run_experiment(model_folder, script_py, hpc, args, delete=False)