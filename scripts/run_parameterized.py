import sys
sys.path.append('../pyqg_generative/tools/')
import os

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR
from slurm_helpers import run_experiment, DEFAULT_HPC

BASIC_FOLDER = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled'
SCRIPT_PATH = '/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'

CONFIGURATION = 'eddy';
MODELS_FOLDER = 'models'
NUM_REALIZATIONS = 5
N_ENS = 10 # number of ensemble members

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

def decor_to_nsteps(decorrelation, dt):
    if decorrelation == 0:
        nsteps = 1
    elif decorrelation < 0:
        nsteps = -1
    elif decorrelation > 0:
        nsteps = int(decorrelation * 3600 / dt)
    return nsteps

for resolution in [48, 64, 96]:
    for operator in ['Operator1', 'Operator2']:
        for folder in ['GZ', 'GAN', 'VAE']:
            print(operator, resolution, folder, CONFIGURATION)
            input()
            _operator = operator+'-'+str(resolution)
            for realization in range(0,NUM_REALIZATIONS):
                model_folder = f'{BASIC_FOLDER}/{MODELS_FOLDER}/' + _operator + '/' + folder+'-'+str(realization)
                script_py = SCRIPT_PATH
                for basic_params, name in zip([EDDY_PARAMS], [f'eddy']):
                    pyqg_params = basic_params.nx(resolution)._update({'tmax': 20*YEAR})
                    
                    nsteps = 1; decorrelation=0; sampling='constant'; model_weight = 1
                    subfolder = name + '-' + sampling + '-' + str(decorrelation)
                    os.system('mkdir -p ' + model_folder + '/' + subfolder)

                    hours = {32:1, 48: 2, 64: 12, 96: 24}[resolution]
                    hpc = DEFAULT_HPC._update({'hours': hours, 
                        'job-name': CONFIGURATION[0]+job_name(folder, operator, resolution),
                        'gres': 'NONE', 'partition': 'cs',
                        'mem': 1, 'ntasks': 1,
                        #'mem': 16, 'ntasks': 8,
                        #'gres': 'gpu:mi50:1',
                        'output': f'{subfolder}/%a.out', 
                        'error':  f'{subfolder}/%a.err',
                        'array': f'0-{str(N_ENS-1)}'})

                    args = {'parameterization': 'yes', 'ensemble_member': '$SLURM_ARRAY_TASK_ID', 'pyqg_params': pyqg_params, 
                        'subfolder': subfolder, 'sampling': sampling, 'nsteps': nsteps, 'model_weight': model_weight}

                    run_experiment(model_folder, script_py, hpc, args, delete=False)