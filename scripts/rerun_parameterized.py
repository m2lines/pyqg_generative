import sys
sys.path.append('../pyqg_generative/tools/')
import os

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR, ANDREW_1000_STEPS
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
    elif 'Zanna' in model:
        m_str = 'zb'
    elif 'Hybrid' in model:
        m_str = 'hb'
    elif 'Reynolds' in model:
        m_str = 'R'
    elif 'ADM' in model:
        m_str = 'adm'
    elif 'Backscatter' in model:
        m_str = 'bs'

    return  dict(Operator1='1', Operator2='2')[operator] + \
      m_str + \
      str(resolution)

N_ENS = 10 # number of ensemble members

CONFIGURATION = 'eddy';
MODELS_FOLDER = 'models_retrain'
params = lambda resolution: EDDY_PARAMS.nx(resolution)._update({'tmax': 20*YEAR})
SAMPLING_FREQ = ANDREW_1000_STEPS
NUM_REALIZATIONS = 1

#CONFIGURATION = 'jet';
#MODELS_FOLDER = 'models_jet_40years'
#params = lambda resolution: JET_PARAMS.nx(resolution)._update({'tmax': 80*YEAR, 'tavestart': 20*YEAR})
#SAMPLING_FREQ = 4*ANDREW_1000_STEPS
#NUM_REALIZATIONS = 1

#CONFIGURATION = 'jet';
#MODELS_FOLDER = 'models_jet'
#params = lambda resolution: JET_PARAMS.nx(resolution)._update({'tmax': 20*YEAR})
#SAMPLING_FREQ = ANDREW_1000_STEPS
#NUM_REALIZATIONS = 1



for resolution in [48, 64, 96]:
    for operator in ['Operator1']:
        print(operator, resolution, MODELS_FOLDER)
        input()
        for folder in ['ZannaBolton', 'ReynoldsStress', 'HybridSymbolic', 'ADM', 'BackscatterEddy', 'BackscatterJet']:
            _operator = operator+'-'+str(resolution)
            for realization in range(0,NUM_REALIZATIONS):
                model_folder = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{MODELS_FOLDER}/' + _operator + '/' + folder+'-'+str(realization)
                script_py ='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'
                
                pyqg_params = params(resolution)
                
                # Only white noise samplings
                nsteps = 1; decorrelation=0; sampling='constant'
                model_weight=1
                subfolder = CONFIGURATION + '-' + sampling + '-' + str(decorrelation)
                os.system('mkdir -p ' + model_folder + '/' + subfolder)

                hours = {32:1, 48: 6, 64: 6, 96: 10}[resolution]
                ntasks = 1 if resolution < 96 else 4
                hpc = DEFAULT_HPC._update({'ntasks': ntasks, 'mem': 2, 'hours': hours, 
                    'job-name': CONFIGURATION[0]+job_name(folder, operator, resolution),
                    'gres': 'NONE', 'partition': 'cs',
                    'output': f'{subfolder}/%a.out', 
                    'error':  f'{subfolder}/%a.err',
                    'array': f'0-{str(N_ENS-1)}'})

                args = {'parameterization': folder, 'ensemble_member': '$SLURM_ARRAY_TASK_ID', 'pyqg_params': pyqg_params, 
                    'subfolder': subfolder, 'sampling': sampling, 'nsteps': nsteps, 'model_weight': model_weight,
                    'sampling_freq': SAMPLING_FREQ}

                run_experiment(model_folder, script_py, hpc, args, delete=False)