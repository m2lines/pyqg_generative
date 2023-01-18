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

def decor_to_nsteps(decorrelation, dt):
    if decorrelation == 0:
        nsteps = 1
    elif decorrelation < 0:
        nsteps = -1
    elif decorrelation > 0:
        nsteps = int(decorrelation * 3600 / dt)
    return nsteps

N_ENS = 10 # number of ensemble members

CONFIGURATION = 'eddy';
MODELS_FOLDER = 'models_retrain'
NUM_REALIZATIONS = 5

#CONFIGURATION = 'jet_300'; TRANSFER = 'eddy'
#MODELS_FOLDER = 'models_jet'
#NUM_REALIZATIONS = 3

for resolution in [48]:
    for operator in ['Operator2']:
        for folder in ['CGANRegression-retrain', 'CVAERegression-None', 'MeanVarModel']:
            print(operator, resolution, folder, CONFIGURATION)
            input()
            _operator = operator+'-'+str(resolution)
            for realization in range(0,NUM_REALIZATIONS):
                model_folder = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{MODELS_FOLDER}/' + _operator + '/' + folder+'-'+str(realization)
                script_py ='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'
                #for dt in [3600*2]:
                    #for basic_params, name in zip([JET_PARAMS], [f'jet-{str(dt)}']):
                    #for basic_params, name in zip([EDDY_PARAMS], [f'eddy-{str(dt)}']):
                for basic_params, name in zip([EDDY_PARAMS], [f'eddy']):
                    pyqg_params = basic_params.nx(resolution)._update({'tmax': 20*YEAR})
                    #pyqg_params = basic_params.nx(resolution)._update({'tmax': 20*YEAR, 'dt': 7200})
                    #pyqg_params = basic_params.nx(resolution)._update({'tmax': 20*YEAR, 'dt': dt})
                    
                    #sampling = 'AR1'
                    for decorrelation in [30]:
                        for sampling in ['constant']:
                            nsteps = decor_to_nsteps(decorrelation, pyqg_params['dt'])
                            model_weight=1
                            #nsteps = 1; decorrelation=0; sampling='deterministic'; model_weight = 1
                            #for model_weight in [0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5]:
                            subfolder = name + '-' + sampling + '-' + str(decorrelation)
                            #subfolder = str(model_weight) + '-' + name + '-' + sampling + '-' + str(decorrelation)
                            #subfolder = name+'-'+'deterministic'
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