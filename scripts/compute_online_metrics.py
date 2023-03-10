import sys
import os
import glob

from slurm_helpers import run_experiment, DEFAULT_HPC

BASIC_FOLDER = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled'
SCRIPT_PATH = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
SAVE_FOLDER = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/online_errors'

CONFIGURATION = 'eddy'
REFERENCE_FOLDER = 'eddy'
MODELS_FOLDER = 'models'
NUM_REALIZATIONS = 5

PHYSICAL_PARAMETERIZATIONS=False

nfile = 0
for res in [48, 64, 96]:
    for operator in ['Operator1', 'Operator2']:
        for model in ['GZ', 'GAN', 'VAE']:
            for realization in range(NUM_REALIZATIONS):
                _operator = operator+'-'+str(res)
                base = f'{BASIC_FOLDER}/{MODELS_FOLDER}/'
                if PHYSICAL_PARAMETERIZATIONS:
                    model_folder = base + '/' + 'Operator1'+'-'+str(res) + '/' + model + '-' + str(realization)
                else:
                    model_folder = base + '/' + _operator + '/' + model + '-' + str(realization)

                for timestep in ['']:
                    nfile += 1
                    subfolder = f'{CONFIGURATION}{timestep}-constant-0'
                    
                    model_path = model_folder + '/' + subfolder + '/[0-9].nc'
                    key = _operator + '/' + model + '-' + str(realization) + '/' + f'{REFERENCE_FOLDER}{timestep}-constant-0'
                    target_path = f'/{BASIC_FOLDER}/{REFERENCE_FOLDER}/reference_256/'+_operator+'.nc'
                    save_file = key.replace('/','-')
                    
                    hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
                            'hours': 0, 'minutes': 4,
                            'job-name': nfile, 'gres': 'NONE', 
                            'output': f'output/{save_file}.out',
                            'error' : f'output/{save_file}.err',
                            'launcher': f'{save_file}.sh',
                            'partition': 'cs'})
                    
                    args = dict(model_path=model_path, target_path=target_path, key=key, save_file=f'output/{save_file}.json')
                    run_experiment(SAVE_FOLDER, SCRIPT_PATH, hpc, args, delete=False)