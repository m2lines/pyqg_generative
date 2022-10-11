import sys
import os
import glob

from slurm_helpers import run_experiment, DEFAULT_HPC

#eddy configuration at default time step
#CONFIGURATION = 'eddy'
#MODELS_FOLDER = 'models_retrain'
#NUM_REALIZATIONS = 5

# CONFIGURATION = 'eddy'
# MODELS_FOLDER = 'models_retrain'
# NUM_REALIZATIONS = 5

# nfile = 0
# for res in [48]:
#     for operator in ['Operator1', 'Operator2']:
#         for model in ['OLSModel', 'CGANRegression', 'MeanVarModel', 'CVAERegression-None']:
#             for realization in range(NUM_REALIZATIONS):
#                 _operator = operator+'-'+str(res)
#                 base = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{MODELS_FOLDER}/'
#                 model_folder = base + '/' + _operator + '/' + model + '-' + str(realization)
#                 for timestep in ['-3600', '-7200', '-28800', '']:
#                     nfile += 1
#                     subfolder = f'{CONFIGURATION}{timestep}-constant-0'
                    
#                     model_path = model_folder + '/' + subfolder + '/[0-9].nc'
#                     key = _operator + '/' + model + '-' + str(realization) + '/' + subfolder
#                     target_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{CONFIGURATION}/reference_256/'+_operator+'.nc'
#                     save_file = key.replace('/','-')

#                     script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
#                     save_folder = '/home/pp2681/pyqg_generative/notebooks/difference_recompute'
                    
#                     hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
#                             'hours': 0, 'minutes': 4,
#                             'job-name': nfile, 'gres': 'NONE', 
#                             'output': f'output/{save_file}.out',
#                             'error' : f'output/{save_file}.err',
#                             'launcher': f'{save_file}.sh',
#                             'partition': 'cs'})
                    
#                     args = dict(model_path=model_path, target_path=target_path, key=key, save_file=f'output/{save_file}.json')
#                     run_experiment(save_folder, script_py, hpc, args, delete=False)

# Reference simulations
# CONFIGURATION = 'jet'

CONFIGURATION = 'eddy'
nfile = 0
for res in [48]:
    for operator in ['Operator1', 'Operator2']:
        nfile += 1
        _operator = operator+'-'+str(res)
        for timestep in ['', '_3600', '_7200', '_28800']:
            model_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{CONFIGURATION}/reference{timestep}_{str(res)}/[0-9].nc'
            target_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{CONFIGURATION}/reference_256/'+_operator+'.nc'
            timestep = timestep.replace('_','-')
            key = _operator+f'/Reference/{CONFIGURATION}{timestep}'
            save_file = key.replace('/','-')

            script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
            save_folder = '/home/pp2681/pyqg_generative/notebooks/difference_recompute'
            
            hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
                    'hours': 0, 'minutes': 4,
                    'job-name': nfile, 'gres': 'NONE', 
                    'output': f'output/{save_file}.out',
                    'error' : f'output/{save_file}.err',
                    'launcher': f'{save_file}.sh',
                    'partition': 'cs'})
            
            args = dict(model_path=model_path, target_path=target_path, key=key, save_file=f'output/{save_file}.json')
            run_experiment(save_folder, script_py, hpc, args, delete=False)