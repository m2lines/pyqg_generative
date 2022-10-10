import sys
import os
import glob

from slurm_helpers import run_experiment, DEFAULT_HPC

# eddy configuration at default time step
# base = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models_retrain/'
# nfile = 0
# for res in [48, 64, 96]:
#     for operator in ['Operator1', 'Operator2']:
#         for model in ['OLSModel', 'CGANRegression', 'MeanVarModel', 'CVAERegression-None']:
#             for realization in range(5):
#                 nfile += 1
#                 _operator = operator+'-'+str(res)
#                 model_folder = base + '/' + _operator + '/' + model + '-' + str(realization)
#                 subfolder = 'eddy-constant-0'
                
#                 model_path = model_folder + '/' + subfolder + '/[0-9].nc'
#                 key = _operator + '/' + model + '-' + str(realization) + '/' + subfolder
#                 target_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_256/'+_operator+'.nc'
#                 save_file = key.replace('/','-')

#                 script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
#                 save_folder = '/home/pp2681/pyqg_generative/notebooks/difference_recompute'
                
#                 hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
#                         'hours': 0, 'minutes': 4,
#                         'job-name': nfile, 'gres': 'NONE', 
#                         'output': f'output/{save_file}.out',
#                         'error' : f'output/{save_file}.err',
#                         'launcher': f'{save_file}.sh',
#                         'partition': 'cs'})
                
#                 args = dict(model_path=model_path, target_path=target_path, key=key, save_file=f'output/{save_file}.json')
#                 run_experiment(save_folder, script_py, hpc, args, delete=False)

# Reference simulations
nfile = 0
for res in [48, 64, 96]:
    for operator in ['Operator1', 'Operator2']:
                nfile += 1
                _operator = operator+'-'+str(res)
                model_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_{str(res)}/[0-9].nc'
                target_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_256/'+_operator+'.nc'
                key = _operator+'/Reference/eddy'
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