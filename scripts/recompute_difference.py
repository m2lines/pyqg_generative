import sys
import os
import glob

from slurm_helpers import run_experiment, DEFAULT_HPC

#eddy configuration at default time step
#CONFIGURATION = 'jet'
#REFERENCE_FOLDER = 'jet'
#MODELS_FOLDER = 'models_jet'
#NUM_REALIZATIONS = 3

CONFIGURATION = 'eddy'
REFERENCE_FOLDER = 'eddy'
MODELS_FOLDER = 'models_retrain'
NUM_REALIZATIONS = 1

#CONFIGURATION = 'jet'
#REFERENCE_FOLDER = 'jet_40years'
#MODELS_FOLDER = 'models_jet_40years'
#NUM_REALIZATIONS = 1

PHYSICAL_PARAMETERIZATIONS=False

nfile = 0
for res in [48]:
    for operator in ['Operator1']:
        for model in ['CGANRegression-retrain', 'CVAERegression-None']:
        #for model in ['CGANRegression-retrain']:
        #for model in ['ZannaBolton', 'ReynoldsStress', 'HybridSymbolic', 'ADM', 'BackscatterEddy', 'BackscatterJet']:
            for realization in range(NUM_REALIZATIONS):
                _operator = operator+'-'+str(res)
                base = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{MODELS_FOLDER}/'
                if PHYSICAL_PARAMETERIZATIONS:
                    model_folder = base + '/' + 'Operator1'+'-'+str(res) + '/' + model + '-' + str(realization)
                else:
                    model_folder = base + '/' + _operator + '/' + model + '-' + str(realization)

                for timestep in ['-7200', '']:
                #for decorrelation in ['0', '12', '24', '36', '48']:
                    nfile += 1
                    #subfolder = f'{CONFIGURATION}-AR1-{decorrelation}'
                    #subfolder = f'{CONFIGURATION}{timestep}-constant-0'
                    subfolder = f'{CONFIGURATION}{timestep}-deterministic'
                    
                    model_path = model_folder + '/' + subfolder + '/[0-9].nc'
                    #key = _operator + '/' + model + '-' + str(realization) + '/' + f'{REFERENCE_FOLDER}{timestep}-constant-0'
                    #key = _operator + '/' + model + '-' + str(realization) + '/' + f'{REFERENCE_FOLDER}-AR1-{decorrelation}'
                    key = _operator + '/' + model + '-' + str(realization) + '/' + f'{REFERENCE_FOLDER}{timestep}-deterministic'
                    target_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{REFERENCE_FOLDER}/reference_256/'+_operator+'.nc'
                    save_file = key.replace('/','-')

                    script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
                    save_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/difference_recompute'
                    
                    hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
                            'hours': 0, 'minutes': 4,
                            'job-name': nfile, 'gres': 'NONE', 
                            'output': f'output/{save_file}.out',
                            'error' : f'output/{save_file}.err',
                            'launcher': f'{save_file}.sh',
                            'partition': 'cs'})
                    
                    args = dict(model_path=model_path, target_path=target_path, key=key, save_file=f'output/{save_file}.json')
                    run_experiment(save_folder, script_py, hpc, args, delete=False)

# Reference simulations
# CONFIGURATION = 'jet'

#CONFIGURATION = 'eddy'

# CONFIGURATION = 'jet_40years'
# nfile = 0
# for res in [48, 64, 96]:
#     for operator in ['Operator1', 'Operator2']:
#         nfile += 1
#         _operator = operator+'-'+str(res)
#         for timestep in ['']:
#             model_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{CONFIGURATION}/reference{timestep}_{str(res)}/[0-9].nc'
#             target_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{CONFIGURATION}/reference_256/'+_operator+'.nc'
#             timestep = timestep.replace('_','-')
#             key = _operator+f'/Reference/{CONFIGURATION}{timestep}'
#             save_file = key.replace('/','-')

#             script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
#             save_folder = '/home/pp2681/pyqg_generative/notebooks/difference_recompute'
            
#             hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
#                     'hours': 0, 'minutes': 4,
#                     'job-name': nfile, 'gres': 'NONE', 
#                     'output': f'output/{save_file}.out',
#                     'error' : f'output/{save_file}.err',
#                     'launcher': f'{save_file}.sh',
#                     'partition': 'cs'})
            
#             args = dict(model_path=model_path, target_path=target_path, key=key, save_file=f'output/{save_file}.json')
#             run_experiment(save_folder, script_py, hpc, args, delete=False)