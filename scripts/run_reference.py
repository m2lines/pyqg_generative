import sys
import os
sys.path.append('../pyqg_generative/tools/')

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR, ANDREW_1000_STEPS
from slurm_helpers import run_experiment, DEFAULT_HPC

BASIC_FOLDER = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled'
SCRIPT_PATH = '/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'

N_ENS = 10 # number of ensemble members
PARAMS=EDDY_PARAMS
FOLDER='eddy'

#PARAMS=JET_PARAMS
#FOLDER='jet_40years'

folder = f'{BASIC_FOLDER}/{FOLDER}'
for nc in [48, 64, 96, 128, 256]:
    subfolder = f'reference_{str(nc)}'
    os.system('mkdir -p ' + os.path.join(folder,subfolder))
    for ens in range(N_ENS):
        run_experiment(
            folder, 
            script_py=SCRIPT_PATH, 
            hpc=DEFAULT_HPC._update({'cpus': 1, 'ntasks': 1, 'mem': 8, 'hours': 24, 
                'gres': 'NONE', 'partition': 'cs', 'job-name': f'ref{str(nc)}',
                'output': f'{subfolder}/out-{ens}.txt', 'error': f'{subfolder}/err-{ens}.txt'}),
            args={'reference': 'yes', 'subfolder' : subfolder, 'ensemble_member': ens, 
                'pyqg_params': str(PARAMS.nx(nc)._update({'tmax': 20*YEAR, 'tavestart': 5*YEAR})),
                'sampling_freq': ANDREW_1000_STEPS}
            )