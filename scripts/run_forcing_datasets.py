import sys
sys.path.append('../pyqg_generative/tools/')

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR, ANDREW_1000_STEPS
from slurm_helpers import run_experiment, DEFAULT_HPC

BASIC_FOLDER = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy'
SCRIPT_PATH = '/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'
PARAMETERS = EDDY_PARAMS
NRUNS = 300

#BASIC_FOLDER = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet'
#SCRIPT_PATH = '/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py'
#PARAMETERS = JET_PARAMS
#NRUNS = 25

def run_simulation(folder, ens, params):
    run_experiment(folder, 
        script_py=SCRIPT_PATH, 
        hpc=DEFAULT_HPC._update({'ntasks': 1, 'mem': 16, 'hours': 10, 
            'output': str(ens)+'.out', 'error': str(ens)+'.err', 'gres': 'NONE'}),
        args={'forcing': 'yes', 'ensemble_member': ens, 'pyqg_params': str(params), 'sampling_freq': ANDREW_1000_STEPS})

for ens in range(NRUNS):
   run_simulation(BASIC_FOLDER, ens, PARAMETERS.nx(256))