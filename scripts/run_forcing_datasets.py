import sys
sys.path.append('../pyqg_generative/tools/')

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR, ANDREW_1000_STEPS
from slurm_helpers import run_experiment, DEFAULT_HPC

def run_simulation(folder, ens, params):
    run_experiment(folder, 
        script_py='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py', 
        hpc=DEFAULT_HPC._update({'ntasks': 1, 'mem': 16, 'hours': 10, 
            'output': str(ens)+'.out', 'error': str(ens)+'.err', 'gres': 'NONE'}),
        args={'forcing': 'yes', 'ensemble_member': ens, 'pyqg_params': str(params), 'sampling_freq': 4*ANDREW_1000_STEPS})

#folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy'
#for ens in range(300):
#    run_simulation(folder, ens, EDDY_PARAMS.nx(256))

folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet_40years'
for ens in range(300):
    run_simulation(folder, ens, JET_PARAMS.nx(256)._update({'tmax': 40 * YEAR}))