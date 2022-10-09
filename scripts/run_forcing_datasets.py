import sys
sys.path.append('../pyqg_generative/tools/')

from parameters import EDDY_PARAMS, JET_PARAMS
from slurm_helpers import run_experiment, DEFAULT_HPC

def run_simulation(folder, ens, params):
    run_experiment(folder, 
        script_py='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py', 
        hpc=DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 'hours': 1, 'job-name': 'SGS_'+str(ens), 'gres': 'NONE'}),
        args={'forcing': 'yes', 'ensemble_member': ens, 'pyqg_params': str(params)})

#folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy'
#for ens in range(300):
#    run_simulation(folder, ens, EDDY_PARAMS.nx(256))

folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet_300'
for ens in range(300):
    run_simulation(folder, ens, JET_PARAMS.nx(256))