import sys
sys.path.append('../pyqg_generative/tools/')

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR
from slurm_helpers import run_experiment, DEFAULT_HPC

def run_simulation(folder, ens, params):
    run_experiment(folder, 
        script_py='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py', 
        hpc=DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 'hours': 1, 'job-name': 'SGS_'+str(ens), 'gres': 'NONE'}),
        args={'reference': 'yes', 'ensemble_member': ens, 'pyqg_params': str(params)})

folder = '/scratch/pp2681/pyqg_generative/SGS_forces_white/eddy'
for ens in range(10):
    run_simulation(folder, ens, EDDY_PARAMS._update({'tmax': 20*YEAR}))

folder = '/scratch/pp2681/pyqg_generative/SGS_forces_white/jet'
for ens in range(10):
    run_simulation(folder, ens, JET_PARAMS._update({'tmax': 20*YEAR}))