import sys
import os
sys.path.append('../pyqg_generative/tools/')

from parameters import EDDY_PARAMS, JET_PARAMS, YEAR
from slurm_helpers import run_experiment, DEFAULT_HPC

folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy'
for nc in [48, 64]:
    for dt in [7200, 28800]:
        print(nc, dt)
        input()
        subfolder = f'reference_{str(dt)}_{str(nc)}'
        os.system('mkdir -p ' + os.path.join(folder,subfolder))
        for ens in range(10):
            run_experiment(folder, 
            script_py='/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py', 
            hpc=DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 'hours': 24, 
                'job-name': f'r{str(nc)}_{str(dt)}', 'gres': 'NONE', 'partition': 'cs',
                'output': f'{subfolder}/out-{ens}.txt', 'error': f'{subfolder}/err-{ens}.txt'}),
            args={'reference': 'yes', 'subfolder' : subfolder, 'ensemble_member': ens, 
                'pyqg_params': str(EDDY_PARAMS._update({'tmax': 20*YEAR, 'nx': nc, 'dt': dt}))})

#folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet'
#for ens in range(10):
#    run_simulation(folder, ens, JET_PARAMS._update({'tmax': 20*YEAR}))