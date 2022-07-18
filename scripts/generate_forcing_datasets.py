from slurm_helpers import run_experiment, DEFAULT_HPC

folder = '/scratch/pp2681/pyqg_generative/SGS_forces/eddy'
for ens in range(300):
    run_experiment(folder, 
        '/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py', 
        DEFAULT_HPC._update({'ntasks': 1, 'mem': 1, 'hours': 1,
        'job-name': 'SGS_'+str(ens), 'gres': 'NONE'}),
        {'ensemble_member': ens, 'configuration': 'eddy'})

folder = '/scratch/pp2681/pyqg_generative/SGS_forces/jet'
for ens in range(300):
    run_experiment(folder, 
        '/home/pp2681/pyqg_generative/pyqg_generative/tools/simulate.py', 
        DEFAULT_HPC._update({'ntasks': 1, 'mem': 1, 'hours': 1,
        'job-name': 'SGSJ_'+str(ens), 'gres': 'NONE'}),
        {'ensemble_member': ens, 'configuration': 'jet'})