import os

class CustomDict(dict):
    def _update(self, d):
        '''
        Copy, modify and return new
        dictionary
        '''
        dd = self.copy()
        dd.update(d)
        return dd

DEFAULT_ARGS = CustomDict({})
DEFAULT_HPC = CustomDict({'nodes': 1, 'ntasks': 14, 'cpus': 1, 'mem': 64, 
    'hours': 48, 'job-name': 'pyqg_generative', 'gres': 'gpu', 'mail': 'NONE'})

def create_slurm(folder, script_py, hpc=DEFAULT_HPC, args=DEFAULT_ARGS):
    '''
    Create slurm script launcher.sub in folder

    folder - folder where to save the slurm script launcher.sub
    hpc - dict with HPC parameters
    args - dict with arguments to the script
    script_py - name of the Python script to run (absolute path)
    '''
    for key in hpc.keys():
        hpc[key] = str(hpc[key])
    lines = [
        '#!/bin/bash',
        '#SBATCH --nodes='+hpc['nodes'],
        '#SBATCH --ntasks-per-node='+hpc['ntasks'],
        '#SBATCH --cpus-per-task='+hpc['cpus'],
        '#SBATCH --mem='+hpc['mem']+'GB',
        '#SBATCH --time='+hpc['hours']+':00:00',
        '#SBATCH --job-name='+hpc['job-name'],
        '#SBATCH --gres='+hpc['gres'] if hpc['gres'] != 'NONE' else '',
        '#SBATCH --output='+os.path.join(folder, 'out.slurm'),
        '#SBATCH --error='+os.path.join(folder, 'err.slurm'),
        '#SBATCH --mail-user=pp2681@nyu.edu',
        '#SBATCH --mail-type='+hpc['mail'],
        'module purge'
    ]
    singularity = 'singularity exec --nv ' \
        + '--overlay /scratch/pp2681/python-container/python-overlay.ext3:ro ' \
        + '/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif ' \
        + '/bin/bash -c "source /ext3/env.sh; time '
    
    python_command = 'python -u ' + script_py
    for key in args.keys():
        python_command += ' --'+key+'='+str(args[key])

    python_command += ' > out.txt"'

    lines.append(singularity + python_command)

    with open(os.path.join(folder, 'launcher.sub'),'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def run_experiment(folder, script_py, hpc=DEFAULT_HPC, args=DEFAULT_ARGS):
    '''
    Creates folder, slurm script and runs slurm script
    All local output will occur in the folder, 
    because slurm script will be run in the folder
    '''
    os.system('rm -rf '+ folder)
    os.system('mkdir -p '+ folder)

    create_slurm(folder, script_py, hpc, args)
    os.system('cd '+folder+'; sbatch launcher.sub')


#run_experiment('/scratch/pp2681/pyqg_generative/trivial_stochastic', 
#    '/home/pp2681/pyqg_generative/scripts/test_parameterization.py', DEFAULT_HPC._update({'gres':'NONE'}))
