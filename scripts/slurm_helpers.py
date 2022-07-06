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
DEFAULT_HPC = CustomDict({'nodes': 1, 'ntasks': 8, 'cpus': 1, 'mem': 64, 
    'hours': 8, 'job-name': 'pyqg_generative', 'gres': 'gpu', 'mail': 'NONE'})

def create_slurm(script_py, folder, hpc=DEFAULT_HPC, args=DEFAULT_ARGS):
    '''
    Create slurm script launcher.sub in folder

    hpc - dict with HPC parameters
    args - dict with arguments to the script
    script_py - name of the Python script to run (absolute path)
    folder - folder where to save the slurm script launcher.sub
    '''
    lines = [
        '#!/bin/bash',
        '#SBATCH --nodes='+hpc['nodes'],
        '#SBATCH --ntasks-per-node='+hpc['ntasks'],
        '#SBATCH --cpus-per-task='+hpc['cpus'],
        '#SBATCH --mem='+hpc['mem']+'GB',
        '#SBATCH --time='+hpc['hours']+':00:00',
        '#SBATCH --job-name='+hpc['job-name'],
        '#SBATCH --gres='+hpc['gres'],
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

def run_experiment(params, folder):
    '''
    
    '''
    os.system('rm -rf '+ folder)
    os.system('mkdir -p '+ folder)
    os.system('mkdir -p '+ folder+'/checkpoints')

    os.system('cp -r ../models '+ folder+'/models')
    os.system('cp -r ../trainers '+ folder+'/trainers')
    os.system('cp -r ../tools '+ folder+'/tools')
    
    create_slurm(params, folder+'/launcher.sub')
    os.system('cd '+folder+'; sbatch launcher.sub')

