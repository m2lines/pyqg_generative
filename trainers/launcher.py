import os

# creates slurm script launcher.sub
def create_slurm(params, filename):
    lines = [
    '#!/bin/bash',
    '#SBATCH --nodes=1',
    '#SBATCH --ntasks-per-node=4',
    '#SBATCH --cpus-per-task=1',
    '#SBATCH --mem=64GB',
    '#SBATCH --time=08:00:00',
    '#SBATCH --job-name=CGAN',
    '#SBATCH --gres=gpu:a100',
    'module purge'
    ]
    sing = 'singularity exec --nv ' \
        + '--overlay /scratch/pp2681/python-container/python-overlay.ext3:ro ' \
        + '/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif ' \
        + '/bin/bash -c "source /ext3/env.sh; time '
    python = 'python -u trainers/train_CGAN.py'

    for key in params.keys():
        python += ' --'+key+'='+str(params[key])

    python += ' > out.txt"'
    sing += python

    lines.append(sing)    

    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def run_experiment(params, folder):
    os.system('rm -rf '+ folder)
    os.system('mkdir -p '+ folder)
    os.system('mkdir -p '+ folder+'/checkpoints')

    os.system('cp -r ../models '+ folder+'/models')
    os.system('cp -r ../trainers '+ folder+'/trainers')
    os.system('cp -r ../tools '+ folder+'/tools')
    
    create_slurm(params, folder+'/launcher.sub')
    os.system('cd '+folder+'; sbatch launcher.sub')

folder = '/scratch/pp2681/pyqg_NN/CGAN_regression/'

class CustomDict(dict):
    def _update(self, d):
        dd = self.copy()
        dd.update(d)
        return dd

params = CustomDict()
params.update({
    'n_latent': 2,
    'minibatch_discrimination': 0,
    'deterministic': 0,
    'loss_type': 'WGAN',
    'lambda_MSE_mean': 0.,
    'lambda_MSE_sample': 0.,
    'ncritic': 5,
    'training': 'DCGAN',
    'generator': 'Andrew',
    'discriminator': 'DCGAN',
    'bn': 'None',
    'GP_shuffle': 1,
    'num_epochs': 100,
    'regression': 1,
    'ensemble_size': 50
    })

if os.path.exists(folder):
    print(f'Delete the following subfolders in folder:')
    print(folder)
    print(*os.listdir(folder))
    input('Press any key to accept...')

run_experiment(params._update({'loss_type': 'GAN', 'ncritic': 1}), folder+'EXP0')
run_experiment(params, folder+'EXP1')
run_experiment(params._update({'minibatch_discrimination': 1}), folder+'EXP2')
run_experiment(params._update({'deterministic': 1}), folder+'EXP3')

'''
run_experiment(params._update({'bn': 'BatchNorm'}), folder+'EXP1')
run_experiment(params._update({'bn': 'LayerNorm'}), folder+'EXP2')
run_experiment(params._update({'bn': 'InstanceNorm'}), folder+'EXP3')
run_experiment(params._update({'bn': 'None'}), folder+'EXP4')

run_experiment(params._update({'generator': 'DeepInversion', 'bn': 'BatchNorm'}), folder+'EXP5')
run_experiment(params._update({'generator': 'DeepInversion', 'bn': 'LayerNorm'}), folder+'EXP6')
run_experiment(params._update({'generator': 'DeepInversion', 'bn': 'InstanceNorm'}), folder+'EXP7')
run_experiment(params._update({'generator': 'DeepInversion', 'bn': 'None'}), folder+'EXP8')

run_experiment(params._update({'discriminator': 'DeepInversion', 'bn': 'BatchNorm'}), folder+'EXP9')
run_experiment(params._update({'discriminator': 'DeepInversion', 'bn': 'LayerNorm'}), folder+'EXP10')
run_experiment(params._update({'discriminator': 'DeepInversion', 'bn': 'InstanceNorm'}), folder+'EXP11')
run_experiment(params._update({'discriminator': 'DeepInversion', 'bn': 'None'}), folder+'EXP12')

run_experiment(params._update({'generator': 'DeepInversion', 'discriminator': 'DeepInversion', 'bn': 'BatchNorm'}), folder+'EXP13')
run_experiment(params._update({'generator': 'DeepInversion', 'discriminator': 'DeepInversion', 'bn': 'LayerNorm'}), folder+'EXP14')
run_experiment(params._update({'generator': 'DeepInversion', 'discriminator': 'DeepInversion', 'bn': 'InstanceNorm'}), folder+'EXP15')
run_experiment(params._update({'generator': 'DeepInversion', 'discriminator': 'DeepInversion', 'bn': 'None'}), folder+'EXP16')
'''