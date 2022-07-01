#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100
#SBATCH --job-name=CGAN
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

FOLDER=/scratch/pp2681/pyqg_NN/CGAN_residual_track_Jun06/EXP1

rm -rf $FOLDER
mkdir -p $FOLDER

cp -r ../models $FOLDER/models
cp -r ../trainers $FOLDER/trainers
cp -r ../tools/ $FOLDER/tools
cd $FOLDER
mkdir checkpoints

module purge

singularity exec --nv \
	    --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
		/bin/bash -c "source /ext3/env.sh; time python -u trainers/train_CGAN.py --n_latent=2 --minibatch_discrimination=1 --lambda_MSE_mean=0 --lambda_MSE_sample=0 --ncritic=5 --training='DCGAN' --generator='Andrew' --discriminator='DCGAN' --deterministic=0 --loss_type='WGAN' --bn='BatchNorm' --GP_shuffle=0 --num_epochs=300 --residual=0 > out.txt"
		
		#/bin/bash -c "source /ext3/env.sh; time python -u trainers/train_mean_var_model.py > out.txt"
		
		#/bin/bash -c "source /ext3/env.sh; time python -u train_conditional.py --num_epochs=200 --ensemble_size=100 --loss_type='GAN' --minibatch_discrimination=0 --deterministic=0 --MSE_mean_alpha=0 --MSE_sample_alpha=0 --configuration='eddies' --folder='.' > out.txt"

		#/bin/bash -c "source /ext3/env.sh; time python -u train_conditional.py --n_latent=1 --num_epochs=3000 --ensemble_size=100 --learning_rate=0.0001 --var_channel='logvar' --decoder_var='adaptive' --small_weights='' --configuration='eddies' --folder='.' > out.txt"
