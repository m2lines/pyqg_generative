#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=0:0:10
#SBATCH --job-name=test_rocm
#SBATCH --gres=gpu:mi50:1

#SBATCH --output=training.txt
#SBATCH --error=error.txt
echo " "
scontrol show jobid -dd $SLURM_JOB_ID
echo " "
echo "The number of alphafold processes:"
ps -e | grep -i alphafold | wc -l
echo " "
module purge
singularity exec --rocm --overlay /scratch/pp2681/python-container/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/rocm5.1.1-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u test.py"
