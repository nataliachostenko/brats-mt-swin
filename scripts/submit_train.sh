#!/bin/bash
#SBATCH --job-name=brats_swin_ctx
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/slurm-%j.out
#SBATCH --error=/work/ab0995/a270263/mgr/logs/slurm-%j.err
#SBATCH --account=ab0995

echo "Starting Slurm job: $SLURM_JOB_ID"
echo "Allocated node: $SLURM_JOB_NODELIST"

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri

export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export NCCL_BLOCKING_WAIT=1

echo "Starting PyTorch Lightning"
srun python /work/ab0995/a270263/mgr/train.py

echo "Job finished!"
