#!/bin/bash
#SBATCH --job-name=brats_swin_base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/slurm-base-%j.out
#SBATCH --error=/work/ab0995/a270263/mgr/logs/slurm-base-%j.err
#SBATCH --account=ab0995

cd /work/ab0995/a270263/mgr
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri

export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export NCCL_BLOCKING_WAIT=1
export PYTHONUNBUFFERED=1

echo "Starting PyTorch Lightning (model=swin_base, logger.name=Swin-UNETR-Base)"
srun python -u train.py model=swin_base logger.name="Swin-UNETR-Base" ckpt_path='"/work/ab0995/a270263/logs/brats-mgr-project/4y57l2mg/checkpoints/epoch=6-step=1134.ckpt"'

echo "Job finished!"
