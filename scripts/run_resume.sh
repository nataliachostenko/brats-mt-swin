#!/bin/bash
#SBATCH --job-name=brats_resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/slurm-resume-%j.out
#SBATCH --error=/work/ab0995/a270263/mgr/logs/slurm-resume-%j.err
#SBATCH --account=ab0995

cd /work/ab0995/a270263/mgr
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Resuming Slurm job in: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri

export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export NCCL_BLOCKING_WAIT=1

srun python train.py \
  trainer.max_epochs=100 \
  +ckpt_path=/work/ab0995/a270263/logs/brats-mgr-project/r7uperj0/checkpoints/resume.ckpt

echo "Job finished!"
