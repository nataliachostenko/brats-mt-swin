#!/bin/bash
#SBATCH --job-name=brats_detach_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/slurm-detach-test-%j.out
#SBATCH --error=/work/ab0995/a270263/mgr/logs/slurm-detach-test-%j.err
#SBATCH --account=ab0995

cd /work/ab0995/a270263/mgr
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting test Slurm job: $SLURM_JOB_ID"
echo "Allocated node: $SLURM_JOB_NODELIST"

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri

export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export NCCL_BLOCKING_WAIT=1
export PYTHONUNBUFFERED=1

echo "Starting PyTorch Lightning (model=swin_multitask_detach, logger.name=Swin-UNETR-MT-Detach-Test)"
srun python -u train.py model=swin_multitask_detach logger.name="Swin-UNETR-MT-Detach-Test"

echo "Job finished!"
