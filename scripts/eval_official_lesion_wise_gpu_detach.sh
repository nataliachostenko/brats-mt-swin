#!/bin/bash
#SBATCH --job-name=eval_official_lesion_wise_gpu_detach
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/eval_official_lesion_wise_gpu_detach_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri
export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr

export MODEL_TAG=detach

mkdir -p /work/ab0995/a270263/mgr/scratch/run_detach_full
cd /work/ab0995/a270263/mgr/scratch/run_detach_full
python -u /work/ab0995/a270263/mgr/tools/eval_official_lesion_wise.py
