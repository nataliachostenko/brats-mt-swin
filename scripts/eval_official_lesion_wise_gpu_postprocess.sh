#!/bin/bash
#SBATCH --job-name=eval_official_postprocess_gpu
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/eval_official_postprocess_gpu_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri
export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr
export EVAL_LIMIT=15
export POSTPROCESS_MIN_VOXELS=100

mkdir -p /work/ab0995/a270263/mgr/scratch/run_postprocess
cd /work/ab0995/a270263/mgr/scratch/run_postprocess
python -u /work/ab0995/a270263/mgr/tools/eval_official_lesion_wise.py
