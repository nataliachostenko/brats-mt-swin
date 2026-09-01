#!/bin/bash
#SBATCH --job-name=eval_lesion
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/eval_lesion_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri
export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr

cd /work/ab0995/a270263/mgr
python tools/eval_regions_lesion.py
