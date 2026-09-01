#!/bin/bash
#SBATCH --job-name=eval_lesion_cpu
#SBATCH --account=ab0995
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/eval_lesion_cpu_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri
export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr

cd /work/ab0995/a270263/mgr
python tools/eval_regions_lesion.py
