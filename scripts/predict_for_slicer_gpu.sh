#!/bin/bash
#SBATCH --job-name=predict_for_slicer
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/predict_for_slicer_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri
export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr

export CHECKPOINT_PATH="/work/ab0995/a270263/mgr/logs/brats-mgr-project/jnopa6kt/checkpoints/epoch=99-step=16200.ckpt"

cd /work/ab0995/a270263/mgr
python -u tools/predict_for_slicer.py "$@"
