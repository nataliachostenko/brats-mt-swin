#!/bin/bash
#SBATCH --job-name=eval_official_postprocess_gpu_detach
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/eval_official_postprocess_gpu_detach_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri
export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr

export CHECKPOINT_PATH="/work/ab0995/a270263/mgr/logs/brats-mgr-project/jnopa6kt/checkpoints/epoch=99-step=16200.ckpt"
export EVAL_LIMIT=15
export POSTPROCESS_MIN_VOXELS=100

mkdir -p /work/ab0995/a270263/mgr/scratch/run_detach_postprocess
cd /work/ab0995/a270263/mgr/scratch/run_detach_postprocess
python -u /work/ab0995/a270263/mgr/tools/eval_official_lesion_wise_detach.py model=swin_multitask_detach
