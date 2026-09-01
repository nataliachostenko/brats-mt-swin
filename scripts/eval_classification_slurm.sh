#!/bin/bash
#SBATCH --job-name=eval_cls
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/eval_cls_%x_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri

export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr

export MODEL_TAG="${MODEL_TAG:-multitask}"
export EVAL_LIMIT="${EVAL_LIMIT:-0}"
export PATCHES_PER_PATIENT="${PATCHES_PER_PATIENT:-8}"

RUNDIR="/work/ab0995/a270263/mgr/scratch/run_cls_${MODEL_TAG}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

echo "MODEL_TAG=$MODEL_TAG EVAL_LIMIT=$EVAL_LIMIT PATCHES=$PATCHES_PER_PATIENT"
python -u /work/ab0995/a270263/mgr/tools/eval_classification.py
