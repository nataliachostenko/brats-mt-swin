#!/bin/bash
#SBATCH --job-name=eval_pr_curves
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/work/ab0995/a270263/mgr/logs/eval_pr_curves_%x_%j.out

eval "$(conda shell.bash hook)"
conda activate /work/ab0995/a270263/env/mri

export LD_LIBRARY_PATH=/work/ab0995/a270263/env/mri/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/ab0995/a270263/mgr

export MODEL_TAG="${MODEL_TAG:-multitask}"
export EVAL_LIMIT="${EVAL_LIMIT:-0}"

RUNDIR="/work/ab0995/a270263/mgr/scratch/run_pr_${MODEL_TAG}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

echo "MODEL_TAG=$MODEL_TAG  EVAL_LIMIT=$EVAL_LIMIT"
python -u /work/ab0995/a270263/mgr/tools/eval_pr_curves.py
