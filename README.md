# Multi-task Swin-UNETR with Cross-Task Attention for Brain Tumor MRI

![Deep Learning](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=flat-square&logo=pytorch)
![Medical AI](https://img.shields.io/badge/Library-MONAI-ade02a?style=flat-square)
![MLOps](https://img.shields.io/badge/MLOps-W%26B-ffbe00?style=flat-square&logo=weightsandbiases)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions)

## Overview

A multi-task deep learning architecture for brain glioma MRI (BraTS 2024/2025 GLI)
that couples semantic segmentation of tumor substructures with classification of
their presence via a Cross-Task Attention (CtA) module that feeds the
classifier's features back into the segmentation decoder.

<p align="center">
  <img src="./assets/demo.gif" width="700">
</p>

Compared architectures:

| Architecture | Config | Purpose |
|---|---|---|
| **Swin-UNETR Base** | `model=swin_base` | Segmentation only baseline |
| **Multi-task (standard)** | `model=swin_multitask` | Segmentation and classification, gradients flow freely between both heads |
| **Multi-task (Detach)** | `model=swin_multitask_detach` | Same architecture, gradients between the classification head and CtA module are blocked to prevent the classification loss from destabilizing the shared encoder |

The classification head predicts presence of three tumor substructures
(NETC, SNFH, ET) per patch. **It is not a malignancy grading task.**

## Installation

Requires Python 3.10+.

```bash
git clone git@github.com:nataliachostenko/brats-mt-swin.git
cd brats-mt-swin

conda create -n brats python=3.10 -y
conda activate brats
pip install -e .
```

## Data

The pipeline expects the BraTS 2024/2025 GLI dataset laid out as one directory
per patient each containing the four MRI modalities (`*-t1n.nii.gz`,
`*-t1c.nii.gz`, `*-t2w.nii.gz`, `*-t2f.nii.gz`) and a segmentation mask
(`*-seg.nii.gz`).

Point the pipeline at your local copy either via an environment variable

```bash
export BRATS_DATA_DIR=/path/to/BraTS-GLI/extracted_data
```

or by overriding it on the command line (see below). It defaults to
`./data/BraTS-GLI/extracted_data/` if neither is set.

The dataset is available at Synapse.org after registration.

## Training

Training is driven by [Hydra](https://hydra.cc/); `configs/train.yaml` composes
the data, model, and trainer configs.

```bash
# default: multi-task, standard (no detach)
python train.py

# a specific variant
python train.py model=swin_base
python train.py model=swin_multitask_detach

# override any config value from the CLI
python train.py model=swin_multitask_detach data.batch_size=4 trainer.devices=2
```

`configs/trainer/default.yaml` is set up for multi-GPU DDP (4 devices) and must be adjusted
`trainer.devices` and `trainer.strategy` for a single-GPU or CPU run.
Logging goes to Weights & Biases; set `WANDB_API_KEY` or run `wandb login`
before training or override `logger` to disable it.

## Tests

`pytest` and `flake8` are already installed via `pip install -e .` above.

```bash
pytest tests/
flake8 src/ tests/
```

Covers Cross-Task Attention output shape and non-triviality, multi-task model
forward-pass shapes, the small-component postprocessing filter, and BraTS label
remapping.

## Project structure

```text
├── .github/workflows/      # CI (flake8 + pytest)
├── configs/                # Hydra configs
│   ├── data/                   # dataset / dataloader
│   ├── model/                  # the 4 compared architectures
│   └── trainer/                # DDP / hardware
├── src/
│   ├── models/
│   │   ├── components/         # BaseSwinUNETR, CrossTaskAttention3D,
│   │   │                       # MultiTaskSwinUNETR(Detach)
│   │   └── brats_module.py     # Lightning training/validation loop
│   ├── data/
│   │   └── brats_datamodule.py # MONAI loading and augmentation pipeline
│   └── utils/
│       └── postprocess.py      # small-component removal
├── tests/
└── train.py                # Hydra entrypoint
```
## Evaluation

All scripts below load a trained checkpoint from [Training](#training). `MODEL_TAG` selects the architecture (`multitask` is always the default); `CHECKPOINT_PATH` overrides the built-in checkpoint path per tag.

### Official lesion-wise metrics (BraTS 2024)

The metric used in the thesis, connected-component matching per the official
BraTS challenge script (`tools/BraTS2024Metrics/`).

```bash
MODEL_TAG=multitask python tools/eval_official_lesion_wise.py   # default
MODEL_TAG=base      python tools/eval_official_lesion_wise.py
MODEL_TAG=detach    python tools/eval_official_lesion_wise.py

# filter predicted components smaller than 100 voxels, full validation set
POSTPROCESS_MIN_VOXELS=100 EVAL_LIMIT=0 MODEL_TAG=detach python tools/eval_official_lesion_wise.py
```

Writes

```text
outputs/official_lesionwise<_base|_detach><_pp_N>_{raw,summary}.csv
```

### Classification head

```bash
MODEL_TAG=multitask python tools/eval_classification.py
MODEL_TAG=detach    python tools/eval_classification.py
```

`base` has no classification head.

Reports MCC / AP / ROC-AUC at patch and volume level
(`src/utils/cls_metrics.py`). `tools/report_classification.py` turns saved
results into a CSV and LaTeX table.

### Precision-recall curves

```bash
MODEL_TAG=multitask python tools/eval_pr_curves.py
```

Produces voxel-wise and lesion-wise precision-recall data
(`src/utils/pr_metrics.py`), plotted by `tools/plot_pr_curves.py`.

### Auxiliary voxel-wise and lesion-wise Dice, Precision, Recall

Non-official metrics, always against the standard multi-task
checkpoint unless overridden with `CHECKPOINT_PATH`:

```bash
python tools/eval_regions.py          # voxel-wise
python tools/eval_regions_lesion.py   # lesion-wise 
python tools/eval_pr_metrics.py       # voxel-wise Dice / Precision / Recall / PR-AUC
```

### Inference

```bash
python tools/predict.py
python tools/predict_for_slicer.py BraTS-GLI-02139-101
```

`predict.py` runs inference for a single hardcoded case (see `__main__`).

`predict_for_slicer.py` writes the outputs for a selected patient to

```text
slicer_cases/<PID>/
```

for visualization in 3D Slicer.

### Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `MODEL_TAG` | official / classification / PR-curve scripts | `multitask` (default) / `base` / `detach` |
| `CHECKPOINT_PATH` | all evaluation and prediction scripts | override the built-in checkpoint for the selected tag |
| `EVAL_LIMIT` | most evaluation scripts | evaluate only the first N patients |
| `POSTPROCESS_MIN_VOXELS` | `eval_official_lesion_wise.py` | drop predicted components smaller than N voxels |
| `PATCHES_PER_PATIENT` | `eval_classification.py` | patches sampled per patient during patch-level evaluation |

### Cluster configs

`scripts/` contains the SLURM launchers used to produce the thesis results. Some examples:

```bash
sbatch --export=ALL,MODEL_TAG=detach scripts/eval_official_lesion_wise_gpu_detach.sh
sbatch --export=ALL,MODEL_TAG=multitask scripts/eval_classification_slurm.sh
sbatch --export=ALL,MODEL_TAG=base scripts/eval_pr_curves_slurm.sh
```


## License

MIT. see [LICENSE](LICENSE).
