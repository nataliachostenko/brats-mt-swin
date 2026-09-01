"""Runs Detach-variant inference for selected patients and writes a
slicer_cases/<PID>/ folder (modalities, GT, prediction, metrics.txt) for
3D Slicer.

Usage:
    python tools/predict_for_slicer.py BraTS-GLI-02139-101 [BraTS-GLI-...]
"""

import os
import sys
import glob
import shutil
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import typing
import omegaconf
import omegaconf.base

torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from monai.inferers import sliding_window_inference  # noqa: E402
from monai.transforms import (  # noqa: E402
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
)

from src.models.brats_module import BraTSLightningModule  # noqa: E402
from src.models.components.multi_task_swin_detach import MultiTaskSwinUNETRDetach  # noqa: E402

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "/work/ab0995/a270263/mgr/logs/brats-mgr-project/jnopa6kt/checkpoints/epoch=99-step=16200.ckpt",
)
DATA_ROOT = os.path.join(project_root, "data", "BraTS-GLI", "extracted_data")
OUT_ROOT = os.environ.get("OUT_ROOT", os.path.join(project_root, "slicer_cases"))

# BraTS 2024 GLI labels after the datamodule's remapping: 1=NETC, 2=SNFH (edema), 3=ET
LABEL_NAMES = {1: "NETC (necrotic/non-enhancing core)", 2: "SNFH (edema)", 3: "ET (enhancing tumor)"}


def find_patient_dir(pid: str) -> str:
    hits = glob.glob(os.path.join(DATA_ROOT, "*", pid))
    if not hits:
        raise FileNotFoundError(f"Patient directory {pid} not found in {DATA_ROOT}")
    return hits[0]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else 2.0 * float((a & b).sum()) / float(s)


def build_model(device):
    net = MultiTaskSwinUNETRDetach(in_channels=4, num_classes=4, feature_size=48)
    model = BraTSLightningModule.load_from_checkpoint(
        CHECKPOINT_PATH, net=net, map_location=device
    )
    model.to(device)
    model.eval()
    return model


def predict_patient(model, device, pid: str):
    patient_dir = find_patient_dir(pid)
    out_dir = os.path.join(OUT_ROOT, pid)
    os.makedirs(out_dir, exist_ok=True)

    mods = ["t1n", "t1c", "t2w", "t2f"]
    img_paths = [os.path.join(patient_dir, f"{pid}-{m}.nii.gz") for m in mods]
    gt_path = os.path.join(patient_dir, f"{pid}-seg.nii.gz")
    for p in img_paths + [gt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    data = transforms({"image": img_paths})
    image = data["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        def predictor_wrapper(x):
            out = model.net(x)
            return out[0] if isinstance(out, tuple) else out

        logits = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=predictor_wrapper,
            overlap=0.5,
        )
        pred = torch.softmax(logits, dim=1).argmax(dim=1)

    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

    gt_img = nib.load(gt_path)
    gt_np = np.asarray(gt_img.dataobj).astype(np.uint8)
    if pred_np.shape != gt_np.shape:
        raise RuntimeError(f"{pid}: prediction shape {pred_np.shape} != GT {gt_np.shape}")

    header = gt_img.header.copy()
    header.set_data_dtype(np.uint8)
    pred_img = nib.Nifti1Image(pred_np, gt_img.affine, header)
    nib.save(pred_img, os.path.join(out_dir, f"{pid}-seg-PRED.nii.gz"))

    for m, p in zip(mods, img_paths):
        shutil.copy2(p, os.path.join(out_dir, f"{pid}-{m}.nii.gz"))
    shutil.copy2(gt_path, os.path.join(out_dir, f"{pid}-seg-GT.nii.gz"))

    lines = [f"Patient: {pid}", f"Source directory: {patient_dir}",
             f"Checkpoint: {CHECKPOINT_PATH}", ""]
    lines.append(f"{'Label':<34}{'GT [vox.]':>12}{'PRED [vox.]':>13}{'Dice':>8}")
    for lab, name in LABEL_NAMES.items():
        g, p_ = (gt_np == lab), (pred_np == lab)
        lines.append(f"{name:<34}{int(g.sum()):>12}{int(p_.sum()):>13}{dice(g, p_):>8.4f}")
    regions = {
        "WT (1+2+3)": (np.isin(gt_np, [1, 2, 3]), np.isin(pred_np, [1, 2, 3])),
        "TC (1+3)": (np.isin(gt_np, [1, 3]), np.isin(pred_np, [1, 3])),
        "ET (3)": (gt_np == 3, pred_np == 3),
    }
    lines.append("")
    for name, (g, p_) in regions.items():
        lines.append(f"{name:<34}{int(g.sum()):>12}{int(p_.sum()):>13}{dice(g, p_):>8.4f}")

    # largest tumor slice - a cursor placement hint
    tumor = np.isin(gt_np, [1, 2, 3])
    et = gt_np == 3
    lines.append("")
    for axis, axname in enumerate(["i (sagittal)", "j (coronal)", "k (axial)"]):
        ax = tuple(a for a in range(3) if a != axis)
        lines.append(
            f"Largest cross-section along {axname}: WT={int(tumor.sum(axis=ax).argmax())}"
            f", ET={int(et.sum(axis=ax).argmax())}"
        )

    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt, flush=True)
    print(f"-> saved {out_dir}\n", flush=True)


def main():
    pids = sys.argv[1:]
    if not pids:
        print("Provide at least one patient ID", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {CHECKPOINT_PATH}", flush=True)
    model = build_model(device)
    for pid in pids:
        predict_patient(model, device, pid)


if __name__ == "__main__":
    main()
