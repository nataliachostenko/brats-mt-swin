"""
Evaluates the classification head (cls_logits) of the multi-task models:
predicted presence of NETC/SNFH/ET, at patch level (training distribution)
and volume level (sliding-window max, the intended screening use case).

Usage:
    MODEL_TAG=multitask python tools/eval_classification.py
"""
import os
import sys
import time
from pathlib import Path
import typing

import numpy as np
import torch
import omegaconf
import omegaconf.base
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

import hydra                                    # noqa: E402
from omegaconf import DictConfig                # noqa: E402
from hydra.utils import instantiate             # noqa: E402

from src.models.brats_module import BraTSLightningModule                       # noqa: E402
from src.models.components.multi_task_swin import MultiTaskSwinUNETR           # noqa: E402
from src.models.components.multi_task_swin_detach import MultiTaskSwinUNETRDetach  # noqa: E402

CKPT_ROOT = f"{project_root}/logs/brats-mgr-project"

MODELS = {
    "multitask": dict(
        ckpt=f"{CKPT_ROOT}/unwx02xp/checkpoints/resume.ckpt",
        build=lambda: MultiTaskSwinUNETR(in_channels=4, num_classes=4, feature_size=48),
        label="Multi-task (standard)",
    ),
    "detach": dict(
        ckpt=f"{CKPT_ROOT}/jnopa6kt/checkpoints/epoch=99-step=16200.ckpt",
        build=lambda: MultiTaskSwinUNETRDetach(in_channels=4, num_classes=4, feature_size=48),
        label="Multi-task (Detach)",
    ),
}

COMPONENTS = ["NETC", "SNFH", "ET"]     # classes 1, 2, 3 after MapLabelValued
CLASS_IDS = [1, 2, 3]
ROI = (96, 96, 96)
PATCHES_PER_PATIENT = int(os.environ.get("PATCHES_PER_PATIENT", 8))
SW_OVERLAP = 0.5
SW_BATCH = 4


def window_starts(size, roi, stride):
    """Sliding-window start offsets along one axis, closed off at the end."""
    if size <= roi:
        return [0]
    starts = list(range(0, size - roi + 1, stride))
    if starts[-1] != size - roi:
        starts.append(size - roi)
    return starts


def enumerate_windows(shape, roi=ROI, overlap=SW_OVERLAP):
    stride = [max(int(r * (1 - overlap)), 1) for r in roi]
    axes = [window_starts(shape[i], roi[i], stride[i]) for i in range(3)]
    return [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]


@torch.no_grad()
def cls_logits_for_batch(model, batch):
    out = model.net(batch)
    if not isinstance(out, tuple):
        raise RuntimeError("Model does not return cls_logits — not a multi-task variant")
    return out[1]


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    tag = os.environ.get("MODEL_TAG", "multitask")
    if tag not in MODELS:
        raise SystemExit(f"MODEL_TAG must be one of {list(MODELS)} (the base model "
                         f"has no classification head), got '{tag}'")
    spec = MODELS[tag]
    ckpt = os.environ.get("CHECKPOINT_PATH", spec["ckpt"])
    limit = int(os.environ.get("EVAL_LIMIT", 0))
    out_dir = os.environ.get("OUT_DIR", f"{project_root}/outputs/classification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cls_{tag}.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model     : {tag} ({spec['label']})")
    print(f"Checkpoint: {ckpt}")
    print(f"Device    : {device}")
    print(f"Patches per patient: {PATCHES_PER_PATIENT}")

    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    net = spec["build"]()
    model = BraTSLightningModule.load_from_checkpoint(ckpt, net=net, map_location=device)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(42)     # reproducible patch sampling

    patch_logits, patch_targets = [], []
    vol_logits, vol_targets, patient_ids = [], [], []

    t0 = time.time()
    count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if limit > 0 and count >= limit:
                break
            image = batch["image"][0].to(device)          # (4, H, W, D)
            seg = batch["seg"][0, 0].cpu().numpy().astype(np.int8)   # (H, W, D)
            try:
                gt_path = batch["seg"].meta["filename_or_obj"][0]
            except Exception:
                gt_path = batch["seg_meta_dict"]["filename_or_obj"][0]
            patient_ids.append(Path(gt_path).parent.name)
            count += 1

            shape = seg.shape
            masks = {c: (seg == c) for c in CLASS_IDS}

            vol_targets.append([float(masks[c].any()) for c in CLASS_IDS])
            wins = enumerate_windows(shape)
            per_window = []
            for i in range(0, len(wins), SW_BATCH):
                chunk = wins[i:i + SW_BATCH]
                stack = torch.stack([
                    image[:, x:x + ROI[0], y:y + ROI[1], z:z + ROI[2]] for x, y, z in chunk
                ])
                per_window.append(cls_logits_for_batch(model, stack).float().cpu())
            per_window = torch.cat(per_window, dim=0)          # (n_windows, 3)
            vol_logits.append(per_window.max(dim=0).values.numpy())

            fg = np.argwhere(seg > 0)
            brain = np.argwhere(image[0].cpu().numpy() != 0)
            centers = []
            n_pos = PATCHES_PER_PATIENT // 2
            for k in range(PATCHES_PER_PATIENT):
                pool = fg if (k < n_pos and len(fg) > 0) else brain
                if len(pool) == 0:
                    pool = np.array([[s // 2 for s in shape]])
                centers.append(pool[rng.integers(len(pool))])

            stack, tgts = [], []
            for cen in centers:
                start = [int(np.clip(cen[i] - ROI[i] // 2, 0, shape[i] - ROI[i]))
                         for i in range(3)]
                sl = tuple(slice(start[i], start[i] + ROI[i]) for i in range(3))
                stack.append(image[:, sl[0], sl[1], sl[2]])
                tgts.append([float(masks[c][sl].any()) for c in CLASS_IDS])
            stack = torch.stack(stack)
            for i in range(0, len(stack), SW_BATCH):
                lg = cls_logits_for_batch(model, stack[i:i + SW_BATCH]).float().cpu().numpy()
                patch_logits.append(lg)
            patch_targets.append(np.array(tgts, dtype=np.float32))

            if count % 25 == 0:
                np.savez_compressed(
                    out_path, tag=tag, label=spec["label"], checkpoint=ckpt,
                    components=np.array(COMPONENTS),
                    patch_logits=np.concatenate(patch_logits),
                    patch_targets=np.concatenate(patch_targets),
                    vol_logits=np.array(vol_logits), vol_targets=np.array(vol_targets),
                    patient_ids=np.array(patient_ids),
                    patches_per_patient=PATCHES_PER_PATIENT)

    patch_logits = np.concatenate(patch_logits)
    patch_targets = np.concatenate(patch_targets)
    vol_logits = np.array(vol_logits)
    vol_targets = np.array(vol_targets)

    np.savez_compressed(
        out_path, tag=tag, label=spec["label"], checkpoint=ckpt,
        components=np.array(COMPONENTS),
        patch_logits=patch_logits, patch_targets=patch_targets,
        vol_logits=vol_logits, vol_targets=vol_targets,
        patient_ids=np.array(patient_ids), patches_per_patient=PATCHES_PER_PATIENT)

    dt = time.time() - t0
    print(f"\nProcessed {count} patients in {dt/60:.1f} min "
          f"({dt/max(count,1):.1f} s/patient)")
    print(f"Saved: {out_path}")

    from src.utils.cls_metrics import summarize, print_summary   # noqa: E402
    print_summary(summarize(patch_logits, patch_targets, COMPONENTS), "PATCH LEVEL",
                  spec["label"])
    print_summary(summarize(vol_logits, vol_targets, COMPONENTS), "VOLUME LEVEL",
                  spec["label"])


if __name__ == "__main__":
    main()
