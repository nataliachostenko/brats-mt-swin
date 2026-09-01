"""Quick PNG montage (FLAIR|T1ce x GT|Prediction) for picking a case from
slicer_cases/; final figures are made in 3D Slicer."""

import os
import sys
import glob

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw

# 1 = NETC (red), 2 = SNFH/edema (green), 3 = ET (yellow)
COLORS = {1: (217, 26, 26), 2: (26, 204, 26), 3: (255, 230, 26)}
ALPHA = 0.45
PANEL = 260  # length of a single panel's longer side, in px


def norm8(x):
    v = x[x > 0]
    lo, hi = (np.percentile(v, [1, 99]) if v.size else (0.0, 1.0))
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def slice_at(vol, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    return np.rot90(vol[tuple(sl)])


def blend(bg2d, mask2d):
    rgb = np.stack([bg2d] * 3, axis=-1).astype(float)
    for lab, col in COLORS.items():
        m = mask2d == lab
        if m.any():
            rgb[m] = rgb[m] * (1 - ALPHA) + np.array(col, dtype=float) * ALPHA
    img = Image.fromarray(rgb.astype(np.uint8))
    w, h = img.size
    s = PANEL / max(w, h)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)


def load_volume(case_dir, pid, suffix):
    return np.asarray(nib.load(os.path.join(case_dir, f"{pid}{suffix}")).dataobj)


def make_preview(case_dir, out_png):
    pid = os.path.basename(case_dir.rstrip("/"))
    t2f = norm8(load_volume(case_dir, pid, "-t2f.nii.gz").astype(float))
    t1c = norm8(load_volume(case_dir, pid, "-t1c.nii.gz").astype(float))
    gt = load_volume(case_dir, pid, "-seg-GT.nii.gz").astype(int)
    pr = load_volume(case_dir, pid, "-seg-PRED.nii.gz").astype(int)

    tumor = np.isin(gt, [1, 2, 3])
    idx = [int(tumor.sum(axis=tuple(a for a in range(3) if a != ax)).argmax()) for ax in range(3)]
    col_axes = [2, 1, 0]  # axial, coronal, sagittal
    col_names = ["axial", "coronal", "sagittal"]

    rows = [("FLAIR (t2f) + Ground Truth", t2f, gt), ("FLAIR (t2f) + Prediction", t2f, pr),
            ("T1ce (t1c) + Ground Truth", t1c, gt), ("T1ce (t1c) + Prediction", t1c, pr)]

    pad, left, top = 6, 190, 46
    canvas = Image.new("RGB", (left + 3 * (PANEL + pad) + pad, top + 4 * (PANEL + pad) + pad), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{pid}   (red=NETC, green=edema/SNFH, yellow=ET)", fill=(255, 255, 255))
    for c, (ax_i, nm) in enumerate(zip(col_axes, col_names)):
        draw.text((left + c * (PANEL + pad) + 4, top - 14), f"{nm}  (idx {idx[ax_i]})", fill=(200, 200, 200))

    for r, (title, bg, mask) in enumerate(rows):
        y = top + r * (PANEL + pad)
        draw.text((8, y + PANEL // 2), title, fill=(230, 230, 230))
        for c, ax_i in enumerate(col_axes):
            panel = blend(slice_at(bg, ax_i, idx[ax_i]), slice_at(mask, ax_i, idx[ax_i]))
            canvas.paste(panel, (left + c * (PANEL + pad), y))

    canvas.save(out_png)
    print("saved", out_png)


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(d for d in glob.glob("slicer_cases/*") if os.path.isdir(d))
    for d in dirs:
        make_preview(d, os.path.join("slicer_cases", os.path.basename(d.rstrip("/")) + "_preview.png"))
