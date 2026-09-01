"""
Precision-recall data at two evaluation levels: voxel-wise (histogram-based AP,
memory-cheap) and lesion-wise (connected components, following the BraTS 2024
official metric: 26-connectivity + dilation merge + a volume filter). Lesion-wise
is computed both as a threshold sweep and as a confidence-ranked detection AP.

Usage:
    MODEL_TAG=multitask python tools/eval_pr_curves.py
"""
import os
import sys
import time
from pathlib import Path
import typing

import numpy as np
import torch
import cc3d
import scipy.ndimage
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
from monai.inferers import sliding_window_inference   # noqa: E402
from monai.data import decollate_batch          # noqa: E402

from src.utils.pr_metrics import (                                 # noqa: E402
    NBINS, RECALL_GRID, score_histograms, pr_from_hist, prec_at_recall_grid)
from src.models.brats_module import BraTSLightningModule           # noqa: E402
from src.models.components.base_swin import BaseSwinUNETR          # noqa: E402
from src.models.components.multi_task_swin import MultiTaskSwinUNETR              # noqa: E402
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
    "base": dict(
        ckpt=f"{CKPT_ROOT}/lvjmfekz/checkpoints/epoch=99-step=16200.ckpt",
        build=lambda: BaseSwinUNETR(in_channels=4, out_channels=4, feature_size=48),
        label="Swin-UNETR Base",
    ),
}

REGIONS = ["WT", "TC", "ET"]

# denser near the extremes, where large lesions are detected at almost any threshold
THRESHOLDS = np.array([
    0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999,
])
BASE_TH = 0.05
BASE_TH_IDX = int(np.argmin(np.abs(THRESHOLDS - BASE_TH)))

DIL_FACTOR = 3                                  # as in metrics_GLI.py
VOL_THRESH = {"WT": 20, "TC": 20, "ET": 10}     # official prediction volume filter
# stricter filter, saved separately as 'lesion_pp'; NOT the same postprocessing as
# src.utils.postprocess.remove_small_islands (that one runs pre-region, 6-connectivity)
PP_MIN_VOXELS = 100
STRUCT = scipy.ndimage.generate_binary_structure(3, 2)


def _bbox(mask, margin):
    nz = np.argwhere(mask)
    if len(nz) == 0:
        return None
    lo = np.maximum(nz.min(0) - margin, 0)
    hi = np.minimum(nz.max(0) + 1 + margin, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def merge_by_dilation(mask, dil=DIL_FACTOR):
    """Components touching after dilation get merged into one label (bbox-scoped)."""
    out = np.zeros(mask.shape, dtype=np.int32)
    sl = _bbox(mask, dil + 1)
    if sl is None:
        return out, 0
    sub = mask[sl]
    dil_cc = cc3d.connected_components(
        scipy.ndimage.binary_dilation(sub, structure=STRUCT, iterations=dil),
        connectivity=26,
    )
    out[sl] = np.where(sub, dil_cc, 0)
    return out, int(out.max())


def lesion_sizes(labels, n_labels):
    if n_labels == 0:
        return np.zeros(1, dtype=np.int64)
    return np.bincount(labels.ravel(), minlength=n_labels + 1)


def gt_lesion_rois(gt_mask, vol_thresh):
    """
    Per-GT-lesion (slice, dilated_submask, counts_toward_tp_fn). GT lesions below
    vol_thresh are excluded from TP/FN counts but kept in the list, since the
    official matching loop still checks every GT lesion when deciding FPs.
    """
    labels, n = merge_by_dilation(gt_mask)
    sizes = lesion_sizes(labels, n)
    rois = []
    for lab in range(1, n + 1):
        lesion = labels == lab
        sl = _bbox(lesion, DIL_FACTOR + 1)
        if sl is None:
            continue
        dilated = scipy.ndimage.binary_dilation(
            lesion[sl], structure=STRUCT, iterations=DIL_FACTOR)
        rois.append((sl, dilated, bool(sizes[lab] > vol_thresh)))
    return rois


def match_lesions(pred_labels, keep_mask, gt_rois):
    """A GT lesion is TP if a kept prediction component intersects it after dilation."""
    matched = set()
    tp = fn = 0
    for sl, dilated, counts in gt_rois:
        sub = pred_labels[sl]
        hit = np.unique(sub[dilated])
        hit = [int(h) for h in hit if h != 0 and keep_mask[h]]
        if hit:
            matched.update(hit)
            if counts:
                tp += 1
        elif counts:
            fn += 1
    fp = int(keep_mask.sum()) - len(matched)
    return tp, fp, fn


def detections_with_confidence(pred_labels, n_pred, keep_mask, prob):
    """Mean and max probability inside each kept component."""
    if n_pred == 0:
        return []
    flat = pred_labels.ravel()
    sums = np.bincount(flat, weights=prob.ravel(), minlength=n_pred + 1)
    counts = np.bincount(flat, minlength=n_pred + 1)
    idxs = np.arange(1, n_pred + 1)
    maxs = np.atleast_1d(scipy.ndimage.maximum(prob, labels=pred_labels, index=idxs))
    out = []
    for lab in idxs:
        if not keep_mask[lab] or counts[lab] == 0:
            continue
        out.append((int(lab), sums[lab] / counts[lab], float(maxs[lab - 1])))
    return out


def greedy_match_detections(dets, pred_labels, gt_rois):
    """Greedy one-to-one matching by descending confidence, object-detection style."""
    hits, ignored = {}, {}
    for gi, (sl, dilated, counts) in enumerate(gt_rois):
        sub = pred_labels[sl]
        for h in np.unique(sub[dilated]):
            if h == 0:
                continue
            (hits if counts else ignored).setdefault(int(h), []).append(gi)
    taken = set()
    results = []
    for lab, conf_mean, conf_max in sorted(dets, key=lambda d: -d[1]):
        is_tp = False
        for g in hits.get(lab, []):
            if g not in taken:
                taken.add(g)
                is_tp = True
                break
        if not is_tp and lab in ignored and lab not in hits:
            continue  # hits only an excluded micro-lesion: neither TP nor FP
        results.append((conf_mean, conf_max, is_tp))
    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    tag = os.environ.get("MODEL_TAG", "multitask")
    if tag not in MODELS:
        raise SystemExit(f"MODEL_TAG must be one of {list(MODELS)}, got '{tag}'")
    spec = MODELS[tag]
    ckpt = os.environ.get("CHECKPOINT_PATH", spec["ckpt"])
    limit = int(os.environ.get("EVAL_LIMIT", 0))
    out_dir = os.environ.get("OUT_DIR", f"{project_root}/outputs/pr_curves")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pr_curves_{tag}.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model     : {tag} ({spec['label']})")
    print(f"Checkpoint: {ckpt}")
    print(f"Device    : {device}")
    print(f"Output    : {out_path}")

    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    net = spec["build"]()
    model = BraTSLightningModule.load_from_checkpoint(ckpt, net=net, map_location=device)
    model.to(device)
    model.eval()

    n_th = len(THRESHOLDS)
    hist_pos = np.zeros((3, NBINS), dtype=np.int64)
    hist_neg = np.zeros((3, NBINS), dtype=np.int64)

    per_patient_ap = []          # [n_pat, 3]
    per_patient_prec = []        # [n_pat, 3, 101]
    lw_raw, lw_pp = [], []       # [n_pat, 3, n_th, 3] (tp, fp, fn); official filter vs stricter PP
    lw_argmax = []               # [n_pat, 3, 3] lesion-wise TP/FP/FN at the argmax decision
    vx_argmax = []                # [n_pat, 3, 3] voxel-wise TP/FP/FN at the argmax decision
    det_records = []             # (region_idx, conf_mean, conf_max, is_tp)
    det_n_gt = np.zeros(3, dtype=np.int64)
    patient_ids = []

    def flush():
        np.savez_compressed(
            out_path,
            tag=tag, label=spec["label"], checkpoint=ckpt,
            regions=np.array(REGIONS), thresholds=THRESHOLDS,
            recall_grid=RECALL_GRID, nbins=NBINS,
            base_threshold=THRESHOLDS[BASE_TH_IDX],
            dil_factor=DIL_FACTOR,
            vol_thresh=np.array([VOL_THRESH[r] for r in REGIONS]),
            pp_min_voxels=PP_MIN_VOXELS,
            hist_pos=hist_pos, hist_neg=hist_neg,
            per_patient_ap=np.array(per_patient_ap),
            per_patient_prec=np.array(per_patient_prec),
            lesion_raw=np.array(lw_raw), lesion_pp=np.array(lw_pp),
            lesion_argmax=np.array(lw_argmax),
            voxel_argmax=np.array(vx_argmax),
            det_records=np.array(det_records) if det_records else np.zeros((0, 4)),
            det_n_gt=det_n_gt,
            patient_ids=np.array(patient_ids),
        )

    def predictor(x):
        out = model.net(x)
        return out[0] if isinstance(out, tuple) else out

    t_start = time.time()
    count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if limit > 0 and count >= limit:
                break
            images = batch["image"].to(device)
            labels_t = batch["seg"].to(device)

            try:
                gt_path = batch["seg"].meta["filename_or_obj"][0]
            except Exception:
                gt_path = batch["seg_meta_dict"]["filename_or_obj"][0]
            patient_ids.append(Path(gt_path).parent.name)
            count += 1

            logits = sliding_window_inference(
                inputs=images, roi_size=(96, 96, 96), sw_batch_size=4,
                predictor=predictor, overlap=0.5,
            )
            prob_t = torch.softmax(decollate_batch(logits)[0], dim=0)
            class_map = torch.argmax(prob_t, dim=0).cpu().numpy()
            prob = prob_t.cpu().numpy().astype(np.float32)
            del logits, prob_t

            gt = decollate_batch(labels_t)[0].squeeze().cpu().numpy().astype(np.int8)

            region_prob = {
                "WT": prob[1] + prob[2] + prob[3],
                "TC": prob[1] + prob[3],
                "ET": prob[3],
            }
            region_gt = {
                "WT": gt > 0,
                "TC": (gt == 1) | (gt == 3),
                "ET": gt == 3,
            }
            region_argmax = {
                "WT": class_map > 0,
                "TC": (class_map == 1) | (class_map == 3),
                "ET": class_map == 3,
            }

            ap_row, prec_row = [], []
            raw_row = np.zeros((3, n_th, 3), dtype=np.int64)
            pp_row = np.zeros((3, n_th, 3), dtype=np.int64)
            argmax_row = np.zeros((3, 3), dtype=np.int64)
            vx_row = np.zeros((3, 3), dtype=np.int64)

            for ri, reg in enumerate(REGIONS):
                p = np.clip(region_prob[reg], 0.0, 1.0)
                y = region_gt[reg]
                pflat, yflat = p.ravel(), y.ravel()

                hp, hn = score_histograms(pflat, yflat)
                hist_pos[ri] += hp
                hist_neg[ri] += hn
                prec_c, rec_c, ap = pr_from_hist(hp, hn)
                ap_row.append(ap)
                prec_row.append(prec_at_recall_grid(prec_c, rec_c))

                vt = VOL_THRESH[reg]
                rois = gt_lesion_rois(y, vt)
                det_n_gt[ri] += sum(1 for *_, counts in rois if counts)

                for ti, th in enumerate(THRESHOLDS):
                    pl, npred = merge_by_dilation(p >= th)
                    sizes = lesion_sizes(pl, npred)
                    keep_raw = sizes > vt
                    keep_pp = sizes >= PP_MIN_VOXELS
                    if len(keep_raw):
                        keep_raw[0] = False
                        keep_pp[0] = False
                    raw_row[ri, ti] = match_lesions(pl, keep_raw, rois)
                    pp_row[ri, ti] = match_lesions(pl, keep_pp, rois)

                    if ti == BASE_TH_IDX:
                        dets = detections_with_confidence(pl, npred, keep_raw, p)
                        for cm, cx, tp_flag in greedy_match_detections(dets, pl, rois):
                            det_records.append((ri, cm, cx, float(tp_flag)))

                am_mask = region_argmax[reg]
                vx_row[ri] = (np.count_nonzero(am_mask & y),
                              np.count_nonzero(am_mask & ~y),
                              np.count_nonzero(~am_mask & y))

                pl, npred = merge_by_dilation(am_mask)
                sizes = lesion_sizes(pl, npred)
                keep = sizes > vt
                if len(keep):
                    keep[0] = False
                argmax_row[ri] = match_lesions(pl, keep, rois)

            per_patient_ap.append(ap_row)
            per_patient_prec.append(prec_row)
            lw_raw.append(raw_row)
            lw_pp.append(pp_row)
            lw_argmax.append(argmax_row)
            vx_argmax.append(vx_row)

            if count % 25 == 0:
                flush()

    flush()
    dt = time.time() - t_start
    print(f"\nProcessed {count} patients in {dt/60:.1f} min ({dt/max(count,1):.1f} s/patient)")

    print("\n" + "=" * 62)
    print(f"PR-AUC — {spec['label']}")
    print("=" * 62)
    ap_arr = np.array(per_patient_ap)
    for ri, reg in enumerate(REGIONS):
        _, _, ap_micro = pr_from_hist(hist_pos[ri], hist_neg[ri])
        col = ap_arr[:, ri]
        ap_macro = np.nanmean(col)
        n_valid = int(np.sum(~np.isnan(col)))

        raw = np.array(lw_raw)[:, ri].sum(axis=0)       # [n_th, 3]
        tp, fp, fn = raw[:, 0], raw[:, 1], raw[:, 2]
        prec = np.divide(tp, tp + fp, out=np.ones_like(tp, float), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp, float), where=(tp + fn) > 0)

        # the sweep only covers the recall range reachable by changing the threshold;
        # this is NOT the full PR-AUC
        span = float(rec.max() - rec.min())
        order = np.argsort(rec)
        lesion_auc = float(np.trapz(prec[order], rec[order])) if span > 1e-9 else np.nan

        am = np.array(lw_argmax)[:, ri].sum(axis=0)
        am_p = am[0] / max(am[0] + am[1], 1)
        am_r = am[0] / max(am[0] + am[2], 1)

        vx = np.array(vx_argmax)[:, ri].astype(np.float64)   # [n_pat, 3]
        vtp, vfp, vfn = vx[:, 0], vx[:, 1], vx[:, 2]
        den = 2 * vtp + vfp + vfn
        with np.errstate(invalid="ignore", divide="ignore"):
            dice_pp = np.where(den > 0, 2 * vtp / den, np.nan)
        dice_macro = float(np.nanmean(dice_pp))
        n_dice = int(np.sum(~np.isnan(dice_pp)))
        dice_micro = float(2 * vtp.sum() / max(2 * vtp.sum() + vfp.sum() + vfn.sum(), 1))
        p_micro = float(vtp.sum() / max(vtp.sum() + vfp.sum(), 1))
        r_micro = float(vtp.sum() / max(vtp.sum() + vfn.sum(), 1))

        print(f"{reg}:")
        print(f"  voxel-wise PR-AUC   micro={ap_micro:.4f}   "
              f"macro={ap_macro:.4f} (n={n_valid})")
        print(f"  voxel-wise @argmax: Dice macro={dice_macro:.4f} (n={n_dice})  "
              f"Dice micro={dice_micro:.4f}  P={p_micro:.4f}  R={r_micro:.4f}")
        print(f"  lesion-wise sweep: recall {rec.min():.4f}-{rec.max():.4f} "
              f"(span {span:.4f}), precision {prec.min():.4f}-{prec.max():.4f}")
        print(f"  lesion-wise AUC over that range   = {lesion_auc:.4f}  "
              f"(NOT full PR-AUC — see detection AP below)")
        print(f"  argmax working point: P={am_p:.4f}  R={am_r:.4f}  "
              f"(TP={am[0]} FP={am[1]} FN={am[2]})")

    if det_records:
        d = np.array(det_records)
        for ri, reg in enumerate(REGIONS):
            sub = d[d[:, 0] == ri]
            if len(sub) == 0 or det_n_gt[ri] == 0:
                continue
            o = np.argsort(-sub[:, 1])
            tp_c = np.cumsum(sub[o, 3])
            fp_c = np.cumsum(1 - sub[o, 3])
            prec = tp_c / np.maximum(tp_c + fp_c, 1)
            rec = tp_c / det_n_gt[ri]
            rec_ext = np.append(0.0, rec)
            ap = float(np.sum((rec_ext[1:] - rec_ext[:-1]) * prec))
            print(f"{reg}: lesion-wise detection AP (ranking) = {ap:.4f} "
                  f"({len(sub)} detections / {det_n_gt[ri]} GT lesions)")
    print("=" * 62)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
