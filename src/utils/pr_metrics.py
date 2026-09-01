"""
Precision-recall curves computed from score histograms, since 325 patients x
7.2M voxels doesn't fit in memory as raw arrays.
"""
import numpy as np

NBINS = 65536
RECALL_GRID = np.linspace(0.0, 1.0, 101)


def score_histograms(scores, positives, nbins=NBINS):
    idx = np.minimum((scores * nbins).astype(np.int64), nbins - 1)
    return (np.bincount(idx[positives], minlength=nbins),
            np.bincount(idx[~positives], minlength=nbins))


def pr_from_hist(hist_pos, hist_neg):
    """PR curve from histograms. Returns (precision, recall, AP); (None, None, nan) if no positives."""
    tp = np.cumsum(hist_pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(hist_neg[::-1])[::-1].astype(np.float64)
    n_pos = tp[0]
    if n_pos == 0:
        return None, None, np.nan
    denom = tp + fp
    prec = np.divide(tp, denom, out=np.ones_like(tp), where=denom > 0)
    rec = tp / n_pos
    rec_ext = np.append(rec, 0.0)
    ap = float(np.sum((rec_ext[:-1] - rec_ext[1:]) * prec))
    return prec, rec, ap


def prec_at_recall_grid(prec, rec, grid=RECALL_GRID):
    """Interpolated P(r) = max{P(r') : r' >= r} on a common recall grid, for monotonicity."""
    if prec is None:
        return np.full(len(grid), np.nan)
    rec_inc = rec[::-1]
    prec_inc = prec[::-1]
    suffix_max = np.maximum.accumulate(prec_inc[::-1])[::-1]
    idx = np.searchsorted(rec_inc, grid, side="left")
    out = np.zeros(len(grid))
    valid = idx < len(rec_inc)
    out[valid] = suffix_max[idx[valid]]
    return out


def simplify_curve(rec, prec, tol=0.0015):
    """Reduces a curve to a plottable polyline, keeping breakpoints adaptively."""
    keep = [0]
    lr, lp = rec[0], prec[0]
    for i in range(1, len(rec) - 1):
        if abs(rec[i] - lr) > tol or abs(prec[i] - lp) > tol:
            keep.append(i)
            lr, lp = rec[i], prec[i]
    keep.append(len(rec) - 1)
    return rec[keep], prec[keep]
