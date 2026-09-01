"""Metrics for a multi-label classification head under strong class imbalance.
Accuracy is omitted (SNFH's base rate is 324/325); AP is always paired with
the base rate, alongside MCC, balanced accuracy, and ROC-AUC."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             precision_recall_curve)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def summarize(logits, targets, components):
    """Returns a list of per-component metric dicts."""
    probs = sigmoid(np.asarray(logits, dtype=np.float64))
    y = np.asarray(targets, dtype=np.int32)
    rows = []
    for i, name in enumerate(components):
        p, t = probs[:, i], y[:, i]
        n_pos, n = int(t.sum()), len(t)
        base = n_pos / n if n else np.nan
        row = dict(component=name, n=n, n_pos=n_pos, base_rate=base)

        if n_pos == 0 or n_pos == n:
            row.update(ap=np.nan, lift=np.nan, roc_auc=np.nan, mcc=np.nan,
                       bacc=np.nan, sens=np.nan, spec=np.nan,
                       best_f1=np.nan, best_thr=np.nan, degenerate=True)
            rows.append(row)
            continue

        ap = average_precision_score(t, p)
        pred = (p >= 0.5).astype(np.int32)
        tp = int(((pred == 1) & (t == 1)).sum())
        fp = int(((pred == 1) & (t == 0)).sum())
        fn = int(((pred == 0) & (t == 1)).sum())
        tn = int(((pred == 0) & (t == 0)).sum())

        prec_c, rec_c, thr_c = precision_recall_curve(t, p)
        f1_c = np.divide(2 * prec_c * rec_c, prec_c + rec_c,
                         out=np.zeros_like(prec_c), where=(prec_c + rec_c) > 0)
        k = int(np.argmax(f1_c))

        row.update(
            ap=ap,
            lift=ap / base if base > 0 else np.nan,
            roc_auc=roc_auc_score(t, p),
            mcc=matthews_corrcoef(t, pred),
            bacc=balanced_accuracy_score(t, pred),
            sens=tp / (tp + fn) if (tp + fn) else np.nan,
            spec=tn / (tn + fp) if (tn + fp) else np.nan,
            best_f1=float(f1_c[k]),
            best_thr=float(thr_c[k]) if k < len(thr_c) else 1.0,
            tp=tp, fp=fp, fn=fn, tn=tn, degenerate=False,
        )
        rows.append(row)
    return rows


def print_summary(rows, title, model_label):
    print("\n" + "=" * 100)
    print(f"{title} — {model_label}")
    print("=" * 100)
    print(f"{'component':<10} {'n':>6} {'base rate':>11} {'AP':>8} {'AP/rate':>10} "
          f"{'ROC-AUC':>9} {'MCC':>7} {'bal. acc.':>12} {'sens.':>8} {'spec.':>8}")
    print("-" * 100)
    for r in rows:
        if r["degenerate"]:
            print(f"{r['component']:<10} {r['n']:>6} {r['base_rate']:>11.4f} "
                  f"{'—':>8} {'—':>10} {'—':>9} {'—':>7} {'—':>12} {'—':>8} {'—':>8}"
                  "   (single class — metrics undefined)")
            continue
        print(f"{r['component']:<10} {r['n']:>6} {r['base_rate']:>11.4f} {r['ap']:>8.4f} "
              f"{r['lift']:>10.3f} {r['roc_auc']:>9.4f} {r['mcc']:>7.4f} "
              f"{r['bacc']:>12.4f} {r['sens']:>8.4f} {r['spec']:>8.4f}")
    print("\nAP is only meaningful relative to the base rate — an AP/rate ratio")
    print("of 1.0 means the model is indistinguishable from a constant answer.")


def to_csv_rows(rows, model, level):
    out = []
    for r in rows:
        out.append((model, level, r["component"], r["n"], r["n_pos"],
                    f"{r['base_rate']:.6f}",
                    "" if r["degenerate"] else f"{r['ap']:.6f}",
                    "" if r["degenerate"] else f"{r['lift']:.6f}",
                    "" if r["degenerate"] else f"{r['roc_auc']:.6f}",
                    "" if r["degenerate"] else f"{r['mcc']:.6f}",
                    "" if r["degenerate"] else f"{r['bacc']:.6f}",
                    "" if r["degenerate"] else f"{r['sens']:.6f}",
                    "" if r["degenerate"] else f"{r['spec']:.6f}",
                    "" if r["degenerate"] else f"{r['best_f1']:.6f}",
                    "" if r["degenerate"] else f"{r['best_thr']:.6f}"))
    return out


CSV_HEADER = ["model", "level", "component", "n", "n_pos", "base_rate",
              "AP", "AP_over_base_rate", "ROC_AUC", "MCC", "balanced_accuracy",
              "sensitivity", "specificity", "best_F1", "best_F1_threshold"]
