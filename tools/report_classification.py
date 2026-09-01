"""Collects outputs/classification/cls_*.npz into a CSV and a LaTeX table
for the thesis.

Usage:
    python tools/report_classification.py
"""
import os
import sys
from pathlib import Path

import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.cls_metrics import summarize, to_csv_rows, CSV_HEADER   # noqa: E402

OUT_DIR = os.environ.get("CLS_DIR", f"{project_root}/outputs/classification")
MODEL_LABEL = {"multitask": "Multi-task (standard)", "detach": "Multi-task (Detach)"}
# Polish labels: pasted directly into the thesis LaTeX
LEVELS = [("patch", "łatkowy"), ("vol", "wolumetryczny")]


def latex_escape(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def fmt(v, nd=4):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}".replace(".", "{,}")


def main():
    runs = {}
    for tag in MODEL_LABEL:
        p = os.path.join(OUT_DIR, f"cls_{tag}.npz")
        if os.path.exists(p):
            runs[tag] = np.load(p, allow_pickle=True)
        else:
            print(f"  skipping '{tag}' — missing {p}")
    if not runs:
        raise SystemExit(f"No cls_*.npz files found in {OUT_DIR}")

    comps = [str(c) for c in runs[list(runs)[0]]["components"]]
    csv_rows, results = [], {}

    for tag, z in runs.items():
        for key, level_pl in LEVELS:
            rows = summarize(z[f"{key}_logits"], z[f"{key}_targets"], comps)
            results[(tag, key)] = rows
            csv_rows += to_csv_rows(rows, tag, level_pl)

    csv_path = os.path.join(OUT_DIR, "classification_summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for r in csv_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Saved {csv_path}")

    # LaTeX table, in Polish - goes directly into the thesis
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ewaluacja głowy klasyfikacyjnej: wykrywanie obecności podstruktur",
        r"guza. Wartości AP należy interpretować wyłącznie w odniesieniu do częstości",
        r"bazowej podanej w sąsiedniej kolumnie.}",
        r"\label{tab:klasyfikacja}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}llccccccc@{}}",
        r"\hline",
        r"\textbf{Model} & \textbf{Komponent} & \shortstack{\textbf{Częstość}\\\textbf{bazowa}} &",
        r"\textbf{AP} & \shortstack{\textbf{AP/czę-}\\\textbf{stość}} & \textbf{ROC-AUC} &",
        r"\textbf{MCC} & \shortstack{\textbf{Zrówn.}\\\textbf{trafność}} & \textbf{Czułość} \\",
        r"\hline",
    ]
    for key, level_pl in LEVELS:
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{Poziom {level_pl}}}}} \\")
        for tag in MODEL_LABEL:
            if (tag, key) not in results:
                continue
            for i, r in enumerate(results[(tag, key)]):
                model_cell = latex_escape(MODEL_LABEL[tag]) if i == 0 else ""
                lines.append(
                    f"{model_cell} & {r['component']} & {fmt(r['base_rate'])} & "
                    f"{fmt(r.get('ap'))} & {fmt(r.get('lift'), 3)} & {fmt(r.get('roc_auc'))} & "
                    f"{fmt(r.get('mcc'))} & {fmt(r.get('bacc'))} & {fmt(r.get('sens'))} \\\\")
            lines.append(r"\cline{2-9}")
        lines[-1] = r"\hline"
    lines += [r"\end{tabular}", r"\end{table}"]

    tex_path = os.path.join(OUT_DIR, "tabela_klasyfikacja.tex")
    open(tex_path, "w").write("\n".join(lines) + "\n")
    print(f"Saved {tex_path}")

    from src.utils.cls_metrics import print_summary
    for key, level_pl in LEVELS:
        for tag in MODEL_LABEL:
            if (tag, key) in results:
                print_summary(results[(tag, key)], f"LEVEL: {level_pl.upper()}",
                              MODEL_LABEL[tag])


if __name__ == "__main__":
    main()
