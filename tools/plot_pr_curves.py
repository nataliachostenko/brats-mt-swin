"""
Rysuje krzywe PR z plikow outputs/pr_curves/pr_curves_<tag>.npz.

Wynik (outputs/pr_curves/):
    pr_voxelwise.svg            - krzywe voxel-wise, 3 panele (WT/TC/ET)
    pr_lesionwise.svg           - krzywe lesion-wise (detection AP, ranking po pewnosci), osie [0,1]
    pr_lesionwise_zoomed.svg    - to samo co wyzej, ale kadrowanie dopasowane do danych (jak sweep)
    pr_lesionwise_sweep.svg     - charakterystyka lesion-wise przy zmianie progu tau
    *.csv                       - te same dane w formacie long, gotowe pod pgfplots

Skrypt celowo nie uzywa matplotliba (nie ma go w srodowisku /work/.../env/mri) -
SVG jest generowany wprost, wiec wynik jest wektorowy i wchodzi do LaTeX-a
przez \\includegraphics (pakiet svg) albo po konwersji do PDF.

Uruchomienie:
    python tools/plot_pr_curves.py
"""
import os
import sys
from pathlib import Path

import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.pr_metrics import pr_from_hist, simplify_curve   # noqa: E402

OUT_DIR = os.environ.get("PR_DIR", f"{project_root}/outputs/pr_curves")
REGIONS = ["WT", "TC", "ET"]

MODEL_STYLE = {
    "base":      dict(color="#0072B2", dash="",      label="Swin-UNETR Base"),
    "detach":    dict(color="#009E73", dash="7,3",   label="Multi-task (Detach)"),
    "multitask": dict(color="#D55E00", dash="2,3",   label="Multi-task (standard)"),
}

PANEL_W, PANEL_H = 330, 300
MARGIN_L, MARGIN_T = 62, 34
MARGIN_R, MARGIN_B = 18, 96
GAP = 30

FG = "#1a1a1a"
GRID = "#d8d8d8"
MUTED = "#666666"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Axes:
    """Prosty uklad wspolrzednych: dane -> piksele."""

    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim = xlim, ylim

    def px(self, x):
        a, b = self.xlim
        return self.x0 + (x - a) / (b - a) * self.w

    def py(self, y):
        a, b = self.ylim
        return self.y0 + self.h - (y - a) / (b - a) * self.h

    def frame(self, title, xlabel, ylabel, xticks, yticks, show_ylabel=True):
        out = [f'<rect x="{self.x0}" y="{self.y0}" width="{self.w}" height="{self.h}" '
               f'fill="#ffffff" stroke="{GRID}" stroke-width="1"/>']
        for t in xticks:
            x = self.px(t)
            out.append(f'<line x1="{x:.1f}" y1="{self.y0}" x2="{x:.1f}" '
                       f'y2="{self.y0+self.h}" stroke="{GRID}" stroke-width="0.8"/>')
            out.append(f'<text x="{x:.1f}" y="{self.y0+self.h+16}" font-size="11" '
                       f'fill="{MUTED}" text-anchor="middle">{t:g}</text>')
        for t in yticks:
            y = self.py(t)
            out.append(f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x0+self.w}" '
                       f'y2="{y:.1f}" stroke="{GRID}" stroke-width="0.8"/>')
            out.append(f'<text x="{self.x0-8}" y="{y+4:.1f}" font-size="11" '
                       f'fill="{MUTED}" text-anchor="end">{t:g}</text>')
        out.append(f'<text x="{self.x0+self.w/2:.1f}" y="{self.y0-12}" font-size="14" '
                   f'font-weight="600" fill="{FG}" text-anchor="middle">{esc(title)}</text>')
        out.append(f'<text x="{self.x0+self.w/2:.1f}" y="{self.y0+self.h+36}" font-size="12" '
                   f'fill="{FG}" text-anchor="middle">{esc(xlabel)}</text>')
        if show_ylabel:
            cy = self.y0 + self.h / 2
            out.append(f'<text x="{self.x0-42}" y="{cy:.1f}" font-size="12" fill="{FG}" '
                       f'text-anchor="middle" transform="rotate(-90 {self.x0-42} {cy:.1f})">'
                       f'{esc(ylabel)}</text>')
        return "\n".join(out)

    def polyline(self, xs, ys, color, dash="", width=1.9, opacity=1.0):
        pts = " ".join(f"{self.px(x):.2f},{self.py(y):.2f}" for x, y in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{width}"{d} stroke-opacity="{opacity}" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')

    def markers(self, xs, ys, color, r=2.6):
        return "\n".join(
            f'<circle cx="{self.px(x):.2f}" cy="{self.py(y):.2f}" r="{r}" '
            f'fill="{color}" stroke="#ffffff" stroke-width="0.8"/>'
            for x, y in zip(xs, ys))

    def star(self, x, y, color):
        cx, cy = self.px(x), self.py(y)
        pts = []
        for i in range(10):
            ang = -np.pi / 2 + i * np.pi / 5
            rad = 7.0 if i % 2 == 0 else 3.0
            pts.append(f"{cx+rad*np.cos(ang):.2f},{cy+rad*np.sin(ang):.2f}")
        return (f'<polygon points="{" ".join(pts)}" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1"/>')

    def annotate(self, x, y, text, dx=6, dy=-6):
        return (f'<text x="{self.px(x)+dx:.1f}" y="{self.py(y)+dy:.1f}" font-size="9.5" '
                f'fill="{MUTED}">{esc(text)}</text>')


def svg_document(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">\n'
            f'<title>{esc(title)}</title>\n'
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
            f'{body}\n</svg>\n')


def legend(x, y, entries, note=None):
    """entries: (kolor, dash, tekst)"""
    out = []
    for i, (color, dash, text) in enumerate(entries):
        yy = y + i * 19
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x}" y1="{yy}" x2="{x+30}" y2="{yy}" stroke="{color}" '
                   f'stroke-width="2.2"{d}/>')
        out.append(f'<text x="{x+38}" y="{yy+4}" font-size="12" fill="{FG}">{esc(text)}</text>')
    if note:
        out.append(f'<text x="{x}" y="{y + len(entries)*19 + 14}" font-size="11" '
                   f'fill="{MUTED}">{esc(note)}</text>')
    return "\n".join(out)


def load_runs():
    runs = {}
    for tag in MODEL_STYLE:
        p = os.path.join(OUT_DIR, f"pr_curves_{tag}.npz")
        if os.path.exists(p):
            runs[tag] = np.load(p, allow_pickle=True)
        else:
            print(f"  pomijam '{tag}' — brak {p}")
    return runs


def voxel_curve(run, ri):
    prec, rec, ap = pr_from_hist(run["hist_pos"][ri], run["hist_neg"][ri])
    if prec is None:
        return None, None, np.nan
    r, p = simplify_curve(rec, prec)
    return r, p, ap


def sweep_curve(run, ri, key="lesion_raw"):
    tot = run[key][:, ri].sum(axis=0)          # [n_th, 3] -> tp, fp, fn
    tp, fp, fn = tot[:, 0].astype(float), tot[:, 1].astype(float), tot[:, 2].astype(float)
    prec = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return rec, prec, tp, fp, fn


def detection_curve(run, ri):
    d = run["det_records"]
    n_gt = int(run["det_n_gt"][ri])
    if len(d) == 0 or n_gt == 0:
        return None, None, np.nan
    sub = d[d[:, 0] == ri]
    if len(sub) == 0:
        return None, None, np.nan
    o = np.argsort(-sub[:, 1])                 # malejaco po sredniej pewnosci
    is_tp = sub[o, 3]
    tp_c = np.cumsum(is_tp)
    fp_c = np.cumsum(1 - is_tp)
    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    rec = tp_c / n_gt
    rec_ext = np.append(0.0, rec)
    ap = float(np.sum((rec_ext[1:] - rec_ext[:-1]) * prec))
    return rec, prec, ap


def voxel_dice(run, ri):
    """
    Dice regionu binarnego == F1, wiec liczy sie w kazdym punkcie krzywej PR.
    Zwraca (dice przy tau=0.5, dice maksymalny, prog realizujacy maksimum).

    To NIE jest Dice w punkcie argmax - argmax to nie jest prog na p regionu.
    Dla ET roznica jest sprawdzalna wzgledem val/dice_ET z W&B.
    """
    prec, rec, _ = pr_from_hist(run["hist_pos"][ri], run["hist_neg"][ri])
    if prec is None:
        return np.nan, np.nan, np.nan
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    nbins = len(f1)
    k_half = nbins // 2                      # bin odpowiadajacy progowi 0.5
    k_best = int(np.argmax(f1))
    return float(f1[k_half]), float(f1[k_best]), k_best / nbins


def voxel_argmax_stats(run, ri):
    """
    Wokselowe Dice/Precision/Recall w punkcie decyzyjnym argmax.

    Zwraca (dice_macro, dice_micro, p_micro, r_micro, n_pacjentow_wazonych).

    'macro' to srednia po pacjentach w konwencji MONAI DiceMetric z domyslnym
    ignore_empty=True: pacjent, u ktorego region jest pusty w GT, daje NaN i
    wypada ze sredniej - nawet jesli model cos tam przewidzial. Dzieki temu
    liczba jest wprost porownywalna z val/dice_* z W&B oraz z wynikami
    eval_pr_metrics.py, na ktorych stoi Tabela 5.3.

    (Wariant "NaN dopiero gdy oba puste" zaniza wynik o ok. 0.01-0.016, bo
    wlicza zera za falszywe pozytywy u pacjentow bez danego regionu.)
    """
    if "voxel_argmax" not in run.files:
        return (np.nan,) * 4 + (0,)
    vx = run["voxel_argmax"][:, ri].astype(np.float64)
    tp, fp, fn = vx[:, 0], vx[:, 1], vx[:, 2]
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        dice_pp = np.where(tp + fn > 0, 2 * tp / np.maximum(den, 1), np.nan)
    return (float(np.nanmean(dice_pp)),
            float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
            float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
            float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
            int(np.sum(~np.isnan(dice_pp))))


def argmax_point(run, ri):
    am = run["lesion_argmax"][:, ri].sum(axis=0)
    p = am[0] / max(am[0] + am[1], 1)
    r = am[0] / max(am[0] + am[2], 1)
    return float(r), float(p), am


def figure_voxelwise(runs):
    width = MARGIN_L + 3 * PANEL_W + 2 * GAP + MARGIN_R
    height = MARGIN_T + PANEL_H + MARGIN_B
    body, rows = [], []
    for ri, reg in enumerate(REGIONS):
        ax = Axes(MARGIN_L + ri * (PANEL_W + GAP), MARGIN_T, PANEL_W, PANEL_H, (0, 1), (0, 1))
        body.append(ax.frame(reg, "Recall (czułość)", "Precision (precyzja)",
                             [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                             show_ylabel=(ri == 0)))
        for tag, st in MODEL_STYLE.items():
            if tag not in runs:
                continue
            r, p, ap = voxel_curve(runs[tag], ri)
            if r is None:
                continue
            body.append(ax.polyline(r, p, st["color"], st["dash"]))
            for rr, pp in zip(r, p):
                rows.append((tag, reg, f"{rr:.6f}", f"{pp:.6f}"))
    entries = []
    for tag, st in MODEL_STYLE.items():
        if tag not in runs:
            continue
        aps = [voxel_curve(runs[tag], ri)[2] for ri in range(3)]
        entries.append((st["color"], st["dash"],
                        f"{st['label']}  —  AP: WT {aps[0]:.3f} / TC {aps[1]:.3f} / ET {aps[2]:.3f}"))
    body.append(legend(MARGIN_L, MARGIN_T + PANEL_H + 58, entries,
                       "PR-AUC (average precision) uśredniane mikro — po wszystkich wokselach zbioru walidacyjnego."))
    return svg_document(width, height, "\n".join(body),
                        "Krzywe PR — voxel-wise"), rows


def figure_lesionwise(runs):
    width = MARGIN_L + 3 * PANEL_W + 2 * GAP + MARGIN_R
    height = MARGIN_T + PANEL_H + MARGIN_B
    body, rows = [], []
    for ri, reg in enumerate(REGIONS):
        ax = Axes(MARGIN_L + ri * (PANEL_W + GAP), MARGIN_T, PANEL_W, PANEL_H, (0, 1), (0, 1))
        body.append(ax.frame(reg, "Recall (detekcja ognisk)", "Precision (detekcja ognisk)",
                             [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                             show_ylabel=(ri == 0)))
        for tag, st in MODEL_STYLE.items():
            if tag not in runs:
                continue
            r, p, ap = detection_curve(runs[tag], ri)
            if r is None:
                continue
            body.append(ax.polyline(r, p, st["color"], st["dash"]))
            for rr, pp in zip(r, p):
                rows.append((tag, reg, f"{rr:.6f}", f"{pp:.6f}"))
            r_am, p_am, _ = argmax_point(runs[tag], ri)
            body.append(ax.star(r_am, p_am, st["color"]))
    entries = []
    for tag, st in MODEL_STYLE.items():
        if tag not in runs:
            continue
        aps = [detection_curve(runs[tag], ri)[2] for ri in range(3)]
        entries.append((st["color"], st["dash"],
                        f"{st['label']}  —  AP: WT {aps[0]:.3f} / TC {aps[1]:.3f} / ET {aps[2]:.3f}"))
    body.append(legend(MARGIN_L, MARGIN_T + PANEL_H + 58, entries,
                       "Gwiazdka = punkt pracy przy decyzji argmax (ten z Tabeli 5.2). "
                       "Ognisko rankingowane średnim prawdopodobieństwem w komponencie."))
    return svg_document(width, height, "\n".join(body),
                        "Krzywe PR — lesion-wise (detection AP)"), rows


# Below this recall, detection_curve's precision is flat (~1); unlike
# sweep_curve (inherently narrow-range), detection_curve runs from recall
# near 0, so without a cutoff the adaptive axis limits wouldn't narrow the view.
ZOOM_RECALL_CUTOFF = 0.5


def figure_lesionwise_zoomed(runs):
    """
    Ta sama krzywa co figure_lesionwise (detection AP, ranking po pewnosci), ale w kadrowaniu
    jak figure_sweep: granice osi dobrane do danych (dopelnienie 12% zakresu, min 0.01,
    przyciecie do [0,1], 5 rownych tickow), markery na krzywej zamiast samej linii.
    """
    width = MARGIN_L + 3 * PANEL_W + 2 * GAP + MARGIN_R
    height = MARGIN_T + PANEL_H + MARGIN_B
    body, rows = [], []

    def ticks(lim):
        return [round(lim[0] + i * (lim[1] - lim[0]) / 4, 3) for i in range(5)]

    for ri, reg in enumerate(REGIONS):
        curves = {}
        for tag in runs:
            r, p, ap = detection_curve(runs[tag], ri)
            if r is None:
                continue
            mask = r >= ZOOM_RECALL_CUTOFF
            curves[tag] = (r[mask], p[mask], ap)

        allr = np.concatenate([c[0] for c in curves.values()])
        allp = np.concatenate([c[1] for c in curves.values()])
        pad_r = max((allr.max() - allr.min()) * 0.12, 0.01)
        pad_p = max((allp.max() - allp.min()) * 0.12, 0.01)
        xlim = (max(allr.min() - pad_r, 0), min(allr.max() + pad_r, 1))
        ylim = (max(allp.min() - pad_p, 0), min(allp.max() + pad_p, 1))

        ax = Axes(MARGIN_L + ri * (PANEL_W + GAP), MARGIN_T, PANEL_W, PANEL_H, xlim, ylim)
        body.append(ax.frame(reg, "Recall (detekcja ognisk)", "Precision (detekcja ognisk)",
                             ticks(xlim), ticks(ylim), show_ylabel=(ri == 0)))
        for tag, st in MODEL_STYLE.items():
            if tag not in curves:
                continue
            r, p, ap = curves[tag]
            body.append(ax.polyline(r, p, st["color"], st["dash"]))
            idx = np.linspace(0, len(r) - 1, min(25, len(r))).astype(int)
            body.append(ax.markers(r[idx], p[idx], st["color"]))
            for rr, pp in zip(r, p):
                rows.append((tag, reg, f"{rr:.6f}", f"{pp:.6f}"))
            r_am, p_am, _ = argmax_point(runs[tag], ri)
            if xlim[0] <= r_am <= xlim[1] and ylim[0] <= p_am <= ylim[1]:
                body.append(ax.star(r_am, p_am, st["color"]))

    entries = []
    for tag, st in MODEL_STYLE.items():
        if tag not in runs:
            continue
        aps = [detection_curve(runs[tag], ri)[2] for ri in range(3)]
        entries.append((st["color"], st["dash"],
                        f"{st['label']}  —  AP: WT {aps[0]:.3f} / TC {aps[1]:.3f} / ET {aps[2]:.3f}"))
    body.append(legend(MARGIN_L, MARGIN_T + PANEL_H + 58, entries,
                       f"Gwiazdka = punkt pracy przy decyzji argmax. Oś przycięta do recall ≥ "
                       f"{ZOOM_RECALL_CUTOFF:g} — poniżej tego progu precyzja pozostaje bliska 1."))
    return svg_document(width, height, "\n".join(body),
                        "Krzywe PR — lesion-wise (detection AP), kadrowanie dopasowane do danych"), rows


def figure_sweep(runs):
    """Charakterystyka lesion-wise przy zmianie progu tau. Osie skalowane do danych."""
    width = MARGIN_L + 3 * PANEL_W + 2 * GAP + MARGIN_R
    height = MARGIN_T + PANEL_H + MARGIN_B
    body, rows = [], []
    for ri, reg in enumerate(REGIONS):
        rs, ps = [], []
        for tag in runs:
            r, p, *_ = sweep_curve(runs[tag], ri)
            rs.append(r)
            ps.append(p)
        allr, allp = np.concatenate(rs), np.concatenate(ps)
        rlo, rhi = allr.min(), allr.max()
        plo, phi = allp.min(), allp.max()
        pad_r = max((rhi - rlo) * 0.12, 0.01)
        pad_p = max((phi - plo) * 0.12, 0.01)
        xlim = (max(rlo - pad_r, 0), min(rhi + pad_r, 1))
        ylim = (max(plo - pad_p, 0), min(phi + pad_p, 1))

        def ticks(lim):
            return [round(lim[0] + i * (lim[1] - lim[0]) / 4, 3) for i in range(5)]

        ax = Axes(MARGIN_L + ri * (PANEL_W + GAP), MARGIN_T, PANEL_W, PANEL_H, xlim, ylim)
        body.append(ax.frame(reg, "Recall (detekcja ognisk)", "Precision (detekcja ognisk)",
                             ticks(xlim), ticks(ylim), show_ylabel=(ri == 0)))
        for tag, st in MODEL_STYLE.items():
            if tag not in runs:
                continue
            run = runs[tag]
            r, p, tp, fp, fn = sweep_curve(run, ri)
            th = run["thresholds"]
            body.append(ax.polyline(r, p, st["color"], st["dash"]))
            body.append(ax.markers(r, p, st["color"]))
            for t, rr, pp, a, b, c in zip(th, r, p, tp, fp, fn):
                rows.append((tag, reg, f"{t:g}", int(a), int(b), int(c),
                             f"{rr:.6f}", f"{pp:.6f}"))
            if tag == "multitask":      # anotacje tau tylko raz, zeby nie zasmiecac
                for t_mark in (0.05, 0.5, 0.9, 0.99):
                    j = int(np.argmin(np.abs(th - t_mark)))
                    body.append(ax.annotate(r[j], p[j], f"τ={th[j]:g}"))
            r_am, p_am, _ = argmax_point(run, ri)
            if xlim[0] <= r_am <= xlim[1] and ylim[0] <= p_am <= ylim[1]:
                body.append(ax.star(r_am, p_am, st["color"]))
    entries = [(st["color"], st["dash"], st["label"])
               for tag, st in MODEL_STYLE.items() if tag in runs]
    body.append(legend(MARGIN_L, MARGIN_T + PANEL_H + 58, entries,
                       "Uwaga: sweep pokrywa tylko wąski, wysoki zakres recall — duże ogniska są "
                       "wykrywane przy każdym progu. Całkowanie po tym wycinku nie jest pełnym PR-AUC."))
    return svg_document(width, height, "\n".join(body),
                        "Lesion-wise — charakterystyka przy zmianie progu"), rows


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def write_summary(runs):
    rows = []
    for tag, st in MODEL_STYLE.items():
        if tag not in runs:
            continue
        run = runs[tag]
        ap_pp = run["per_patient_ap"]
        for ri, reg in enumerate(REGIONS):
            _, _, ap_micro = voxel_curve(run, ri)
            ap_macro = float(np.nanmean(ap_pp[:, ri]))
            _, _, det_ap = detection_curve(run, ri)
            r_am, p_am, am = argmax_point(run, ri)
            r_sw, p_sw, *_ = sweep_curve(run, ri)
            d05, dmax, tbest = voxel_dice(run, ri)
            vdmac, vdmic, vp, vr, vn = voxel_argmax_stats(run, ri)
            rows.append((
                tag, reg,
                f"{ap_micro:.6f}", f"{ap_macro:.6f}", f"{det_ap:.6f}",
                f"{vdmac:.6f}", f"{vdmic:.6f}", f"{vp:.6f}", f"{vr:.6f}", vn,
                f"{d05:.6f}", f"{dmax:.6f}", f"{tbest:.4f}",
                f"{r_am:.6f}", f"{p_am:.6f}", int(am[0]), int(am[1]), int(am[2]),
                f"{r_sw.min():.6f}", f"{r_sw.max():.6f}",
            ))
    write_csv(os.path.join(OUT_DIR, "pr_summary.csv"),
              ["model", "region", "voxel_ap_micro", "voxel_ap_macro", "lesion_detection_ap",
               "voxel_dice_argmax_macro", "voxel_dice_argmax_micro",
               "voxel_precision_argmax", "voxel_recall_argmax", "voxel_dice_n_patients",
               "voxel_dice_tau05", "voxel_dice_max", "tau_at_max_dice",
               "lesion_argmax_recall", "lesion_argmax_precision",
               "lesion_argmax_tp", "lesion_argmax_fp", "lesion_argmax_fn",
               "sweep_recall_min", "sweep_recall_max"], rows)
    return rows


# reference val/dice_ET values from W&B, for cross-checking
WANDB_DICE_ET = {"base": 0.8743857741355896,
                 "detach": 0.8721252679824829,
                 "multitask": 0.8682851195335388}


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f"Brak plikow pr_curves_*.npz w {OUT_DIR}")
    print(f"Wczytano modele: {', '.join(runs)}")
    n_pac = len(runs[list(runs)[0]]["patient_ids"])
    print(f"Pacjentow w pliku: {n_pac}")

    svg, rows = figure_voxelwise(runs)
    open(os.path.join(OUT_DIR, "pr_voxelwise.svg"), "w").write(svg)
    write_csv(os.path.join(OUT_DIR, "pr_voxelwise.csv"),
              ["model", "region", "recall", "precision"], rows)

    svg, rows = figure_lesionwise(runs)
    open(os.path.join(OUT_DIR, "pr_lesionwise.svg"), "w").write(svg)
    write_csv(os.path.join(OUT_DIR, "pr_lesionwise.csv"),
              ["model", "region", "recall", "precision"], rows)

    svg, rows = figure_lesionwise_zoomed(runs)
    open(os.path.join(OUT_DIR, "pr_lesionwise_zoomed.svg"), "w").write(svg)
    write_csv(os.path.join(OUT_DIR, "pr_lesionwise_zoomed.csv"),
              ["model", "region", "recall", "precision"], rows)

    svg, rows = figure_sweep(runs)
    open(os.path.join(OUT_DIR, "pr_lesionwise_sweep.svg"), "w").write(svg)
    write_csv(os.path.join(OUT_DIR, "pr_lesionwise_sweep.csv"),
              ["model", "region", "threshold", "tp", "fp", "fn", "recall", "precision"], rows)

    summary = write_summary(runs)
    print("\n" + "=" * 108)
    print("VOXEL-WISE")
    print(f"{'model':<11} {'reg':<4} {'PR-AUC micro':>13} {'PR-AUC macro':>13} "
          f"{'Dice@argmax macro':>18} {'Dice@argmax micro':>18} {'P':>8} {'R':>8}")
    print("-" * 108)
    for r in summary:
        print(f"{r[0]:<11} {r[1]:<4} {r[2]:>13} {r[3]:>13} {r[5]:>18} {r[6]:>18} "
              f"{r[7]:>8} {r[8]:>8}")
    print("\nLESION-WISE")
    print(f"{'model':<11} {'reg':<4} {'detection AP':>13} {'argmax P':>9} "
          f"{'argmax R':>9} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 108)
    for r in summary:
        print(f"{r[0]:<11} {r[1]:<4} {r[4]:>13} {r[14]:>9} {r[13]:>9} "
              f"{r[15]:>5} {r[16]:>5} {r[17]:>5}")
    print("=" * 108)

    # W&B logs DiceMetric(reduction='mean_batch'), i.e. mean per patient - compare to macro, not micro
    print("\nCross-check for ET (region ET == class 3, so val/dice_ET is directly comparable):")
    print(f"  {'model':<11} {'W&B macro':>10} {'my macro':>11} {'|diff|':>10} "
          f"{'(micro)':>9}")
    for r in summary:
        if r[1] != "ET" or r[0] not in WANDB_DICE_ET:
            continue
        wb, mac, mic = WANDB_DICE_ET[r[0]], float(r[5]), float(r[6])
        print(f"  {r[0]:<11} {wb:>10.4f} {mac:>11.4f} {abs(mac-wb):>10.4f} {mic:>9.4f}")
    print(f"\nSaved SVG + CSV to {OUT_DIR}")


if __name__ == "__main__":
    main()
