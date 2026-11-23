# Figures and tables for the final report.
#
# Inputs (all written by the RIPE_final.ipynb pipeline):
#   --runs-dir  <dir>/<run>/train_metrics.csv, val_metrics.csv   (ripe/train.py)
#   --eval-dir  <dir>/{hpatches,megadepth1500}/<tag>/summaries.json, results.h5  (eval_gluefactory.py)
#   --cost-json output of `compare_backbones.py --all --json ...`
# Run names / eval tags are "<variant>_s<seed>" (e.g. resnet18_s32000); the official weights use the
# tag "official_vgg19_80k". Several seeds of a variant are averaged (error bars = min/max).
#
# Outputs (in --out): fig_*.png and results_table.{md,tex}. Missing inputs are skipped, so the
# script can be re-run after every finished run.

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fixed colour per model (colour follows the entity, never its rank); the official model is a
# grey reference because it was trained far longer than everything else
VARIANTS = {
    "official_vgg19_80k": dict(label="VGG-19 official (80k steps)", color="#8a8984", cost="VGG-19 (original)"),
    "vgg19": dict(label="VGG-19 (retrained)", color="#2a78d6", cost="VGG-19 (original)"),
    "resnet18": dict(label="ResNet-18 (ours)", color="#eb6834", cost="ResNet-18 (ours)"),
    "resnet18_layer4": dict(label="ResNet-18 + layer4", color="#1baf7a", cost="ResNet-18 + layer4"),
    "resnet18_narrowstem": dict(label="ResNet-18 narrow stem", color="#eda100", cost="ResNet-18 narrow stem"),
    "resnet18_scratch": dict(label="ResNet-18 no ImageNet init", color="#e87ba4", cost="ResNet-18 (ours)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def split_tag(tag):
    m = re.match(r"^(.*)_s(\d+)$", tag)
    return (m.group(1), int(m.group(2))) if m else (tag, None)


def ordered(variants):
    return [v for v in VARIANTS if v in variants] + sorted(v for v in variants if v not in VARIANTS)


def meta(v):
    return VARIANTS.get(v, dict(label=v, color="#6250d6", cost=None))


# ---------------------------------------------------------------- loading
def load_runs(runs_dir):
    runs = defaultdict(list)  # variant -> [(seed, train_df, val_df)]
    if not runs_dir or not Path(runs_dir).exists():
        return runs
    for d in sorted(Path(runs_dir).iterdir()):
        if not (d / "train_metrics.csv").exists():
            continue
        tr = pd.read_csv(d / "train_metrics.csv").drop_duplicates("step", keep="last").sort_values("step")
        va = None
        if (d / "val_metrics.csv").exists():
            va = pd.read_csv(d / "val_metrics.csv").drop_duplicates("step", keep="last").sort_values("step")
        v, seed = split_tag(d.name)
        runs[v].append((seed, tr, va))
    return runs


def load_eval(eval_dir):
    ev = {}  # bench -> variant -> list of (summaries, per-pair errors)
    for bench, key in (("hpatches", "H_error_ransac"), ("megadepth1500", "rel_pose_error")):
        ev[bench] = defaultdict(list)
        root = Path(eval_dir) / bench if eval_dir else None
        if root is None or not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not (d / "summaries.json").exists():
                continue
            s = json.loads((d / "summaries.json").read_text())
            errs = None
            if (d / "results.h5").exists():
                import h5py

                with h5py.File(d / "results.h5", "r") as f:
                    if key in f:
                        errs = np.array(f[key])
            ev[bench][split_tag(d.name)[0]].append((s, errs))
    return ev


def auc_keys(bench):
    return (
        ["H_error_ransac@1px", "H_error_ransac@3px", "H_error_ransac@5px"]
        if bench == "hpatches"
        else ["rel_pose_error@5°", "rel_pose_error@10°", "rel_pose_error@20°"]
    )


def stat(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), a.min(), a.max(), len(a)


# ---------------------------------------------------------------- figures
def ema(x, alpha=0.05):
    return pd.Series(x).ewm(alpha=alpha).mean().values


def fig_training(runs, out, budget):
    panels = [
        ("mean_reward", "Mean reward per pair"),
        ("num_kpts", "Selected keypoints per image"),
        ("loss_desc", "Descriptor loss"),
        ("skipped", "Skipped pairs (moving avg.)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
    for ax, (col, title) in zip(axes.ravel(), panels):
        for v in ordered(runs):
            for k, (seed, tr, _) in enumerate(runs[v]):
                y = (tr["num_kpts1"] + tr["num_kpts2"]) / 2 if col == "num_kpts" else tr[col]
                ax.plot(tr["step"], ema(y), color=meta(v)["color"], alpha=1.0 if k == 0 else 0.5,
                        label=meta(v)["label"] if k == 0 else None)
        ax.set_title(title, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("training step")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Training dynamics under an equal budget ({budget}); EMA-smoothed, one line per seed",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_training_curves.png")
    plt.close(fig)


def fig_val(runs, out):
    if not any(va is not None for v in runs for _, _, va in runs[v]):
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, th in zip(axes, (5, 10, 20)):
        for v in ordered(runs):
            for k, (seed, _, va) in enumerate(runs[v]):
                if va is None or f"auc@{th}" not in va:
                    continue
                ax.plot(va["step"], va[f"auc@{th}"], marker="o", markersize=4, color=meta(v)["color"],
                        alpha=1.0 if k == 0 else 0.5, label=meta(v)["label"] if k == 0 else None)
        ax.set_title(f"AUC@{th}°", loc="left")
        ax.set_xlabel("training step")
    axes[0].set_ylabel("IMW2020-val pose AUC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation during training (IMW2020-val, 200 fixed pairs, RIPE's built-in benchmark)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_val_auc.png")
    plt.close(fig)


def fig_bars(ev, bench, out, title, unit):
    vs = ordered(ev[bench])
    if not vs:
        return
    keys = auc_keys(bench)
    n = len(vs)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(keys) * n / 2, 3.4))
    x = np.arange(len(keys))
    for i, v in enumerate(vs):
        st = [stat([s[k] for s, _ in ev[bench][v]]) for k in keys]
        mean = np.array([s[0] for s in st])
        lo = mean - np.array([s[1] for s in st])
        hi = np.array([s[2] for s in st]) - mean
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, mean, width * 0.92, color=meta(v)["color"], label=meta(v)["label"] + (f" (n={st[0][3]})" if st[0][3] > 1 else ""))
        if st[0][3] > 1:
            ax.errorbar(pos, mean, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for p, m in zip(pos, mean):
            ax.text(p, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=6.5, color=MUTED, rotation=90)
    ax.set_xticks(x, [k.split("@")[1] for k in keys])
    ax.set_xlabel(f"error threshold ({unit})")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"fig_{bench}_auc.png")
    plt.close(fig)


def fig_cumulative(ev, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    specs = [("hpatches", 10, "px", "HPatches: homography corner error"),
             ("megadepth1500", 30, "°", "MegaDepth-1500: relative pose error")]
    drawn = False
    for ax, (bench, xmax, unit, title) in zip(axes, specs):
        for v in ordered(ev[bench]):
            for k, (_, errs) in enumerate(ev[bench][v]):
                if errs is None:
                    continue
                e = np.sort(np.nan_to_num(errs, nan=np.inf, posinf=np.inf))
                t = np.linspace(0, xmax, 300)
                ax.plot(t, np.searchsorted(e, t, side="right") / len(e), color=meta(v)["color"],
                        alpha=1.0 if k == 0 else 0.5, label=meta(v)["label"] if k == 0 else None)
                drawn = True
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"threshold ({unit})")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("fraction of pairs below threshold")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    if drawn:
        fig.savefig(out / "fig_cumulative_errors.png")
    plt.close(fig)


def cost_rows(cost_json):
    if not cost_json or not Path(cost_json).exists():
        return None, {}
    c = json.loads(Path(cost_json).read_text())
    return c, {r["name"]: r for r in c["rows"]}


def fig_cost(cost, out):
    c, rows = cost
    if not rows:
        return
    names = list(rows)
    ref = rows[names[0]]
    metrics = [("enc", "Encoder params"), ("total", "Total params"), ("ms", "Forward time"), ("mem", "Peak memory")]
    colors = {VARIANTS[v]["cost"]: VARIANTS[v]["color"] for v in ("vgg19", "resnet18", "resnet18_layer4", "resnet18_narrowstem")}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [rows[n][m] / ref[m] for m, _ in metrics]
        pos = x - 0.4 + w * (i + 0.5)
        ax.bar(pos, vals, w * 0.92, color=colors.get(n, "#6250d6"), label=n)
        for p, val in zip(pos, vals):
            ax.text(p, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylabel("relative to VGG-19 (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Computational cost, {c['device']}, {c['size'][0]}x{c['size'][1]} input", loc="left")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_cost_relative.png")
    plt.close(fig)


def fig_tradeoff(ev, cost, out):
    _, rows = cost
    if not rows:
        return
    specs = [("megadepth1500", "rel_pose_error@10°", "MegaDepth-1500 AUC@10°"),
             ("hpatches", "H_error_ransac@3px", "HPatches AUC@3px")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    drawn = False
    for ax, (bench, key, ylabel) in zip(axes, specs):
        for v in ordered(ev[bench]):
            r = rows.get(meta(v)["cost"])
            if r is None:
                continue
            m, lo, hi, n = stat([s[key] for s, _ in ev[bench][v]])
            ax.errorbar(r["ms"], m, yerr=[[m - lo], [hi - m]] if n > 1 else None, fmt="o", markersize=8,
                        color=meta(v)["color"], markeredgecolor="white", markeredgewidth=1.5, ecolor=INK, capsize=2)
            ax.annotate(f"{meta(v)['label']}\n{r['total']:.1f} M params", (r["ms"], m), xytext=(7, -3),
                        textcoords="offset points", fontsize=7, color=INK)
            drawn = True
        ax.set_xlabel("forward time per image (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title("Accuracy vs. cost (up-left is better)", loc="left")
        ax.margins(x=0.35, y=0.2)
    fig.tight_layout()
    if drawn:
        fig.savefig(out / "fig_accuracy_vs_cost.png")
    plt.close(fig)


# ---------------------------------------------------------------- table
def write_tables(ev, runs, cost, out, budget):
    _, rows = cost
    vs = ordered(set(ev["hpatches"]) | set(ev["megadepth1500"]) | set(runs))
    cols = ["Model", "#runs", "HP@1", "HP@3", "HP@5", "MD@5", "MD@10", "MD@20", "Params (M)", "ms/img", "s/train step"]
    lines = []
    for v in vs:
        cells = [meta(v)["label"]]
        n = max(len(ev["hpatches"].get(v, [])), len(ev["megadepth1500"].get(v, [])))
        cells.append(str(n))
        for bench in ("hpatches", "megadepth1500"):
            for k in auc_keys(bench):
                vals = [s[k] for s, _ in ev[bench].get(v, [])]
                if not vals:
                    cells.append("-")
                else:
                    m, lo, hi, n_ = stat(vals)
                    cells.append(f"{m:.4f}" + (f" ±{np.std(vals):.4f}" if n_ > 1 else ""))
        r = rows.get(meta(v)["cost"])
        cells += [f"{r['total']:.2f}" if r else "-", f"{r['ms']:.1f}" if r else "-"]
        sps = [tr["sec_per_step"].iloc[1:].median() for _, tr, _ in runs.get(v, []) if len(tr) > 1]
        cells.append(f"{np.mean(sps):.2f}" if sps else "-")
        lines.append(cells)

    md = [f"Equal training budget: {budget}. HP = HPatches homography AUC (px), MD = MegaDepth-1500 pose AUC (°).", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(c) + " |" for c in lines]
    (out / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = ["\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    tex += [" & ".join(c).replace("±", "$\\pm$") + " \\\\" for c in lines]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/content/drive/MyDrive/ripe_final/runs")
    ap.add_argument("--eval-dir", default="/content/drive/MyDrive/ripe_final/eval")
    ap.add_argument("--cost-json", default="/content/drive/MyDrive/ripe_final/cost.json")
    ap.add_argument("--out", default="/content/drive/MyDrive/ripe_final/figures")
    ap.add_argument("--budget", default="reduced budget", help='e.g. "5k steps" (used in titles)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs_dir)
    ev = load_eval(args.eval_dir)
    cost = cost_rows(args.cost_json)

    if runs:
        fig_training(runs, out, args.budget)
        fig_val(runs, out)
    fig_bars(ev, "hpatches", out, "HPatches homography estimation (540 pairs)", "px")
    fig_bars(ev, "megadepth1500", out, "MegaDepth-1500 relative pose estimation (1500 pairs)", "°")
    fig_cumulative(ev, out)
    fig_cost(cost, out)
    fig_tradeoff(ev, cost, out)
    write_tables(ev, runs, cost, out, args.budget)
    print("figures written to", out)


if __name__ == "__main__":
    main()
