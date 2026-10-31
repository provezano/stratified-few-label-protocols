"""All figures of the P1 draft. Run from paper/p1/:  nice -n 15 ../../.venv/bin/python figures/make_figures.py

Every number in a figure is computed here from data on disk (nothing is typed in by hand):
- results/study7/{a,b,c}       Study 7 (DEV): pre-registered FMs fm1, fm2, fm3, fm5 and the balanced variants;
- results/study12              Study 12 (T1'): pre-registered FMs, arm T (stratified and random rows only);
- exploratory/fm_extra/out     exploratory FMs: DEV fm6 and fm8 in s7a/, fm7 (seeded re-run) in s7a_v2/;
                               T1' fm6 and fm7 in t1p/, fm8 in t1p_fm8/ (same labelled sets and seeds as Studies 7 and 12);
- exploratory/out/e1_gaps.csv, exploratory/e1_t1p/out/e1t1p_gaps.csv   six classical learners (E1);
- exploratory/out/e10_pools_dev.csv   pool imbalance TV of each DEV table;
- exploratory/review_reanalysis/out/mech_tables.csv   coverage-model prediction per DEV table,
  Study 7 Part (a), TabPFN v3.5 (Fig. 4a; exploratory re-analysis of pre-registered fits).

Outputs (vector PDF plus a PNG preview), sized for the 6.5 in TMLR text width and included at \\linewidth:
  figures/fig1_protocols, fig2_gap, fig3_ranks, fig4_mechanism, fig_forest, fig8_heldout_placeholder;
  exploratory/out/e6_accuracy_imbalance (path kept for main.tex).
Encodings fixed across all figures (see FM below): one colour and one marker shape per FM;
filled marks = pre-registered FM, open marks = exploratory FM. Colour never carries meaning alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from simaudit.analysis import study7 as a7  # noqa: E402  (Nemenyi CD, as in the Study 7 report)

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
SIM = P1.parents[1]
RES7 = SIM / "results" / "study7"
RES12 = SIM / "results" / "study12"
REP = SIM / "reports"
EXP = P1 / "exploratory"
FMX = EXP / "fm_extra" / "out"

# ------------------------------------------------------------------ encodings ---------
# Fixed order, colour and marker per FM. Colours: validated default categorical palette, slots 1-7
# (adjacent pairs pass the CVD check; with seven series the marker shape and the fixed dodge position
# are the secondary encodings). Pre-registered FMs are drawn filled, exploratory FMs open.
FM = {
    "fm1": dict(label="TabPFN v2", color="#2a78d6", marker="o", prereg=True),
    "fm2": dict(label="TabICLv2", color="#eb6834", marker="s", prereg=True),
    "fm3": dict(label="Kumo-Tabular", color="#1baf7a", marker="^", prereg=True),
    "fm5": dict(label="TabPFN v3.5", color="#eda100", marker="D", prereg=True),
    "fm6": dict(label="TabDPT v1.3", color="#e87ba4", marker="v", prereg=False),
    "fm7": dict(label="Mitra-v2", color="#008300", marker="p", prereg=False),
    "fm8": dict(label="LimiX-16M", color="#6250d6", marker="h", prereg=False),
}
PREREG = ["fm1", "fm2", "fm3", "fm5"]
ALL_FMS = list(FM)
MULTS = [1, 3, 5, 10]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#9a9890"
BAND = "#ecebe6"  # neutral context band (classical learners)
MS = 4.6  # marker size (pt) for summary marks

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASE, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.1,
    "pdf.fonttype": 42, "savefig.dpi": 200,
})


def mark_kw(fm: str, size: float = MS) -> dict:
    """Marker style for one FM: filled if pre-registered, open (white face) if exploratory."""
    f = FM[fm]
    return dict(marker=f["marker"], ms=size, mec=f["color"], mew=1.0,
                mfc=f["color"] if f["prereg"] else "white")


def fm_label(fm: str) -> str:
    return FM[fm]["label"]


def fm_handles(fms: list[str]) -> list[Line2D]:
    return [Line2D([], [], color=FM[f]["color"], lw=0, label=fm_label(f), **mark_kw(f)) for f in fms]


def fm_legend(fig, fms: list[str], extra: list | None = None, y: float = 1.0, x: float = 0.5,
              tight: bool = False) -> None:
    """Two-row legend: pre-registered FMs (filled) on the first row, exploratory FMs (open) and extras below."""
    blank = Line2D([], [], lw=0, label=" ")
    row1 = [Line2D([], [], lw=0, label="pre-registered:")] + fm_handles([f for f in fms if FM[f]["prereg"]])
    row2 = [Line2D([], [], lw=0, label="exploratory:")] + fm_handles([f for f in fms if not FM[f]["prereg"]])
    row2 += list(extra or [])
    n = max(len(row1), len(row2))
    row1 += [blank] * (n - len(row1))
    row2 += [blank] * (n - len(row2))
    handles = [h for pair in zip(row1, row2) for h in pair]  # the legend fills column by column
    leg = fig.legend(handles=handles, loc="lower center", ncol=n, bbox_to_anchor=(x, y), handlelength=1.0,
                     columnspacing=0.9 if tight else 1.2, labelspacing=0.15 if tight else 0.5,
                     borderpad=0.1 if tight else 0.4)
    for t, h in zip(leg.get_texts(), handles):
        if h.get_label().endswith(":"):
            t.set_color(INK2)


def provenance_handles() -> list[Line2D]:
    return [Line2D([], [], color=INK2, lw=0, marker="o", ms=MS, mfc=INK2, mec=INK2, label="filled: pre-registered"),
            Line2D([], [], color=INK2, lw=0, marker="o", ms=MS, mfc="white", mec=INK2, label="open: exploratory")]


def zero_line(ax, axis: str = "y") -> None:
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK2, lw=0.8, zorder=1)


def boot_ci(x, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean (unit = table)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


TEXT_WIDTH = 6.5  # in; figures are included at \linewidth, so fonts print at their nominal size


def save(fig, path: Path) -> None:
    # widen the canvas until the tight-cropped figure is exactly the text width (no rescaling in LaTeX)
    for _ in range(4):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        if abs(bb.width - TEXT_WIDTH) < 0.01:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + TEXT_WIDTH - bb.width, h)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})  # no local timezone in the PDF
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {path.relative_to(P1)}")


def budget_axis(ax) -> None:
    ax.set_xticks(range(len(MULTS)), [f"{m}C" for m in MULTS])
    ax.set_xlim(-0.5, len(MULTS) - 0.5)
    ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------ data --------------
RUN_COLS = ["dataset", "learner", "strategy", "mult", "rep", "macro_f1", "accuracy"]


def _read(files) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f, columns=RUN_COLS) for f in files], ignore_index=True)
    d["dataset"] = d.dataset.astype(str)
    return d[d.strategy.isin(["stratified", "random"])]


def dev_runs() -> pd.DataFrame:
    """Study 7 Part (a) draws for the seven FMs on DEV (same labelled sets for every FM)."""
    files = [f for fm in PREREG for f in sorted((RES7 / "a").glob(f"*_{fm}.parquet"))]
    files += sorted((FMX / "s7a").glob("*_fm6.parquet")) + sorted((FMX / "s7a").glob("*_fm8.parquet"))
    files += sorted((FMX / "s7a_v2").glob("*_fm7.parquet"))  # Mitra-v2, seeded per fit (the run the paper uses)
    return _read(files)


def t1p_runs() -> pd.DataFrame:
    """Study 12 draws (arm T) on T1' for the pre-registered FMs, plus the exploratory fm6, fm7 and fm8."""
    files = [f for fm in PREREG for f in sorted(RES12.glob(f"*_{fm}.parquet"))]
    files += sorted((FMX / "t1p").glob("*_fm6.parquet")) + sorted((FMX / "t1p").glob("*_fm7.parquet"))
    files += sorted((FMX / "t1p_fm8").glob("*_fm8.parquet"))  # LimiX-16M, same Study 12 draws
    return _read(files)


def table_gaps(runs: pd.DataFrame, metric: str) -> pd.Series:
    """Per (learner, dataset, mult): mean over reps of stratified minus mean over reps of random."""
    g = runs.groupby(["learner", "dataset", "mult", "strategy"])[metric].mean().unstack("strategy")
    return (g["stratified"] - g["random"]).dropna()


def classical_range(path: Path, metric: str, set_: str | None = None) -> pd.DataFrame:
    """Min and max over the six classical learners (E1) of the mean gap per budget."""
    g = pd.read_csv(path)
    if set_ is not None and "set" in g:
        g = g[g.set == set_]
    g = g[(g.metric == metric) & g.learner.isin(["lr", "knn1", "knn3", "gnb", "lda", "hgb"])]
    g = g[g.mult.astype(str).isin([str(m) for m in MULTS])].assign(mult=lambda d: d.mult.astype(int))
    assert g.learner.nunique() == 6, f"expected six classical learners in {path.name}"
    return g.groupby("mult")["mean"].agg(["min", "max"])


# ------------------------------------------------------------------ Fig 1 -------------
def fig_protocols() -> None:
    """Schematic: one imbalanced pool (C = 3), two ways to spend B = 3 labels."""
    rng = np.random.default_rng(3)
    counts = [30, 12, 4]
    centers = [(0.30, 0.55), (0.72, 0.62), (0.52, 0.22)]
    shapes = ["o", "s", "^"]
    tones = ["#3d3c39", "#7d7b75", "#b3b1aa"]  # neutral class tones: class identity also carried by shape
    X = np.vstack([rng.normal(c, 0.085, size=(n, 2)) for n, c in zip(counts, centers)])
    y = np.repeat(np.arange(3), counts)
    strat = [int(rng.choice(np.where(y == k)[0])) for k in range(3)]
    rand = [int(i) for i in rng.choice(np.where(y == 0)[0], 2, replace=False)] + [int(rng.choice(np.where(y == 1)[0]))]
    panels = [("Stratified: $m=1$ row of each class", strat, "labelled per class: 1 / 1 / 1\nbalanced; choosing them needs $y$"),
              ("Random: $B=mC=3$ rows at random", rand, "labelled per class: 2 / 1 / 0\nclass 3 is missed; shares follow the pool")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.55), gridspec_kw={"wspace": 0.08})
    for ax, (title, sel, note) in zip(axes, panels):
        for k in range(3):
            m = y == k
            ax.scatter(X[m, 0], X[m, 1], s=16, marker=shapes[k], facecolor="none", edgecolor=tones[k], lw=0.7)
        for i in sel:
            ax.scatter(X[i, 0], X[i, 1], s=80, marker=shapes[y[i]], facecolor=INK, edgecolor="white", lw=1.5, zorder=3)
        ax.set_title(title, loc="left")
        ax.text(0.0, -0.04, note, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 0.95)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
    leg = [Line2D([], [], lw=0, marker=shapes[k], ms=5, mfc="none", mec=tones[k], label=f"class {k + 1} ({counts[k]} rows)")
           for k in range(3)]
    leg.append(Line2D([], [], lw=0, marker="o", ms=7, mfc=INK, mec="white", label="labelled row"))
    fig.legend(handles=leg, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    save(fig, HERE / "fig1_protocols")


# ------------------------------------------------------------------ Fig 2 -------------
def gap_panel(ax, gaps: pd.Series, fms: list[str], band: pd.DataFrame | None) -> tuple[float, float]:
    """Mean gap per FM and budget with bootstrap 95% CIs over tables; FMs dodged in a fixed order."""
    zero_line(ax)
    if band is not None:
        for i, m in enumerate(MULTS):
            ax.add_patch(plt.Rectangle((i - 0.42, band.loc[m, "min"]), 0.84, band.loc[m, "max"] - band.loc[m, "min"],
                                       color=BAND, lw=0, zorder=0))
    step = 0.105
    ext = [0.0, 0.0] if band is None else [float(band["min"].min()), float(band["max"].max())]
    for j, fm in enumerate(ALL_FMS):
        if fm not in fms:
            continue
        x, mean, lo, hi = [], [], [], []
        for i, m in enumerate(MULTS):
            g = gaps.loc[(fm, slice(None), m)].to_numpy()
            a, b = boot_ci(g)
            x.append(i + (j - 3) * step)
            mean.append(g.mean())
            lo.append(a)
            hi.append(b)
        mean = np.array(mean)
        ax.errorbar(x, mean, yerr=[mean - lo, np.array(hi) - mean], color=FM[fm]["color"], lw=0, elinewidth=1.0,
                    capsize=0, zorder=3, **mark_kw(fm))
        ext = [min(ext[0], min(lo)), max(ext[1], max(hi))]
    budget_axis(ax)
    return ext[0], ext[1]


def fig_gap(dev: pd.DataFrame, t1p: pd.DataFrame) -> None:
    sets = [("DEV", dev, EXP / "out" / "e1_gaps.csv", None),
            ("PMLB/UCI", t1p, EXP / "e1_t1p" / "out" / "e1t1p_gaps.csv", "all")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.05), sharey=True, sharex=True,
                             gridspec_kw={"hspace": 0.22, "wspace": 0.05})
    lo, hi = 0.0, 0.0
    for r, (name, runs, cls, set_) in enumerate(sets):
        fms = [f for f in ALL_FMS if f in set(runs.learner)]
        for c, (metric, mname) in enumerate([("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")]):
            ax = axes[r, c]
            a, b = gap_panel(ax, table_gaps(runs, metric), fms, classical_range(cls, metric, set_))
            lo, hi = min(lo, a), max(hi, b)
            ax.set_title(f"{mname}, {name}", loc="left", pad=2)
        axes[r, 0].set_ylabel("strat. $-$ random")
    # y-range from the data (all intervals and bands), ticks every 0.05
    ax = axes[0, 0]
    ax.set_ylim(np.floor(lo / 0.05) * 0.05 - 0.005, np.ceil(hi / 0.05) * 0.05 + 0.005)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    for a in axes[1]:
        a.set_xlabel("budget $B = mC$", labelpad=1)
    axes[0, 0].annotate("stratified higher", xy=(0.99, 0.97), xycoords="axes fraction", ha="right", va="top",
                        fontsize=7, color=INK2)
    fm_legend(fig, ALL_FMS, [Patch(color=BAND, label="classical learners (range)")], y=0.955, tight=True)
    save(fig, HERE / "fig2_gap")


# ------------------------------------------------------------------ Fig 3 -------------
def friedman(per_table: pd.Series, learners: list[str]) -> tuple[dict, dict]:
    """Mean Friedman rank (1 = best) per protocol and the Nemenyi CD at alpha = 0.05 (Study 7 report)."""
    ranks, cds = {}, {}
    for strat in ["stratified", "random"]:
        wide = per_table.xs(strat, level="strategy").unstack("learner")[learners].dropna()
        ranks[strat] = wide.rank(axis=1, ascending=False).mean()
        cds[strat] = a7.nemenyi_cd(wide.shape[1], len(wide))
    return ranks, cds


def spread(values: dict, gap: float) -> dict:
    """Label positions at least `gap` apart, in the order of the values."""
    pos, last = {}, -np.inf
    for k in sorted(values, key=values.get):
        pos[k] = max(values[k], last + gap)
        last = pos[k]
    shift = (np.mean(list(pos.values())) - np.mean(list(values.values())))
    return {k: v - shift for k, v in pos.items()}


def rank_panel(ax, ranks: dict, cds: dict, learners: list[str], title: str) -> None:
    def base(lr: str) -> str:
        return lr.replace("_bal", "")

    lab = {lr: fm_label(base(lr)) + (" (bal.)" if lr.endswith("_bal") else "") for lr in learners}
    for lr in reversed(learners):  # pre-registered FMs drawn last, on top
        f = base(lr)
        ax.plot([0, 1], [ranks["stratified"][lr], ranks["random"][lr]], color=FM[f]["color"],
                ls=(0, (3, 1.5)) if lr.endswith("_bal") else "-", lw=1.3, zorder=2, **mark_kw(f, 4.2))
    # direct labels on both ends, spread apart and joined to their marker by a thin leader line
    for side, x0, x1, ha in [("stratified", 0, -0.22, "right"), ("random", 1, 1.22, "left")]:
        val = ranks[side][learners].to_dict()
        pos = spread(val, 0.62)
        for lr in learners:
            ax.plot([x0 + (x1 - x0) * 0.25, x1 - (x1 - x0) * 0.1], [val[lr], pos[lr]], color=MUTED, lw=0.5, zorder=1)
            ax.text(x1, pos[lr], lab[lr], ha=ha, va="center", fontsize=7, color=INK)
    # Nemenyi critical difference: ranks further apart than this bar differ at alpha = 0.05
    cd = cds["random"]
    top = 0.15
    ax.plot([0.5, 0.5], [top, top + cd], color=INK, lw=1.2, solid_capstyle="butt")
    for yy in (top, top + cd):
        ax.plot([0.46, 0.54], [yy, yy], color=INK, lw=1.2)
    ax.text(0.57, top + cd / 2, "CD", va="center", ha="left", fontsize=7, color=INK)
    ax.set_xticks([0, 1], ["stratified", "random"])
    ax.set_xlim(-1.3, 2.3)
    ax.set_ylim(len(learners) + 0.9, -0.05)
    ax.set_yticks(range(1, len(learners) + 1))
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")


def diff_panel(ax, runs: pd.DataFrame, title: str) -> None:
    """TabICLv2 minus TabPFN v3.5 in macro-F1 per budget, under each protocol (95% CI over tables)."""
    ds = runs.groupby(["learner", "strategy", "mult", "dataset"]).macro_f1.mean()
    zero_line(ax)
    for k, (strat, mfc) in enumerate([("random", INK), ("stratified", MUTED)]):
        mean, lo, hi = [], [], []
        for m in MULTS:
            g = (ds.loc[("fm2", strat, m)] - ds.loc[("fm5", strat, m)]).dropna().to_numpy()
            a, b = boot_ci(g)
            mean.append(g.mean())
            lo.append(a)
            hi.append(b)
        mean = np.array(mean)
        x = np.arange(4) + (k - 0.5) * 0.16
        ax.errorbar(x, mean, yerr=[mean - lo, np.array(hi) - mean], color=INK, lw=0, elinewidth=1.0,
                    marker="o", ms=MS, mfc=mfc, mec=mfc, mew=1.0, zorder=3, ecolor=mfc)
        ax.text(x[-1] + 0.12, mean[-1], strat, ha="left", va="center", fontsize=7, color=INK2)
    budget_axis(ax)
    ax.set_xlim(-0.5, 4.15)
    ax.set_xlabel("budget $B = mC$")
    ax.set_title(title, loc="left")


def fig_ranks(dev: pd.DataFrame, t1p: pd.DataFrame) -> None:
    bal = load_bal()
    both = pd.concat([dev, bal], ignore_index=True)
    # per table: score averaged over reps and budgets, then ranked (as in the Study 7 report)
    per_table = both.groupby(["learner", "strategy", "mult", "dataset"]).macro_f1.mean() \
        .groupby(["learner", "strategy", "dataset"]).mean()
    pre = ["fm1", "fm1_bal", "fm2", "fm3", "fm5", "fm5_bal"]
    r_pre, cd_pre = friedman(per_table, pre)
    r_all, cd_all = friedman(per_table, ALL_FMS)
    print("fig3 pre-registered: CD = %.2f; ranks" % cd_pre["random"], r_pre["random"].round(2).to_dict())
    print("fig3 seven FMs: CD = %.2f; ranks" % cd_all["random"], r_all["random"].round(2).to_dict())
    fig = plt.figure(figsize=(6.5, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.75, 1], hspace=0.36, wspace=0.12)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rank_panel(ax_a, r_pre, cd_pre, pre, "(a) Pre-registered learners, DEV")
    rank_panel(ax_b, r_all, cd_all, ALL_FMS, "(b) All seven FMs, DEV (exploratory)")
    ax_a.set_ylabel("mean rank in macro-F1 (1 = best)")
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)
    diff_panel(ax_c, dev, "(c) TabICLv2 $-$ TabPFN v3.5, DEV")
    diff_panel(ax_d, t1p, "(d) TabICLv2 $-$ TabPFN v3.5, PMLB/UCI")
    ax_c.set_ylabel("difference in macro-F1")
    plt.setp(ax_d.get_yticklabels(), visible=False)
    save(fig, HERE / "fig3_ranks")


def load_bal() -> pd.DataFrame:
    return _read([f for lr in ["fm1_bal", "fm5_bal"] for f in sorted((RES7 / "b").glob(f"*_{lr}.parquet"))])


# ------------------------------------------------------------------ Fig 4 -------------
def parse_md_table(text: str, header_start: str) -> pd.DataFrame:
    i = text.index(header_start)
    lines = []
    for ln in text[i:].splitlines():
        if ln.startswith("|"):
            lines.append(ln)
        elif lines:
            break
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines if not re.match(r"^\|[\s:|-]+\|$", ln)]
    return pd.DataFrame(rows[1:], columns=rows[0])


def fig_mechanism() -> None:
    # round-2 CRITICAL #1: Study 7 Part (a) fits (pre-registered; exploratory re-analysis), not Study 4
    mt = pd.read_csv(P1 / "exploratory" / "review_reanalysis" / "out" / "mech_tables.csv")
    cov = mt[(mt.set == "dev") & (mt.fm == "fm5")].rename(
        columns={"f1_gap_pred_first": "f1_gap_pred", "f1_gap": "f1_gap_obs"})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.85), gridspec_kw={"wspace": 0.36})
    ax = axes[0]
    lim = [-0.11, 0.49]
    ax.plot(lim, lim, color=INK2, lw=0.8, zorder=1)
    ax.text(0.47, 0.38, "$y=x$", fontsize=7, color=INK2, ha="right", va="top")
    for m, mk, kw in [(1, "o", dict(facecolor=INK, edgecolor="white", lw=0.6, s=20)),
                      (3, "x", dict(color=INK2, lw=0.9, s=18))]:
        t = cov[cov.mult == m]
        ax.scatter(t.f1_gap_pred, t.f1_gap_obs, marker=mk, zorder=3, **kw)
    # direct labels next to each cloud (positions follow the data)
    t1, t3 = cov[cov.mult == 1], cov[cov.mult == 3]
    ax.text(t1.f1_gap_pred.max() + 0.02, t1.f1_gap_obs.min(), "$m=1$", fontsize=7, color=INK, va="bottom")
    ax.text(t3.f1_gap_pred.min(), t3.f1_gap_obs.max() + 0.02, "$m=3$", fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_xlabel("predicted from coverage alone")
    ax.set_ylabel("observed macro-F1 gap")
    ax.set_title("(a) Coverage model", loc="left")
    ax.text(0.03, 0.97, "TabPFN v3.5, DEV study", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK2)
    # dose-response, Study 7 Part (c): paired within table and budget, relative to the balanced context s = 0.5
    frames = [pd.read_parquet(p, columns=["learner", "dataset", "arm", "share", "mult", "feasible", "macro_f1",
                                          "pc_macro_f1"])
              for fm in PREREG for p in sorted((RES7 / "c").glob(f"*_{fm}.parquet"))]
    c = pd.concat(frames, ignore_index=True)
    c = c[c.feasible.astype(bool)]
    cell = c[c.arm.isin(["majority", "minority"])].groupby(["learner", "arm", "dataset", "mult", "share"])[
        ["macro_f1", "pc_macro_f1"]].mean()
    base = cell.xs(0.5, level="share")
    diff = cell.sub(base.reindex(cell.index.droplevel("share")).to_numpy()).dropna()
    paired = diff.groupby(["learner", "arm", "share", "dataset"]).mean()
    lo_y = 0.0
    for ax, arm, ttl in [(axes[1], "majority", "(b) Majority upweighted"),
                         (axes[2], "minority", "(c) Minority upweighted")]:
        zero_line(ax)
        for fm in PREREG:
            t = paired.loc[(fm, arm)]
            s = t.groupby("share").mean()
            band = np.array([boot_ci(t.xs(sh, level="share").macro_f1.to_numpy()) for sh in s.index])
            col = FM[fm]["color"]
            ax.fill_between(s.index, band[:, 0], band[:, 1], color=col, alpha=0.13, lw=0, zorder=1)
            kw_pc = dict(mark_kw(fm, 3.0), mfc="white")  # open marks: corrected curve (all four are pre-registered)
            ax.plot(s.index, s.pc_macro_f1, color=col, ls=(0, (4, 2)), lw=1.1, zorder=2, **kw_pc)
            ax.plot(s.index, s.macro_f1, color=col, lw=1.3, zorder=3, **mark_kw(fm, 3.6))
            lo_y = min(lo_y, band[:, 0].min(), float(s.pc_macro_f1.min()))
        ax.set_xlabel("share $s$ of that class")
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_title(ttl, loc="left")
    for ax in axes[1:]:
        ax.set_ylim(lo_y - 0.01, 0.02)
    axes[1].set_ylabel("macro-F1 change from $s=0.5$")
    handles = fm_handles(PREREG) + [Line2D([], [], color=INK2, lw=1.3, label="raw"),
                                    Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), marker="o", ms=3.0, mfc="white", mec=INK2,
                                           label="after $p/\\hat\\pi$ correction")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.99), borderpad=0.1,
               columnspacing=1.0, handlelength=1.6)
    save(fig, HERE / "fig4_mechanism")


# ------------------------------------------------------------------ E6 ----------------
def fig_accimb(dev: pd.DataFrame, mean_of: list[str] = ALL_FMS) -> None:
    """Accuracy gap (mean over budgets) against pool imbalance TV, one small panel per FM plus the mean.

    The last panel averages the FMs in `mean_of`: all seven by default (exploratory), PREREG for the four.
    """
    tv = pd.read_csv(EXP / "out" / "e10_pools_dev.csv").assign(dataset=lambda d: d.dataset.astype(str)) \
        .set_index("dataset").tv
    g = table_gaps(dev, "accuracy").groupby(["learner", "dataset"]).mean()
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.5), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.38, "wspace": 0.1})
    xs = np.linspace(0, tv.max(), 20)

    def panel(ax, y: pd.Series, title: str, kw: dict, color: str) -> None:
        x = tv.reindex(y.index)
        assert x.notna().all(), "pool TV missing for a DEV table"
        zero_line(ax)
        b, a = np.polyfit(x, y, 1)
        ax.plot(xs, a + b * xs, color=INK2, lw=1.0, zorder=2)
        ax.plot(x, y, lw=0, color=color, zorder=3, **kw)
        ax.set_title(title, loc="left", fontsize=7.5)

    for ax, fm in zip(axes.flat, ALL_FMS):
        panel(ax, g.loc[fm], fm_label(fm) + ("" if FM[fm]["prereg"] else " (expl.)"), mark_kw(fm, 3.8),
              FM[fm]["color"])
    title = "Mean of the four\npre-registered FMs" if list(mean_of) == PREREG else "Mean of all\n" + {7: "seven"}.get(len(mean_of), str(len(mean_of))) + " FMs"
    panel(axes[1, 3], g.loc[list(mean_of)].groupby("dataset").mean(), title,
          dict(marker="o", ms=3.8, mfc=INK, mec=INK, mew=1.0), INK)
    for ax in axes[1]:
        ax.set_xlabel("pool imbalance TV")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy gap")
    axes[0, 0].set_xticks([0, 0.2, 0.4])
    save(fig, EXP / "out" / "e6_accuracy_imbalance")


# ------------------------------------------------------------------ forest (E9, DEV and T1') --
def meta_rows(runs: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-table gaps (mean over budgets) and the REML random-effects summary per FM.

    Same estimator as exploratory/e9_meta_analysis.py and analysis/t1p_heterogeneity.py (imported)."""
    import sys
    sys.path.insert(0, str(P1 / "analysis"))
    import t1p_heterogeneity as th  # noqa: PLC0415

    d = runs.rename(columns={"learner": "fm"})
    eff = th.effects(d, metric)
    res = pd.DataFrame(th.meta(eff, metric))
    return eff[eff.mult == "all"], res[res.mult == "all"].set_index("fm")


def fig_forest(dev: pd.DataFrame, t1p: pd.DataFrame) -> None:
    sets = [("DEV", dev), ("PMLB/UCI", t1p)]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex="col", gridspec_kw={"hspace": 0.3, "wspace": 0.06})
    for r, (name, runs) in enumerate(sets):
        fms = [f for f in ALL_FMS if f in set(runs.learner)]
        k = runs.dataset.nunique()
        for c, (metric, mname) in enumerate([("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")]):
            ax = axes[r, c]
            eff, res = meta_rows(runs, metric)
            zero_line(ax, "x")
            jit = np.random.default_rng(11)
            for i, fm in enumerate(ALL_FMS):
                yy = -i
                if fm not in fms:
                    if c == 0:
                        ax.text(0.5, yy, f"{fm_label(fm)}: not run on PMLB/UCI", transform=ax.get_yaxis_transform(),
                                ha="center", va="center", fontsize=7, color=MUTED)
                    continue
                col = FM[fm]["color"]
                e = eff[eff.fm == fm]
                ax.scatter(e.y, yy + jit.uniform(-0.2, 0.2, len(e)), s=6, color=col, alpha=0.45, lw=0, zorder=2)
                q = res.loc[fm]
                ax.plot([q.pi_lo, q.pi_hi], [yy, yy], color=INK, lw=0.9, zorder=3, solid_capstyle="butt")
                for xe in (q.pi_lo, q.pi_hi):
                    ax.plot([xe, xe], [yy - 0.16, yy + 0.16], color=INK, lw=0.9, zorder=3)
                ax.plot([q.ci_lo, q.ci_hi], [yy, yy], color=INK, lw=3.2, zorder=4, solid_capstyle="butt")
                ax.plot([q.mu], [yy], color=col, lw=0, zorder=5, **mark_kw(fm, 5.2))
            rows = [i for i, f in enumerate(ALL_FMS) if f in fms]
            ax.set_yticks([-i for i in rows], [fm_label(ALL_FMS[i]) for i in rows] if c == 0 else [])
            ax.set_ylim(-len(ALL_FMS) + 0.4, 0.6)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{mname}, {name} ({k} tables)", loc="left")
            if c == 1:
                ax.tick_params(axis="y", length=0)
    for ax in axes[1]:
        ax.set_xlabel("stratified $-$ random, mean over budgets")
    filled, open_ = provenance_handles()
    handles = [Line2D([], [], lw=0, marker="o", ms=3, mfc=MUTED, mec=MUTED, alpha=0.6, label="one table"), filled,
               Line2D([], [], color=INK, lw=3.2, label="pooled mean, 95% CI"), open_,
               Line2D([], [], color=INK, lw=0.9, marker="|", ms=5, mew=0.9, label="95% prediction interval")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.93), handlelength=1.6)
    save(fig, HERE / "fig_forest")


# ------------------------------------------------------------------ Fig 8 (placeholder) -------
def fig_heldout() -> None:
    """Fig. 8: the HELD-OUT replication (Study 13, arm T), laid out like one row of Fig. 2."""
    from simaudit.analysis import study13 as a13  # noqa: PLC0415

    out_dir = SIM / "results" / "study13" / "heldout"
    if not any(out_dir.glob("*_*.parquet")):
        print(f"fig8_heldout: skipped, no HELD-OUT runs in {out_dir} (existing figure kept)")
        return
    runs = a13.load_runs(out_dir)
    runs = runs[(runs.arm == "T") & runs.strategy.isin(["stratified", "random"])]
    classical = ["lr", "knn1", "knn3", "gnb", "lda", "hgb"]
    fms = [f for f in ALL_FMS if f in set(runs.learner)]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.85), sharey=True, gridspec_kw={"wspace": 0.05})
    lo, hi = 0.0, 0.0
    for ax, (metric, mname) in zip(axes, [("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")]):
        cg = table_gaps(runs[runs.learner.isin(classical)], metric)
        band = cg.groupby(["learner", "mult"]).mean().groupby("mult").agg(["min", "max"])
        a, b = gap_panel(ax, table_gaps(runs[runs.learner.isin(fms)], metric), fms, band)
        lo, hi = min(lo, a), max(hi, b)
        ax.set_title(f"{mname}, HELD-OUT", loc="left", pad=2)
        ax.set_xlabel("budget $B = mC$", labelpad=1)
    axes[0].set_ylabel("strat. $-$ random")
    axes[0].set_ylim(np.floor(lo / 0.05) * 0.05 - 0.005, np.ceil(hi / 0.05) * 0.05 + 0.005)
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    fm_legend(fig, fms, [Patch(color=BAND, label="classical learners (range)")], y=1.0, tight=True)
    save(fig, HERE / "fig8_heldout")


if __name__ == "__main__":
    dev, t1p = dev_runs(), t1p_runs()
    print("DEV tables per FM:", dev.groupby("learner").dataset.nunique().to_dict())
    print("T1' tables per FM:", t1p.groupby("learner").dataset.nunique().to_dict())
    fig_protocols()
    fig_gap(dev, t1p)
    fig_ranks(dev, t1p)
    fig_mechanism()
    fig_accimb(dev)
    fig_forest(dev, t1p)
    fig_heldout()
