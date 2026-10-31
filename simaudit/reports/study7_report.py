"""Generate reports/STUDY7_REPORT.md (prereg/STUDY7_PROTOCOL_MECHANISM.md, DEV).

Usage: uv run python reports/study7_report.py
Reads results/study7/{a,b,c}/*.parquet; writes the markdown report and the
critical-difference figures under reports/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from simaudit.analysis import study7 as a7  # noqa: E402
from simaudit.analysis.report import holm  # noqa: E402
from simaudit.settings import settings  # noqa: E402

ROOT = settings.results_dir / "study7"
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "STUDY7_REPORT.md"
FMS = ("fm1", "fm2", "fm3", "fm5")
NAMES = dict(
    fm1="TabPFN v2",
    fm2="TabICL",
    fm3="Kumo",
    fm5="TabPFN v3.5",
    fm1_bal="TabPFN v2 (bal)",
    fm5_bal="TabPFN v3.5 (bal)",
)
SELECTORS = ("kmedoids", "kmeans_central", "typiclust", "lucos")
N_DEV = 20
# Addendum 1: data-integrity sensitivity sets.
LEAK_ID = (694,)
CONFLICTING = (40496, 860, 46336, 1063, 1071, 801)


def load(part: str, root: Path = ROOT) -> pd.DataFrame:
    """Every finished file of a part, without the probability matrices."""
    files = sorted(p for p in (root / part).glob("*.parquet") if "partial" not in p.name)
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        cols = [c for c in pq.read_schema(path).names if c not in ("proba", "labeled")]
        frames.append(pd.read_parquet(path, columns=cols))
    return pd.concat(frames, ignore_index=True)


def per_dataset(frame: pd.DataFrame, metric: str, mults) -> pd.DataFrame:
    """Mean over reps, then over the given mults: (dataset, learner, strategy)."""
    sub = frame[frame.mult.isin(mults)]
    reps = sub.groupby(["dataset", "learner", "strategy", "mult"])[metric].mean()
    return reps.groupby(["dataset", "learner", "strategy"]).mean().reset_index()


def delta(frame, metric, mults, a, b):
    """Per-dataset Δ = a − b; a, b are (learner, strategy) pairs."""
    vals = per_dataset(frame, metric, mults).set_index(["learner", "strategy"]).sort_index()
    left = vals.loc[a].set_index("dataset")[metric]
    right = vals.loc[b].set_index("dataset")[metric]
    return (left - right).dropna()


def complete(values: pd.Series) -> bool:
    return len(values) == N_DEV


def without(frame: pd.DataFrame, drop) -> pd.DataFrame:
    return frame if frame.empty else frame[~frame.dataset.isin(drop)]


# ------------------------------------------------------------------ confirmatory
def jt_strata(dose, fm, arm, metric="macro_f1", drop=None):
    """Strata (dataset, mult) -> groups ordered by share (feasible only)."""
    sub = dose[(dose.learner == fm) & (dose.arm == arm) & dose.feasible]
    if drop is not None:
        sub = sub[sub.dataset != drop]
    strata = []
    for _, g in sub.groupby(["dataset", "mult"]):
        groups = [
            gg[metric].to_numpy(float) for _, gg in sorted(g.groupby("share"))
        ]
        strata.append(groups)
    return strata


def jt_row(dose, fm, arm, metric="macro_f1"):
    res = a7.jt_stratified(jt_strata(dose, fm, arm, metric))
    datasets = sorted(dose[dose.learner == fm].dataset.unique())
    lodo_p = max(
        a7.jt_stratified(jt_strata(dose, fm, arm, metric, drop=d))["p"]
        for d in datasets
    )
    return res, lodo_p, len(datasets)


def confirmatory(a, b, dose, drop=()):
    """The frozen Sec. 4.1 family; ``drop`` gives the Addendum 1 sensitivity sets."""
    global N_DEV
    saved, N_DEV = N_DEV, 20 - len(drop)
    try:
        return _confirmatory(without(a, drop), without(b, drop), without(dose, drop))
    finally:
        N_DEV = saved


def _confirmatory(a, b, dose):
    rows = []
    for fm in FMS:
        if a.empty or fm not in set(a.learner):
            continue
        v = delta(a, "macro_f1", (1, 3, 5, 10), (fm, "stratified"), (fm, "random"))
        if complete(v):
            s = a7.summarize(v)
            rows.append(dict(id=f"(i)-{fm}", contrast=f"stratified − random, {NAMES[fm]}", p=s["p_wilcoxon"], test="Wilcoxon", **s))
    ab = pd.concat([a, b]) if not b.empty else a
    for cid, left, right, strat, test in (
        ("(ii-a)", "fm2", "fm5", "random", "Wilcoxon"),
        ("(ii-b)", "fm2", "fm5_bal", "random", "Wilcoxon"),
        ("(ii-c)", "fm2", "fm5", "stratified", "TOST ±0.01"),
    ):
        if ab.empty or not {left, right} <= set(ab.learner):
            continue
        v = delta(ab, "macro_f1", (3, 5, 10), (left, strat), (right, strat))
        if complete(v):
            s = a7.summarize(v)
            p = s["p_tost"] if test.startswith("TOST") else s["p_wilcoxon"]
            rows.append(dict(id=cid, contrast=f"{NAMES[left]} − {NAMES[right]}, {strat}", p=p, test=test, **s))
    for fm in FMS:
        for arm in ("majority", "minority"):
            if dose.empty or fm not in set(dose.learner):
                continue
            res, lodo_p, n = jt_row(dose, fm, arm)
            if n == N_DEV:
                rows.append(dict(id=f"(iii)-{fm}-{arm}", contrast=f"dose-response, {NAMES[fm]}, {arm} upweighted", p=res["p"], test=f"JT z={res['z']:.2f}, {res['strata']} strata", n=n, lodo_p_max=lodo_p))
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm(table.p.tolist())
    return table


def fmt_conf(table):
    if table.empty:
        return "_No confirmatory contrast has complete data yet._"
    out = []
    for r in table.to_dict("records"):
        if r["id"].startswith("(iii)"):
            out.append(f"| {r['id']} | {r['contrast']} | {r['test']} | — | — | — | — | {r['p']:.2g} | **{r['p_holm']:.2g}** | — | — | LODO max p {r['lodo_p_max']:.2g} |")
        else:
            out.append(
                f"| {r['id']} | {r['contrast']} | {r['test']} | {r['mean']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['median']:+.3f} | {r['win_rate']:.2f} | {r['p_tost']:.2g} | {r['p']:.2g} | **{r['p_holm']:.2g}** | "
                f"{r['p_bayes_pos']:.2f}/{r['p_bayes_rope']:.2f}/{r['p_bayes_neg']:.2f} | [{r['lodo_min']:+.3f}, {r['lodo_max']:+.3f}] | sign stable: {'yes' if r['lodo_sign_stable'] else 'no'}, max p {r['lodo_p_max']:.2g} |"
            )
    head = (
        "| ID | Contrast | Test | Mean Δ [95% CI] | Median | Win-rate | TOST p | p | Holm p | Bayes P(+)/ROPE/(−) | LODO mean range | LODO |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    return head + "\n" + "\n".join(out)


# ------------------------------------------------------------------ dose-response
def dose_tables(dose):
    lines = []
    feas = dose[dose.feasible & dose.arm.isin(["majority", "minority"])]
    for metric in ("macro_f1", "pc_macro_f1", "accuracy", "balanced_accuracy"):
        piv = feas.groupby(["learner", "arm", "share"])[metric].mean().unstack("share")
        lines += [f"#### Mean {metric} by share of the upweighted class (feasible cells, all datasets × mult)", "", piv.round(3).to_markdown(), ""]
    counts = feas.groupby(["arm", "share", "mult"]).dataset.nunique().unstack("share")
    lines += ["#### Datasets with a feasible cell (arm × mult × share)", "", counts.to_markdown(), ""]
    return lines


def drop_and_slope(dose, fm, arm, metric):
    sub = dose[(dose.learner == fm) & (dose.arm == arm) & dose.feasible]
    drops, slopes = [], []
    for _, g in sub.groupby(["dataset", "mult"]):
        m = g.groupby("share")[metric].mean()
        if len(m) >= 2:
            drops.append(m.iloc[0] - m.iloc[-1])
            slopes.append(a7.slope(g.share.to_numpy(float), g[metric].to_numpy(float)))
    by_ds = sub.groupby(["dataset", "mult"]).ngroups
    return np.array(drops), np.nanmedian(slopes) if slopes else np.nan, by_ds


def mechanism(dose):
    rows = []
    for fm in FMS:
        if fm not in set(dose.learner):
            continue
        for arm in ("majority", "minority"):
            raw_drop, raw_slope, _ = drop_and_slope(dose, fm, arm, "macro_f1")
            pc_drop, pc_slope, _ = drop_and_slope(dose, fm, arm, "pc_macro_f1")
            jt_raw = a7.jt_stratified(jt_strata(dose, fm, arm, "macro_f1"))
            jt_pc = a7.jt_stratified(jt_strata(dose, fm, arm, "pc_macro_f1"))
            acc_drop, _, _ = drop_and_slope(dose, fm, arm, "accuracy")
            tost = a7.tost_p(pc_drop) if len(pc_drop) > 1 else np.nan
            beyond = jt_pc["p"] < 0.05 and pc_drop.mean() >= 0.5 * raw_drop.mean() and raw_drop.mean() > 0
            verdict = "beyond prior shift" if beyond else ("prior shift" if tost < 0.05 else "unresolved")
            sub = dose[(dose.learner == fm) & (dose.arm == arm) & dose.feasible]
            full = sub.groupby(["dataset", "share"]).macro_f1.mean().unstack("share")
            full = full.dropna()
            page = a7.page_test(full.to_numpy()) if full.shape[1] == len(LEVELS) else dict(p=np.nan, blocks=0)
            rows.append(dict(
                learner=fm, arm=arm,
                drop=raw_drop.mean(), slope_median=raw_slope, jt_p=jt_raw["p"],
                page_p=page["p"], page_blocks=page["blocks"],
                acc_drop=acc_drop.mean(),
                pc_drop=pc_drop.mean(), pc_slope_median=pc_slope, pc_jt_p=jt_pc["p"],
                pc_tost_p=tost, verdict=verdict,
            ))
    return pd.DataFrame(rows)


LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)


# ------------------------------------------------------------------ exploratory
def c2_table(a, metric):
    rows = []
    for fm in FMS:
        for mult in (1, 3, 5, 10):
            block = []
            for strat in SELECTORS:
                v = delta(a, metric, (mult,), (fm, strat), (fm, "random"))
                if not complete(v):
                    continue
                pos, rope, neg = a7.bayes_signed_rank(v.to_numpy(float))
                block.append(dict(learner=fm, mult=mult, strategy=strat, mean=v.mean(), median=v.median(), win=(v > 0).mean(), p=a7.wilcoxon_p(v.to_numpy(float)), tost=a7.tost_p(v.to_numpy(float)), p_pos=pos, p_rope=rope))
            if block:
                ps = holm([r["p"] for r in block])
                for r, ph in zip(block, ps):
                    r["p_holm"] = ph
                rows += block
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["cell"] = t.apply(lambda r: f"{r['mean']:+.3f} (p={r['p_holm']:.3f}; P+={r['p_pos']:.2f})", axis=1)
    return t.pivot_table(index=["strategy", "learner"], columns="mult", values="cell", aggfunc="first")


def per_budget(a, b, metric):
    rows = []
    ab = pd.concat([a, b]) if not b.empty else a
    for mult in (1, 3, 5, 10):
        for fm in ("fm1", "fm2", "fm3", "fm5", "fm1_bal", "fm5_bal"):
            if fm not in set(ab.learner):
                continue
            v = delta(ab, metric, (mult,), (fm, "stratified"), (fm, "random"))
            if complete(v):
                rows.append(dict(contrast=f"strat − random {fm}", mult=mult, cell=f"{v.mean():+.3f} (p={a7.wilcoxon_p(v.to_numpy(float)):.3f}, TOST p={a7.tost_p(v.to_numpy(float)):.2f})"))
        for left, right in (("fm2", "fm5"), ("fm2", "fm5_bal"), ("fm5_bal", "fm5"), ("fm1_bal", "fm1")):
            for strat in ("random", "stratified"):
                if not {left, right} <= set(ab.learner):
                    continue
                v = delta(ab, metric, (mult,), (left, strat), (right, strat))
                if complete(v):
                    rows.append(dict(contrast=f"{left} − {right}, {strat}", mult=mult, cell=f"{v.mean():+.3f} (p={a7.wilcoxon_p(v.to_numpy(float)):.3f}, TOST p={a7.tost_p(v.to_numpy(float)):.2f})"))
    t = pd.DataFrame(rows)
    return t.pivot_table(index="contrast", columns="mult", values="cell", aggfunc="first") if len(t) else t


def reversal(a, b, metric):
    ab = pd.concat([a, b]) if not b.empty else a
    sub = ab[ab.strategy.isin(["stratified", "random"])]
    means = sub.groupby(["dataset", "mult", "strategy", "learner"], as_index=False)[metric].mean()
    means = means.rename(columns=dict(strategy="protocol"))
    learners = means.groupby("learner").dataset.nunique()
    keep = learners[learners == N_DEV].index
    means = means[means.learner.isin(keep)]
    if means.learner.nunique() < 3:
        return []
    out = a7.rank_reversal(means, metric)
    lines = [f"- learners: {', '.join(sorted(keep))}"]
    for mult, tau in out["tau"].items():
        g = means[means.mult == mult].groupby(["protocol", "learner"])[metric].mean()
        order = {p: " > ".join(g[p].sort_values(ascending=False).index) for p in ("stratified", "random")}
        lines.append(f"- {mult}C: Kendall τ = {tau:.2f}; stratified: {order['stratified']}; random: {order['random']}")
    share = out["flips"] / out["decisive"] if out["decisive"] else float("nan")
    lines.append(f"- decisive pairwise comparisons (|Δ| > 0.01 under both) that flip winner: {out['flips']}/{out['decisive']} = {share:.1%}")
    return lines


def cd_diagram(ranks: pd.Series, cd: float, title: str, path: Path):
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.32 * k))
    ax.set_xlim(1, k)
    ax.set_ylim(0, k + 1)
    ax.invert_xaxis()
    ax.get_yaxis().set_visible(False)
    for spine in ("left", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.xaxis.tick_top()
    names = list(ranks.index)
    vals = ranks.to_numpy()
    for i, (name, r) in enumerate(zip(names, vals)):
        y = k - i
        ax.plot([r, r], [k + 0.5, y], color="0.4", lw=0.8)
        ax.text(r, y - 0.15, f"{NAMES.get(name, name)} ({r:.2f})", ha="center", va="top", fontsize=8)
    level, last = 0, -1
    for i in range(k):
        j = max(jj for jj in range(k) if vals[jj] - vals[i] <= cd)
        if j > i and j > last:  # maximal cliques only
            last = j
            ax.plot([vals[i], vals[j]], [k + 0.25 - 0.18 * level, k + 0.25 - 0.18 * level], color="black", lw=2.5)
            level += 1
    ax.plot([1, 1 + cd], [0.3, 0.3], color="crimson", lw=2)
    ax.text(1 + cd / 2, 0.45, f"CD = {cd:.2f}", ha="center", fontsize=8, color="crimson")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cd_section(a, b):
    FIG.mkdir(exist_ok=True)
    ab = pd.concat([a, b]) if not b.empty else a
    lines = []
    for protocol in ("stratified", "random"):
        m = per_dataset(ab[ab.strategy == protocol], "macro_f1", (1, 3, 5, 10))
        wide = m.pivot_table(index="dataset", columns="learner", values="macro_f1").dropna(axis=1, thresh=N_DEV).dropna()
        if wide.shape[1] < 3 or len(wide) < N_DEV:
            continue
        ranks = a7.avg_ranks(wide)
        cd = a7.nemenyi_cd(wide.shape[1], len(wide))
        name = f"study7_cd_learners_{protocol}.png"
        cd_diagram(ranks, cd, f"Learners under {protocol} sampling (macro-F1, DEV)", FIG / name)
        lines.append(f"- Learners, {protocol}: Friedman p = {a7.friedman_p(wide):.3g}; CD = {cd:.2f}; ranks: " + ", ".join(f"{k} {v:.2f}" for k, v in ranks.items()) + f" (`figures/{name}`)")
    for fm in FMS:
        m = per_dataset(a[a.learner == fm], "macro_f1", (1, 3, 5, 10))
        wide = m.pivot_table(index="dataset", columns="strategy", values="macro_f1").dropna()
        if wide.shape[1] < 3 or len(wide) < N_DEV:
            continue
        ranks = a7.avg_ranks(wide)
        cd = a7.nemenyi_cd(wide.shape[1], len(wide))
        name = f"study7_cd_strategies_{fm}.png"
        cd_diagram(ranks, cd, f"Strategies with {NAMES[fm]} (macro-F1, DEV)", FIG / name)
        lines.append(f"- Strategies, {NAMES[fm]}: Friedman p = {a7.friedman_p(wide):.3g}; CD = {cd:.2f}; ranks: " + ", ".join(f"{k} {v:.2f}" for k, v in ranks.items()) + f" (`figures/{name}`)")
    return lines


def balance_check():
    """Max |fm1_bal − post-hoc(fm1)| over shared selections (semantics check)."""
    out = []
    for plain, bal in (("fm1", "fm1_bal"), ("fm5", "fm5_bal")):
        diffs = []
        for path in sorted((ROOT / "b").glob(f"*_{bal}.parquet")):
            did = path.name.split("_")[0]
            other = ROOT / "a" / f"{did}_{plain}.parquet"
            if not other.exists():
                continue
            cols = ["strategy", "mult", "rep", "labeled", "proba", "n_classes"]
            pb = pd.read_parquet(path, columns=cols)
            pa = pd.read_parquet(other, columns=cols)
            pa = pa[pa.strategy.isin(["random", "stratified"])]
            m = pa.merge(pb, on=["strategy", "mult", "rep"], suffixes=("_a", "_b"))
            for r in m.itertuples():
                C = r.n_classes_a
                labeled = np.asarray(r.labeled_a)
                pa_ = np.asarray(r.proba_a, dtype=float).reshape(-1, C)
                pb_ = np.asarray(r.proba_b, dtype=float).reshape(-1, C)
                diffs.append((did, pa_, pb_, labeled))
        if diffs:
            out.append((plain, bal, diffs))
    return out


def balance_lines():
    from simaudit.data.openml_loader import load as load_data

    lines = []
    for plain, bal, items in balance_check():
        worst = 0.0
        agree = []
        cache = {}
        for did, pa_, pb_, labeled in items:
            if did not in cache:
                cache[did] = load_data(int(did), settings.data_cache_dir).y
            y = cache[did]
            post = a7.prior_correct(pa_, y[labeled])
            worst = max(worst, float(np.abs(post - pb_).max()))
            agree.append(float(np.mean(post.argmax(1) == pb_.argmax(1))))
        lines.append(f"- {bal} vs p/π̂ applied to {plain} (same selections, {len(items)} fits): max |Δp| = {worst:.2g}; argmax agreement {np.mean(agree):.4f}")
    return lines


def row_order(a, perm):
    """S3: permuted-row selections (fm2) vs the original ones and vs random."""
    if perm.empty:
        return ["_Part p not finished._"]
    a2 = a[a.learner == "fm2"]
    both = pd.concat([a2, perm.assign(strategy=perm.strategy + "_perm")])
    lines = []
    for metric in ("macro_f1", "accuracy"):
        rows = []
        for strat in sorted(set(perm.strategy)):
            for mult in (1, 3, 5, 10):
                for ref in (strat, "random"):
                    v = delta(both, metric, (mult,), ("fm2", strat + "_perm"), ("fm2", ref))
                    if len(v) < 20:
                        continue
                    arr = v.to_numpy(float)
                    rows.append(dict(contrast=f"{strat}: permuted − {'original' if ref == strat else 'random'}", mult=mult, cell=f"{arr.mean():+.3f} (p={a7.wilcoxon_p(arr):.3f}, TOST p={a7.tost_p(arr):.2f})"))
        if rows:
            t = pd.DataFrame(rows).pivot_table(index="contrast", columns="mult", values="cell", aggfunc="first")
            lines += [f"#### {metric}", "", t.to_markdown(), ""]
    return lines


def random_balance():
    """Exploratory: in full-coverage random draws, context imbalance vs macro-F1."""
    from scipy.stats import spearmanr

    from simaudit.data.openml_loader import load as load_data

    rows = []
    for path in sorted((ROOT / "a").glob("*.parquet")):
        did, learner = path.stem.split("_", 1)
        frame = pd.read_parquet(path, columns=["strategy", "mult", "labeled", "coverage", "macro_f1"])
        frame = frame[(frame.strategy == "random") & (frame.coverage == 1.0)]
        y = load_data(int(did), settings.data_cache_dir).y
        C = int(y.max()) + 1
        frame = frame.assign(tv=[0.5 * np.abs(np.bincount(y[np.asarray(l)], minlength=C) / len(l) - 1 / C).sum() for l in frame.labeled])
        for mult, g in frame.groupby("mult"):
            if len(g) >= 8 and g.tv.nunique() > 2:
                rows.append(dict(dataset=int(did), learner=learner, mult=mult, rho=spearmanr(g.tv, g.macro_f1)[0], n=len(g)))
    t = pd.DataFrame(rows).dropna()
    if t.empty:
        return []
    summary = t.groupby("learner").rho.agg(["median", "mean", lambda r: (r < 0).mean(), "count"])
    summary.columns = ["median rho", "mean rho", "share rho < 0", "cells"]
    return ["## Exploratory (not pre-registered): context imbalance within random draws", "", "Within each (dataset, mult), over the 20 random draws that cover every class: Spearman ρ between the context's distance from a uniform class distribution (total variation) and macro-F1. Negative ρ = less balanced contexts score lower, as Part (c) predicts.", "", summary.round(3).to_markdown(), ""]


def sensitivity(a, b, c, perm):
    lines = ["## Sensitivity analyses (Addendum 1; no confirmatory weight)", ""]
    for label, drop in (("S1: excluding 694 (ID column leaks the class)", LEAK_ID), ("S2: excluding 40496, 860, 46336, 1063, 1071, 801 (conflicting duplicates)", CONFLICTING)):
        t = confirmatory(a, b, c, drop)
        lines += [f"### {label}", "", "Holm recomputed within the reduced family.", "", fmt_conf(t), ""]
    s4 = {part: load(part, settings.results_dir / "study7_s4") for part in "abc"}
    lines += ["### S4: 694 re-run with its ID column (index 1) dropped", ""]
    if [len(set(s4[k].learner)) if not s4[k].empty else 0 for k in "abc"] == [4, 2, 4]:
        swap = [pd.concat([without(f, LEAK_ID), s4[k]], ignore_index=True) for k, f in (("a", a), ("b", b), ("c", c))]
        lines += ["All contrasts with 694's rows replaced by the column-dropped run (n = 20, own Holm family).", "", fmt_conf(confirmatory(*swap)), ""]
        for name, frame, new in (("a", a, s4["a"]),):
            old = frame[(frame.dataset == 694) & frame.strategy.isin(["random", "stratified"])].groupby(["learner", "strategy"]).macro_f1.mean()
            fresh = new[new.strategy.isin(["random", "stratified"])].groupby(["learner", "strategy"]).macro_f1.mean()
            lines += ["694 macro-F1 (mean over mult and reps), with vs without the ID column:", "", pd.DataFrame(dict(with_id=old, dropped=fresh)).round(3).to_markdown(), ""]
    else:
        lines += ["_S4 not finished; S1 (excluding 694) is the primary sensitivity._", ""]
    lines += ["### S3: row order (fm2; selections on permuted rows, same seeds)", "", *row_order(a, perm)]
    return lines


THREATS = """## Threats to validity
- **Row order.** 11 of the 20 DEV datasets are stored sorted by class (46562, 46875, 45615, 841, 42544, 337, 1063, 782, 40496, 694, 41511; audit in research_notes/Systematic discovery 2026-10/22_row_order_leakage_fix.md §7). Random, stratified and the Part (c) designs draw uniformly, so their distribution does not depend on order; the deterministic selectors (kmedoids, kmeans_central, typiclust, lucos) use index tie-breaks and index-based seeding, hence S3.
- **694 (diggle_table_a2)** has an ID/counter column (encoded index 1) that encodes row order and so leaks the class through the features (1-NN on it recovers the class); FM scores on 694 are inflated (S1, S4).
- **Also sorted by a feature column:** 860, 841, 782, 801 (reported only).
- **Conflicting duplicates** in 40496 (37 groups), 860 (23), 46336 (9), 1063 (6), 1071 (3) and 801 (1): an irreducible error floor and, when a duplicate pair is split between context and complement, a forced error (S2).
- **Clean** by the audit: 827, 931, 46544, 1510, 47155.
- The complement evaluation set differs between selections; the p/π̂ correction for TabICL and Kumo is ours; chance-level datasets (827, 931) are kept; at high s many multiclass Part (c) cells are infeasible.
- The Part p code was added during the run (Addendum 1); it does not change Parts (a)-(c) (tested: unpermuted selections are byte-identical in their inputs)."""


# ------------------------------------------------------------------ main
def status(parts):
    rows = []
    for part, frame in parts.items():
        if frame.empty:
            continue
        for learner, g in frame.groupby("learner"):
            rows.append(dict(part=part, learner=learner, datasets=g.dataset.nunique(), rows=len(g)))
    return pd.DataFrame(rows).to_markdown(index=False) if rows else "none"


def main():
    a, b, c, perm = load("a"), load("b"), load("c"), load("p")
    conf = confirmatory(a, b, c)
    lines = [
        "# Study 7: protocol, mechanism and selectors (DEV) — report",
        "",
        "Pre-registration: prereg/STUDY7_PROTOCOL_MECHANISM.md (frozen before any run). DEV only, 20 datasets. Generated by `reports/study7_report.py` from `results/study7/`.",
        "",
        READING,
        "",
        "## Completed jobs",
        "",
        status(dict(a=a, b=b, c=c)),
        "",
        "## Confirmatory contrasts (one Holm family; macro-F1)",
        "",
        "Per-dataset values (n = 20). (i): mean over mult ∈ {1,3,5,10}; (ii): mean over mult ∈ {3,5,10}. (ii-c)'s p is the TOST p (margin 0.01). Bayes: baycomp signed-rank, ROPE 0.01. LODO: leave-one-dataset-out.",
        "",
        fmt_conf(conf),
        "",
    ]
    if not c.empty:
        mech = mechanism(c)
        lines += [
            "## Part (c): dose-response and the prior-shift alternative",
            "",
            "drop = mean over (dataset, mult) of macro-F1 at s = 0.5 minus at the largest feasible s; slope = median per-(dataset, mult) OLS slope on s; JT = stratified Jonckheere–Terpstra p (one-sided, decline); Page = Page's L on dataset means over the datasets with all five levels feasible; pc_* = after the p/π̂ prior correction; verdict = prereg Sec. 3 decision rule.",
            "",
            mech.round(4).to_markdown(index=False),
            "",
            *dose_tables(c),
        ]
        anchor = c[(c.arm == "anchor") & c.feasible]
        if len(anchor):
            lines += ["#### Multiclass anchor (s = 1/C, balanced), mean macro-F1", "", anchor.groupby(["learner", "mult"]).macro_f1.mean().unstack().round(3).to_markdown(), ""]
    if not b.empty:
        lines += ["## Part (b): TabPFN's balance option", "", *balance_lines(), ""]
    if not a.empty:
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "auc", "log_loss", "seen_macro_f1", "seen_accuracy"):
            t = per_budget(a, b, metric)
            if len(t):
                lines += [f"## Per-budget contrasts, {metric} (Wilcoxon p unadjusted; TOST margin 0.01)", "", t.to_markdown(), ""]
        lines += ["## Rank reversal between protocols", ""]
        for metric in ("macro_f1", "accuracy"):
            lines += [f"### {metric}", "", *reversal(a, b, metric), ""]
        for metric in ("macro_f1", "accuracy"):
            t = c2_table(a, metric)
            if len(t):
                lines += [f"## C2 (exploratory): label-free selectors − random, {metric}", "", "Mean Δ (Holm p within FM × mult; P+ = Bayesian P(Δ > 0.01)).", "", t.to_markdown(), ""]
        lines += random_balance()
        lines += ["## Critical-difference diagrams (Friedman + Nemenyi, α = 0.05)", "", *cd_section(a, b), ""]
    lines += [*sensitivity(a, b, c, perm), THREATS, ""]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text()[:6000])


READING = """## Reading (2026-10-22; DEV only, n = 20; nothing here is HELD-OUT evidence)

**Bottom line.** C1 holds: the stratified protocol is optimistic in macro-F1 for all four FMs. The context-balance mechanism is confirmed causally, and the protocol changes rankings. Only the "tie under stratified" half of the reversal claim is not formally confirmed. C2 (exploratory here) replicates Study 4 with new seeds. None of the data-integrity sensitivities changes a conclusion.

### Confirmatory family (15 contrasts, Holm)
1. **(i) Stratified − random, macro-F1 (mean over 1C to 10C): positive for every FM.**
   - TabPFN v2 +0.059, TabICL +0.040, Kumo +0.048, TabPFN v3.5 +0.082; Holm p ≤ 0.0008.
   - Win-rate 0.90 to 0.95; the sign is stable under leave-one-dataset-out.
   - The gap shrinks with budget: +0.105 to +0.136 at 1C, −0.014 to +0.024 at 10C. At 10C it is significant only for the two TabPFNs (unadjusted).
   - In accuracy no FM × budget difference is significant. The TOST does not establish equivalence either, so the accuracy effect is "not detected", not "absent".
2. **(ii-a) The reversal replicates on new seeds.** TabICL − TabPFN v3.5 under random: +0.046 [0.028, 0.065], Holm p = 0.0008, win-rate 0.75.
3. **(ii-c) The tie under stratified is not established.** The difference is −0.003 [−0.013, +0.007], but the TOST p is 0.11 (Holm 0.11; Bayesian P(ROPE) = 0.74).
   - By the pre-registered rule, the ranking claim is therefore only half supported: the gap appears under random and is undetectable under stratified, but not demonstrably within ±0.01.
4. **(ii-b) With TabPFN v3.5's balance option on, the reversal is unresolved.** The gap shrinks from +0.046 to +0.014 [0.004, 0.025]. Holm p = 0.11 and TOST p = 0.76, so it neither survives nor is removed. About 70% of the reversal is TabPFN v3.5's context-prior sensitivity.
5. **(iii) Dose-response: macro-F1 falls monotonically as the context departs from balance, in both arms, for all four FMs.** All 8 stratified JT tests have Holm p < 1e-70.
   - The JT treats the 10 reps per level as independent draws. The more conservative Page test on dataset means (19 to 20 blocks) gives p ≤ 2e-9 in all 8 cells.
   - Mean drop from s = 0.5 to the largest feasible s:
     - majority arm: 0.09 (TabICL) to 0.20 (TabPFN v3.5);
     - minority arm: 0.14 to 0.29.
   - In the minority arm, adding labels of the minority class beyond balance still lowers macro-F1. More minority labels are not what helps; balance is.
   - Caveat: at a fixed budget, upweighting one class necessarily starves the others. The design tests balance together with the per-class counts it implies, as the stratified-versus-random contrast does.

### Mechanism versus "only a prior shift"
- The balance option was checked from the source and empirically: `balance_probabilities=True` is exactly p(c|x)/π̂_c renormalised (max |Δp| = 1.5e-7 over 4,000 fits).
- It is a no-op under stratified sampling, where the context is balanced. Under random sampling it raises TabPFN v3.5's macro-F1 by +0.020 to +0.042 and does not significantly change accuracy.
- After this prior correction, the decline is still significant for every FM and arm (JT p < 1e-57), so a prior shift is never the whole effect.
  - For the TabPFNs, the correction removes 43 to 79% of the drop. Verdict: "unresolved" in 3 of 4 cells and "beyond prior shift" in TabPFN v2's majority arm.
  - For TabICL and Kumo, it removes nothing in the majority arm (the corrected drop is larger) and 30 to 50% in the minority arm. Verdict: "beyond prior shift" in all four cells.
- **Reading:** TabPFN's imbalance sensitivity is largely a prior effect that a post-hoc correction mostly fixes. TabICL's and Kumo's is mostly not a prior effect, but it is smaller to begin with.
- Exploratory, not pre-registered: within random draws that cover every class, less balanced contexts score lower in macro-F1 (median Spearman ρ −0.21 for TabICL, −0.47 for TabPFN v3.5). This links Part (c) back to the realistic protocol.

### Rankings
- Kendall τ between the learner rankings under the two protocols is −0.07 to 0.50 in macro-F1.
- 28.5% of decisive pairwise comparisons flip their winner (25.0% in accuracy).
- TabPFN v3.5 ranks second or third under stratified sampling and last under random at every budget. Kumo is first under both.

### C2, label-free selectors versus random (exploratory)
- At 3C, all 16 FM × selector pairs beat random in macro-F1 (Holm within FM × budget); at 5C, 14 of 16.
- At 1C and 10C, only LUCoS is consistent. TypiClust turns negative at 10C.
- Accuracy gains are smaller and significant mainly for LUCoS (1C, 3C, 10C).
- On the CD diagrams, LUCoS ranks level with the y-using stratified reference for TabICL and Kumo.

### Data integrity (Addendum 1)
- Every conclusion above holds under:
  - S1: without 694;
  - S2: without the 6 datasets with conflicting duplicates. Here (ii-a) is +0.041, Holm 0.016;
  - S4: 694 without its ID column.
- With its ID column, 694's macro-F1 is inflated by 0.11 to 0.16.
- S3: re-selecting on permuted rows changes no selector significantly (|Δ| ≤ 0.016, all p ≥ 0.099). The selectors still beat random at 3C, so row order does not drive C2. The TOST cannot certify equivalence with 5 reps.
- In S4 the (ii-b) contrast reaches Holm p = 0.043. This is a sensitivity analysis only; the confirmatory reading of (ii-b) stays "unresolved"."""

if __name__ == "__main__":
    main()
