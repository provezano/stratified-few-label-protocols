"""Correction 4 on T1′ (Study 12 stored runs), EXPLORATORY; plan: PLAN.md (commit R022).

Reuses reports/p1_corrections.py unchanged (CellModel, composition_features, greg_beta,
subsample_cell). New: the T1′ loader, REDRAW+A7, and interval coverage of NAIVE / GREG.
No FM is called.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "reports"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

import p1_corrections as pc  # noqa: E402
from simaudit.analysis.report import bootstrap_ci  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner.study9 import dataset_key  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
S12 = settings.results_dir / "study12"
FMS = pc.FMS
MULTS = pc.MULTS
METRICS = pc.METRICS
KS = (5, 10)
NAMES = dict(fm1="TabPFN v2", fm2="TabICLv2", fm3="Kumo-Tabular", fm5="TabPFN v3.5")


def load_all():
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    pools, strat, redraw, models, small = {}, {}, {}, {}, {}
    for did in (int(d["did"]) for d in pool["datasets"]):
        y = load(did, settings.data_cache_dir).y
        key = dataset_key(did)
        small[key] = len(y) <= 2000
        for m in MULTS:
            models[(key, m)] = pc.CellModel(y, m)
        for fm in FMS:
            fr = pd.read_parquet(S12 / f"{key}_{fm}.parquet",
                                 columns=["strategy", "mult", "labeled", *METRICS])
            parts = {}
            for st in ("random", "stratified", "redraw"):
                sub = fr[fr.strategy == st].reset_index(drop=True)
                feats = pc.composition_features(y, list(sub.labeled))
                parts[st] = pd.concat([sub, feats], axis=1)
            pools[(key, fm)], strat[(key, fm)], redraw[(key, fm)] = (
                parts["random"], parts["stratified"], parts["redraw"])
        print(f"load: {key}", file=sys.stderr, flush=True)
    return pools, strat, redraw, models, small


def interval_cover(cell, model, beta, metric, k, rng) -> dict[str, float]:
    """Share of subsamples whose t-interval (with held-out-truth noise) covers the truth."""
    m = cell[metric].to_numpy()
    z = cell[["missed", "d"]].to_numpy(float)
    ez = np.array([model.e_missed, model.e_d])
    adj = m - (z - ez) @ beta
    n = len(m)
    t = stats.t.ppf(0.975, k - 1)
    hits = {"NAIVE": [], "GREG": []}
    for _ in range(pc.N_SUB):
        pick = rng.choice(n, k, replace=False)
        rest = np.setdiff1d(np.arange(n), pick)
        truth = m[rest].mean()
        scale = np.sqrt(1 / k + 1 / (n - k))
        for name, v in (("NAIVE", m[pick]), ("GREG", adj[pick])):
            half = t * v.std(ddof=1) * scale
            hits[name].append(abs(v.mean() - truth) <= half + 1e-12)
    return {k_: float(np.mean(v)) for k_, v in hits.items()}


def post_strat(pools, models) -> pd.DataFrame:
    rng = np.random.default_rng([settings.base_seed, 22, 4])
    rows = []
    for (key, fm), pool in pools.items():
        for mult in MULTS:
            cell = pool[pool.mult == mult].reset_index(drop=True)
            if len(cell) < 12:
                continue
            model = models[(key, mult)]
            for metric in METRICS:
                beta = pc.greg_beta({d: p for (d, f), p in pools.items() if f == fm},
                                    key, mult, metric)
                s2 = cell[metric].var(ddof=1)
                for k in KS:
                    errs = pc.subsample_cell(cell, model, beta, metric, k, rng)
                    cov = interval_cover(cell, model, beta, metric, k, rng)
                    for name, e in errs.items():
                        e = np.asarray(e)
                        rows.append(dict(dataset=key, fm=fm, mult=mult, metric=metric, k=k,
                                         estimator=name, bias=e.mean(),
                                         mse=np.mean(e**2) - s2 / (len(cell) - k),
                                         sd=e.std(), cover=cov.get(name, np.nan),
                                         binary=model.k == 2))
        print(f"ps: {key} {fm}", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def cov_only(pools, strat, redraw, models) -> pd.DataFrame:
    rows = []
    for (key, fm), pool in pools.items():
        for mult in MULTS:
            model = models[(key, mult)]
            if model.k != 2:
                continue
            cell = pool[pool.mult == mult]
            st = strat[(key, fm)]
            st = st[st.mult == mult]
            rd = redraw[(key, fm)]
            rd = rd[rd.mult == mult]
            for metric in METRICS:
                truth = cell[metric].mean()
                row = dict(dataset=key, fm=fm, mult=mult, metric=metric,
                           strat_bias=st[metric].mean() - truth,
                           strat_a7_bias=(1 - model.p_miss) * st[metric].mean()
                           + model.miss_contrib[metric] - truth)
                if len(rd):
                    row["redraw_a7_bias"] = ((1 - model.p_miss) * rd[metric].mean()
                                             + model.miss_contrib[metric] - truth)
                rows.append(row)
    return pd.DataFrame(rows)


def ci(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    lo, hi = bootstrap_ci(v)
    return float(v.mean()), lo, hi


def summarise(ps: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = []
    for key, g in ps.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        naive = g[g.estimator == "NAIVE"].groupby("dataset").mse.mean().clip(lower=0)
        for name, e in g.groupby("estimator"):
            bias = e.groupby("dataset").bias.mean()
            mse = e.groupby("dataset").mse.mean().clip(lower=0)
            b, lo, hi = ci(bias.to_numpy())
            out.append(dict(zip(keys, key), estimator=name, n=len(bias), bias=b, bias_lo=lo,
                            bias_hi=hi, rmse=np.sqrt(mse.mean()),
                            ratio=np.sqrt(mse.mean() / naive.loc[mse.index].mean()),
                            cover=e.cover.mean()))
    return pd.DataFrame(out)


def fmt_row(r) -> str:
    cov = "" if not np.isfinite(r.cover) else f"{r.cover:.3f}"
    return (f"{r.bias:+.4f} [{r.bias_lo:+.4f}, {r.bias_hi:+.4f}] | {r.rmse:.4f} | "
            f"{r.ratio:.2f} | {cov}")


def tables(ps, co, small) -> list[str]:
    L = ["# Correction 4 on T1′ (Study 12 runs), EXPLORATORY", "",
         "Plan: PLAN.md (commit R022). Bias = estimate − held-out random mean; "
         "CI = bootstrap over tables; ratio = RMSE / RMSE(NAIVE) at equal runs k; "
         "cover = share of 95% intervals containing the held-out truth.", ""]
    for metric in METRICS:
        sub = ps[ps.metric == metric]
        for title, keys, frame in (
            ("by FM and k", ["fm", "k"], sub),
            ("by FM and m, k = 5", ["fm", "mult"], sub[sub.k == 5]),
            ("by FM and m, k = 10", ["fm", "mult"], sub[sub.k == 10]),
            ("by FM and k, tables with n <= 2000 (41; the prereg prose says 39)", ["fm", "k"],
             sub[sub.dataset.map(small)]),
            ("binary only, by FM and k", ["fm", "k"], sub[sub.binary]),
        ):
            s = summarise(frame, keys)
            L += [f"## {metric}: {title}", "",
                  "| " + " | ".join(keys) + " | estimator | n | bias [CI] | RMSE | ratio | cover |",
                  "|" + "---|" * (len(keys) + 6)]
            for r in s.itertuples():
                L.append("| " + " | ".join(str(getattr(r, k_)) for k_ in keys)
                         + f" | {r.estimator} | {r.n} | " + fmt_row(r) + " |")
            L.append("")
    for metric in METRICS:
        L += [f"## Coverage-only correction (binary, 35 tables), {metric}", "",
              "| FM | m | STRAT bias | STRAT+A7 bias | REDRAW+A7 bias |", "|---|---|---|---|---|"]
        for fm in FMS:
            for m in MULTS:
                g = co[(co.fm == fm) & (co.mult == m) & (co.metric == metric)]
                cells = []
                for c in ("strat_bias", "strat_a7_bias", "redraw_a7_bias"):
                    b, lo, hi = ci(g[c].to_numpy())
                    cells.append(f"{b:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                L.append(f"| {NAMES[fm]} | {m} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def macros(ps: pd.DataFrame, co: pd.DataFrame) -> None:
    dig = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six",
                                  "Seven", "Eight", "Nine"]))
    ft = dict(fm1="PFNtwo", fm5="PFNthreefive", fm2="ICL", fm3="Kumo")
    mt = dict(macro_f1="F", accuracy="A")
    et = {"NAIVE": "Naive", "GREG": "Greg", "PS-SEEN": "PsSeen", "PS-A7": "PsAseven",
          "PS-COMP": "PsComp"}

    def name(*p: str) -> str:
        return "".join(dig.get(c, c) for c in "ExCfourTp" + "".join(p)).replace("-", "")

    def sg(x: float, nd: int = 3) -> str:
        return f"{x:+.{nd}f}".replace("-", "$-$").replace("+", "$+$")

    lines = ["% AUTO-GENERATED by paper/p1/exploratory/c4_t1p/run_c4_t1p.py. EXPLORATORY "
             "(Correction 4 on stored Study 12 T1′ runs; not pre-registered). Do not edit."]

    def put(n: str, v: str, src: str) -> None:
        lines.append(f"\\newcommand{{\\{n}}}{{{v}}} % EXPLORATORY; source: {src}")

    for keys, tag in ((["fm", "k"], "K"), (["fm", "mult"], "M")):
        frames = [(ps, "")] if tag == "K" else [(ps[ps.k == 5], "KFive"), (ps[ps.k == 10], "KTen")]
        for frame, ktag in frames:
            for metric in METRICS:
                s = summarise(frame[frame.metric == metric], keys)
                for r in s.itertuples():
                    base = name(ft[r.fm], mt[metric], et[r.estimator], ktag, tag,
                                str(getattr(r, keys[1])))
                    src = f"out/ps_summary.csv ({r.fm}, {metric}, {r.estimator}, {keys[1]}={getattr(r, keys[1])}{', ' + ktag if ktag else ''})"
                    put(base + "Bias", sg(r.bias, 4), src)
                    put(base + "BiasLo", sg(r.bias_lo, 4), src)
                    put(base + "BiasHi", sg(r.bias_hi, 4), src)
                    put(base + "Ratio", f"{r.ratio:.2f}", src)
                    if np.isfinite(r.cover):
                        put(base + "Cover", f"{r.cover:.2f}", src)
    for fm in FMS:
        for m in MULTS:
            for metric in METRICS:
                g = co[(co.fm == fm) & (co.mult == m) & (co.metric == metric)]
                for c, t in (("strat_bias", "Strat"), ("strat_a7_bias", "StratAseven"),
                             ("redraw_a7_bias", "RedrawAseven")):
                    b, lo, hi = ci(g[c].to_numpy())
                    base = name(ft[fm], mt[metric], "CovOnly", t, "M", str(m))
                    src = f"out/covonly.csv ({fm}, {metric}, {c}, m={m})"
                    put(base, sg(b), src)
                    put(base + "Lo", sg(lo), src)
                    put(base + "Hi", sg(hi), src)
    (OUT / "c4_t1p_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    pools, strat, redraw, models, small = load_all()
    ps_path = OUT / "ps.parquet"
    if ps_path.exists():
        ps = pd.read_parquet(ps_path)
    else:
        ps = post_strat(pools, models)
        ps.to_parquet(ps_path, index=False)
    co = cov_only(pools, strat, redraw, models)
    co.to_csv(OUT / "covonly.csv", index=False)
    pd.concat([summarise(ps[ps.metric == mt], ["fm", "k"]).assign(metric=mt, by="k")
               for mt in METRICS]
              + [summarise(ps[(ps.metric == mt) & (ps.k == k)], ["fm", "mult"])
                 .assign(metric=mt, by=f"mult_k{k}") for mt in METRICS for k in KS]
              ).to_csv(OUT / "ps_summary.csv", index=False)
    (OUT / "TABLES.md").write_text("\n".join(tables(ps, co, small)) + "\n")
    macros(ps, co)
    (OUT / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
