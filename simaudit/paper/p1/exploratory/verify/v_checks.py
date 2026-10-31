"""Verification: reproduction of stored metrics, integrity, T3 caps, T7, and the
1C hypergeometric miss check (claim k) incl. a re-derivation of the random sampler.

Run: OMP_NUM_THREADS=1 nice -n 15 uv run python paper/p1/exploratory/verify/v_checks.py
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from simaudit.data.openml_loader import load  # noqa: E402

BASE_SEED = 20261020
lines: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


t1 = pd.read_parquet(OUT / "fits_t1.parquet")
dev = pd.read_parquet(OUT / "fits_dev.parquet")

# ---------------------------------------------------------------- reproduction
say("## Reproduction of stored metrics (own code vs stored)")
for name, df in (("T1'", t1), ("DEV", dev)):
    for a, b in (("my_macro_f1", "st_macro_f1"), ("my_accuracy", "st_accuracy"),
                 ("my_balanced_accuracy", "st_balanced_accuracy"),
                 ("my_log_loss", "st_log_loss"), ("my_auc_seen", "st_auc")):
        diff = (df[a] - df[b]).abs()
        nan_mismatch = int((df[a].isna() != df[b].isna()).sum())
        say(f"{name} {b}: max|diff|={diff.max():.2e} nan-mismatch={nan_mismatch}")
say("T1' integrity: query==panel\\labeled", bool(t1.ok_query.all()),
    "labeled in pool", bool(t1.ok_pool.all()), "seen matches", bool(t1.seen_ok.all()))
say("T1' max |row sum - 1| =", t1.rowsum_maxdev.max(), "; unseen prob mass max =",
    t1.unseen_mass.max())
say("T1' fits:", len(t1), "tables:", t1.dataset.nunique(), "binary tables:",
    t1[t1.C == 2].dataset.nunique())
say("rows per strategy:", t1.groupby("strategy").size().to_dict())
say("single-class (n_seen==1) fits:", int((t1.n_seen == 1).sum()),
    "AUC NaN:", int(t1.st_auc.isna().sum()))

# ---------------------------------------------------------------- T3 caps
say("\n## T3 per-draw caps")
tol = 1e-12
for name, col in (("stored macro_f1", "st_macro_f1"), ("own macro_f1", "my_macro_f1")):
    v = int((t1[col] > t1.mf1_cap + tol).sum())
    say(f"{name} > absorption cap: {v} of {len(t1)}; max excess",
        float((t1[col] - t1.mf1_cap).max()))
# BA <= |S present|/|present| approximated by n_seen/C (looser) ; accuracy <= 1 - s
v_ba = int((t1.st_balanced_accuracy > t1.n_seen / t1.C + tol).sum())
say("BA > n_seen/C:", v_ba)

# ---------------------------------------------------------------- T7
say("\n## T7: stratified minus covered-random macro-F1 at 1C (per table)")
rng = np.random.default_rng(20261028)
for fm in ("fm2", "fm1", "fm3", "fm5"):
    s = t1[(t1.learner == fm) & (t1.m == 1)]
    st = s[s.strategy == "stratified"].groupby("dataset").st_macro_f1.mean()
    rc = s[(s.strategy == "random") & (s.n_seen == s.C)].groupby("dataset").st_macro_f1.mean()
    k = st.index.intersection(rc.index)
    d = (st[k] - rc[k]).to_numpy()
    boots = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = stats.wilcoxon(d).pvalue
    say(f"{fm}: n={len(d)} mean={d.mean():+.4f} [{lo:+.3f},{hi:+.3f}] Wilcoxon p={p:.2f}")

# ---------------------------------------------------------------- claim k
say("\n## Claim k: 1C miss rate of random draws vs exact hypergeometric (pool counts)")


def p_miss(counts, B):
    """1 - P(every class drawn) for a without-replacement draw of B (incl.-excl.)."""
    N = int(sum(counts))
    C = len(counts)
    tot = 0
    from itertools import combinations
    for r in range(0, C + 1):
        for J in combinations(range(C), r):
            nJ = sum(counts[j] for j in J)
            tot += (-1) ** r * comb(N - nJ, B)
    return 1 - tot / comb(N, B)


r1 = t1[(t1.learner == "fm2") & (t1.strategy == "random") & (t1.m == 1)].copy()
rows = []
sampler_ok = 0
fresh = []
for key, g in r1.groupby("dataset"):
    did = int(g.did.iloc[0])
    y = load(did, ROOT / "data" / "cache").y.astype(np.int64)
    n = len(y)
    C = int(y.max()) + 1
    perm = np.random.default_rng([BASE_SEED, did if did >= 0 else 2**40 + did, 13]).permutation(n)
    s = min(500, n // 5)
    pool_pos = np.arange(s, n)
    yp = y[perm]
    pc = np.bincount(yp[s:], minlength=C)
    fc = np.bincount(y, minlength=C)
    pm = p_miss(list(pc), C)
    pm_full = p_miss(list(fc), C)
    # re-derive the stored draws from sel_seed with an independent sampler copy
    for r in g.itertuples():
        order = np.random.default_rng(int(r.sel_seed)).permutation(pool_pos)
        mine = np.sort(perm[order[:C]])
        stored = np.sort(np.asarray(r.labeled_first))
        sampler_ok += int(np.array_equal(mine, stored))
    # fresh-seed simulation through the same sampler (does the sampler match?)
    sim_rng = np.random.default_rng(did % (2**31))
    miss = 0
    NS = 4000
    for _ in range(NS):
        order = np.random.default_rng(int(sim_rng.integers(2**31 - 1))).permutation(pool_pos)
        miss += len(np.unique(yp[order[:C]])) < C
    rows.append(dict(dataset=key, C=C, n_draws=len(g), obs_miss=int((g.n_seen < C).sum()),
                     p_exact=pm, p_full=pm_full, sim_miss=miss / NS, pool=pc.tolist()))
k = pd.DataFrame(rows)
k.to_csv(OUT / "claim_k_per_table.csv", index=False)
say("stored 1C random draws re-derived from sel_seed exactly:", sampler_ok, "of", len(r1))
for lab, sub in (("binary", k[k.C == 2]), ("multiclass", k[k.C > 2]), ("all", k)):
    nd = sub.n_draws.sum()
    obs = sub.obs_miss.sum()
    exp = (sub.p_exact * sub.n_draws).sum()
    var = (sub.p_exact * (1 - sub.p_exact) * sub.n_draws).sum()
    z = (obs - exp) / np.sqrt(var)
    exp_full = (sub.p_full * sub.n_draws).sum()
    simexp = (sub.sim_miss * sub.n_draws).sum()
    say(f"{lab}: draws={nd} observed miss rate={obs / nd:.3f} exact(pool)={exp / nd:.3f} "
        f"exact(full table)={exp_full / nd:.3f} fresh-seed sim={simexp / nd:.3f} z={z:+.2f}")
# exact Poisson-binomial tail for binary
from numpy.polynomial import polynomial as Ppoly
b = k[k.C == 2]
pmf = np.array([1.0])
for _, r in b.iterrows():
    pmf = np.convolve(pmf, stats.binom.pmf(np.arange(r.n_draws + 1), r.n_draws, r.p_exact))
obs = int(b.obs_miss.sum())
say(f"binary exact Poisson-binomial P(X<={obs}) = {pmf[:obs + 1].sum():.4f} (two-sided ~{2 * pmf[:obs + 1].sum():.4f})")
say("per-table residual (obs - expected), binary, sorted:")
b = b.assign(resid=b.obs_miss - b.n_draws * b.p_exact).sort_values("resid")
say(b[["dataset", "pool", "obs_miss", "p_exact", "sim_miss", "resid"]].to_string(index=False))
# Multiple seeds across m? m=1 draws per rep are used for every FM: only 10/table.
(OUT / "v_checks.txt").write_text("\n".join(lines) + "\n")
