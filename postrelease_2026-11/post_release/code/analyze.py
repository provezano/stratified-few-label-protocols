"""Frozen analysis of the fresh-table confirmation (PLAN.txt). Runs once, automatically, on ALL_DONE.
Reads plan/frame.csv, plan/predictions.csv (written before any fit), results/fits/*.parquet. Writes
results/fresh_pertable.csv, results/RESULTS.json, results/RESULTS.md, results/confirm_numbers.tex.
Run from a temp dir: nice -n 15 <simaudit venv python> -I analyze.py"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import json  # noqa: E402
import zlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("CONFIRM_BASE", str(HERE.parent)))
sys.path.insert(0, str(HERE))
import confirm_lib as L  # noqa: E402
import family as F  # noqa: E402

SEED = 20261108
NB = 5000
NMC = 4000
FITS = BASE / "results" / "fits"
OUT = BASE / "results"


def counts_of(row) -> np.ndarray:
    return np.array([int(c) for c in str(row["class_counts"]).split(";")])


def mech_terms(key: str, counts: np.ndarray) -> dict:
    """pred_cov and signal per FM, exactly as evidence_2026-11-06/code/diagnose_nonrep.py (MC 4000, own seed)."""
    rng = np.random.default_rng([SEED, zlib.crc32(key.encode())])
    n, C = int(counts.sum()), len(counts)
    pi = counts / n
    ms = {}
    for m in L.MULTS:
        N = rng.multivariate_hypergeometric(counts, m * C, size=NMC)
        ms[m] = ((N == 0).any(1), (N == 0).sum(1), (N > 0).sum(1) == 1, N)
    out = {}
    for fm in L.FMS:
        d = L.load_fits(FITS, key, fm)
        pc = []
        for m in L.MULTS:
            mfs = d[(d.strategy == "stratified") & (d.mult == m)].macro_f1.mean()
            miss, K, single, N = ms[m]
            pic = pi[N.argmax(1)]
            fmiss = np.where(single, (2 * pic / (1 + pic)) / C, mfs * (C - K) / C)
            pc.append(float((miss * (mfs - fmiss)).mean()))
        out[f"pred_cov_{fm}"] = float(np.mean(pc))
        ba = d[(d.strategy == "random") & (d.mult == 10)].balanced_accuracy.mean()
        out[f"signal_{fm}"] = float((ba - 1 / C) / (1 - 1 / C))
    out["pred_cov_mean"] = float(np.mean([out[f"pred_cov_{fm}"] for fm in L.FMS]))
    out["signal_mean"] = float(np.mean([out[f"signal_{fm}"] for fm in L.FMS]))
    return out


def boot(t: pd.DataFrame, fn) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    v = [fn(t.iloc[rng.integers(0, len(t), len(t))]) for _ in range(NB)]
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def slope(t, x="pred_pre", y="gap_mf_mean"):
    return float(np.polyfit(t[x].to_numpy(float), t[y].to_numpy(float), 1)[0])


def main() -> None:
    frame = pd.read_csv(BASE / "plan" / "frame.csv", dtype={"key": str})
    pred = pd.read_csv(BASE / "plan" / "predictions.csv", dtype={"key": str})
    rows, missing = [], []
    for _, r in frame.iterrows():
        if not all((FITS / f"{r.key}_{fm}.parquet").exists() for fm in L.FMS):
            missing.append(r.key)
            continue
        c = counts_of(r)
        q = L.table_quantities(FITS, r.key, c)
        q.update(mech_terms(r.key, c))
        rows.append(q)
    t = pd.DataFrame(rows).rename(columns={"dataset": "key"}).merge(pred, on="key")
    t.to_csv(OUT / "fresh_pertable.csv", index=False)
    res: dict = dict(n_tables=len(t), missing_complete_case=missing)
    ps, ok, fam = F.family_pvalues(t)
    adj = F.adjusted(ps, fam)
    res["tests"] = {k: dict(p=ps[k], p_adj=adj[k], direction_ok=ok[k], family=fam[k],
                            passed=bool(adj[k] < 0.05 and ok[k])) for k in ps}
    # H-pred calibration (decision CIs)
    t["resid_pre"] = t.gap_mf_mean - t.pred_pre
    res["H-pred"] = dict(
        rho=float(L.spearman_one_sided(t.pred_pre, t.gap_mf_mean)[0]),
        slope=slope(t), slope_ci=boot(t, slope),
        mean_resid=float(t.resid_pre.mean()), mean_resid_ci=boot(t, lambda s: s.resid_pre.mean()),
        pred_mean=float(t.pred_pre.mean()), obs_mean=float(t.gap_mf_mean.mean()))
    h = res["H-pred"]
    calib = (h["slope_ci"][0] <= 1 <= h["slope_ci"][1]) or (h["mean_resid_ci"][0] <= 0 <= h["mean_resid_ci"][1])
    h["calibrated"] = bool(calib)
    h["verdict"] = ("confirmed" if res["tests"]["H-pred"]["passed"] and calib
                    else "weaker: ranks but mis-sized" if res["tests"]["H-pred"]["passed"] else "not confirmed")
    # M2 (paper's model, frozen coefficients; components from the fits)
    t["pred_m2"] = F.predict_m2(t)
    t["resid_m2"] = t.gap_mf_mean - t.pred_m2
    res["M2"] = dict(rho=float(L.spearman_one_sided(t.pred_m2, t.gap_mf_mean)[0]),
                     mean_resid=float(t.resid_m2.mean()), mean_resid_ci=boot(t, lambda s: s.resid_m2.mean()),
                     slope=slope(t, "pred_m2"), slope_ci=boot(t, lambda s: slope(s, "pred_m2")))
    res["pre92"] = dict(rho=float(L.spearman_one_sided(t.pred_pre92, t.gap_mf_mean)[0]),
                        mean_resid=float((t.gap_mf_mean - t.pred_pre92).mean()))
    # H-cov1 descriptives
    b1 = t[[f"bal1_mf_{fm}" for fm in L.FMS]].mean(axis=1)
    res["H-cov1"] = dict(bal1_mean=float(b1.mean()), bal1_ci=boot(t.assign(b1=b1), lambda s: s.b1.mean()),
                         cov1_mean=float(t.cov1_mf_mean.mean()),
                         per_fm={fm: dict(bal1=float(t[f"bal1_mf_{fm}"].mean()),
                                          cov1=float(t[f"cov1_mf_{fm}"].mean())) for fm in L.FMS})
    # per-FM descriptives
    imb = t[t.r <= F.IMBALANCED_R]
    res["n_imbalanced"] = int(len(imb))
    res["per_fm"] = {fm: dict(gap_mf=float(t[f"gap_mf_{fm}"].mean()), gap_mf_ci=boot(t, lambda s, f=fm: s[f"gap_mf_{f}"].mean()),
                              bal_mf=float(t[f"bal_mf_{fm}"].mean()), gap_ba=float(t[f"gap_ba_{fm}"].mean()),
                              gap_acc_imb=float(imb[f"gap_acc_{fm}"].mean()) if len(imb) else None,
                              gap_acc_all=float(t[f"gap_acc_{fm}"].mean()))
                     for fm in L.FMS}
    # H-rank
    res["H-rank"] = dict(flip=int(t.flip_prot.sum()), clear=int(t.clear_prot.sum()),
                         noise_flip=int(t.flip_noise.sum()), noise_clear=int(t.clear_noise.sum()),
                         rate=float(t.flip_prot.sum() / max(1, t.clear_prot.sum())),
                         noise_rate=float(t.flip_noise.sum() / max(1, t.clear_noise.sum())))
    # H-miss reproducibility (decision CI) and the miss share alone (descriptive)
    dq = t.q_stored - t.q_exact
    ci = boot(t.assign(dq=dq), lambda s: s.dq.mean())
    res["H-miss"] = dict(mean_diff=float(dq.mean()), ci=ci, mean_abs=float(dq.abs().mean()),
                         reproducible=bool(-0.03 <= ci[0] and ci[1] <= 0.03),
                         rho_missshare_cov=float(L.spearman_one_sided(t.q_exact, t.cov_mf_mean)[0]),
                         rho_predcov_cov=float(L.spearman_one_sided(t.pred_cov_mean, t.cov_mf_mean)[0]))
    (OUT / "RESULTS.json").write_text(json.dumps(res, indent=2, default=float))
    lines = [f"# Fresh-table confirmation: results (analysed once by code/analyze.py; {len(t)} tables)", ""]
    for k, v in res["tests"].items():
        lines.append(f"- {k} [{v['family']}]: p {v['p']:.4g}, adjusted {v['p_adj']:.4g}, direction "
                     f"{'ok' if v['direction_ok'] else 'WRONG'} -> {'PASS' if v['passed'] else 'fail'}")
    lines += ["", "```", json.dumps({k: res[k] for k in ("H-pred", "M2", "pre92", "H-cov1", "H-rank", "H-miss",
                                                          "per_fm", "n_imbalanced")}, indent=1, default=float), "```"]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
