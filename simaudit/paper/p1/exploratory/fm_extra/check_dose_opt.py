"""Check that the optimized Part (c) learners give the reference outputs (plan: dose_plan.md Sec. 4).

``compare`` (project .venv): matches every fit of ``out/dose_c_refcheck`` (``run_dose_c.py --impl ref``
with ``DOSE_C_HISTORICAL_THREADS=1``, i.e. the adapters and thread settings of the DEV Part (a) runs)
with the same cell of ``out/dose_c_pilot`` (``--impl opt``, new thread settings, a different fit
order and more fits per process). Reports, per FM: max |proba diff|, max |metric diff| over every
raw and ``pc_*`` metric, argmax agreement, and the seconds per fit of both sides. Pass = max |proba
diff| <= 1e-6 and identical macro-F1 / pc macro-F1 (<= 1e-12). Writes ``out/dose_c_opt_check.json``.

``bench FM IMPL DID`` (model venv): wall and CPU seconds per fit and peak RSS for 6 fits of one
table (mult 10, rep 0), after one warm-up fit. Run it with and without ``DOSE_C_HISTORICAL_THREADS=1``
to isolate the thread settings.

Usage (from simaudit/):
    PYTHONPATH=src .venv/bin/python paper/p1/exploratory/fm_extra/check_dose_opt.py compare
    PYTHONPATH=src [DOSE_C_HISTORICAL_THREADS=1] nice -n 15 .venv_fm_extra/bin/python \
      paper/p1/exploratory/fm_extra/check_dose_opt.py bench fm6 ref 46562
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL_PROBA, TOL_METRIC = 1e-6, 1e-12
METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "log_loss", "auc")


def compare() -> None:
    import numpy as np
    import pandas as pd

    ref_dir, opt_dir = HERE / "out" / "dose_c_refcheck", HERE / "out" / "dose_c_pilot"
    report: dict[str, dict] = {}
    for fm in ("fm6", "fm7", "fm8"):
        pairs = []
        for rp in sorted(ref_dir.glob(f"*_{fm}.parquet")):
            op = opt_dir / rp.name
            if op.exists():
                pairs.append((pd.read_parquet(rp), pd.read_parquet(op)))
        if not pairs:
            continue
        dp = dm = 0.0
        n = agree = total = 0
        sec_ref, sec_opt = [], []
        mcols = [p + m for p in ("", "seen_", "pc_", "pc_seen_") for m in METRICS]
        for r, o in pairs:
            r, o = r[r.feasible], o[o.feasible]
            m = r.merge(o, on="cell", suffixes=("_r", "_o"))
            assert len(m) == len(r), "pilot lacks some ref cells"
            for x in m.itertuples():
                pr, po = np.asarray(x.proba_r, float), np.asarray(x.proba_o, float)
                dp = max(dp, float(np.abs(pr - po).max()))
                C = int(x.n_classes_r)
                agree += int((pr.reshape(-1, C).argmax(1) == po.reshape(-1, C).argmax(1)).sum())
                total += len(pr) // C
            for c in mcols:
                a, b = m[c + "_r"].to_numpy(float), m[c + "_o"].to_numpy(float)
                both = ~(np.isnan(a) & np.isnan(b))
                if both.any():
                    dm_c = float(np.nanmax(np.abs(a[both] - b[both]))) if np.isfinite(a[both] - b[both]).any() else 0.0
                    if c in ("macro_f1", "pc_macro_f1"):
                        assert (np.isnan(a) == np.isnan(b)).all()
                        report.setdefault(fm, {})[f"max_diff_{c}"] = max(
                            report.get(fm, {}).get(f"max_diff_{c}", 0.0), dm_c)
                    dm = max(dm, dm_c)
            n += len(m)
            once = m.drop_duplicates("seed_r")  # one row per fit (shared designs repeat it)
            sec_ref += once["seconds_r"].tolist()
            sec_opt += once["seconds_o"].tolist()
        rec = report.setdefault(fm, {})
        rec.update(cells=n, tables=len(pairs), max_abs_proba_diff=dp, max_abs_metric_diff_any=dm,
                   argmax_agreement=agree / total, ref_mean_s_per_fit=float(np.mean(sec_ref)),
                   opt_mean_s_per_fit=float(np.mean(sec_opt)))
        rec["pass"] = bool(dp <= TOL_PROBA and rec["max_diff_macro_f1"] <= TOL_METRIC
                           and rec["max_diff_pc_macro_f1"] <= TOL_METRIC)
    out = HERE / "out" / "dose_c_opt_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def bench(fm: str, impl: str, did: int) -> None:
    sys.path.insert(0, str(HERE))
    import run_dose_c as rc
    from simaudit.data.openml_loader import load

    predict = rc.predictor(fm, impl)
    data = load(did, rc.settings.data_cache_dir)
    d = rc.designs(did, 10, (0,))
    groups = [g for _, g in d[d.feasible].groupby("seed", sort=False)][:7]
    rc.fit_group(predict, data.X, data.y, groups[0])  # warm-up (model load)
    wall = cpu = 0.0
    for g in groups[1:]:
        t, u = time.perf_counter(), os.times()
        rc.fit_group(predict, data.X, data.y, g)
        u2 = os.times()
        wall += time.perf_counter() - t
        cpu += (u2.user - u.user) + (u2.system - u.system)
    k = len(groups) - 1
    print(json.dumps(dict(fm=fm, impl=impl, did=did, historical_threads=rc.HISTORICAL_THREADS,
                          wall_s_per_fit=wall / k, cpu_s_per_fit=cpu / k, cpu_over_wall=cpu / wall,
                          peak_rss_mb=rc.peak_rss_mb())))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare()
    else:
        bench(sys.argv[2], sys.argv[3], int(sys.argv[4]))
