"""The frozen hypothesis family of PLAN.txt (22 p-value tests, one Holm family) and the frozen pre-data transfer model.
Pure functions of the per-table frame built by confirm_lib.table_quantities (+ traits). Used for power and analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
import confirm_lib as L

# Pre-data transfer model, frozen: OLS on the 72 standard tables (20 development + 52 PMLB/UCI) of the FM-mean
# stratified-minus-random macro-F1 gap on [1, qmc_bar, V, binary, log10n]; coefficients written into PLAN.txt.
PRE_TERMS = ("qmc_bar", "V", "binary", "log10n")
PRE_COEF = None  # filled by freeze_model.py into frozen_model.json; loaded below
MARGIN_COV1 = 0.02     # P1 coverage-test equivalence margin (\SSvtMargin)
IMBALANCED_R = 0.6
RANK_FAMILY = "secondary"  # set from the pilot power (power.py) before the freeze     # imbalanced table: C * pi_min <= 0.6 (binary: minority share <= 0.30)


def load_coef() -> dict:
    import json
    from pathlib import Path
    import os
    base = Path(os.environ.get("CONFIRM_BASE", str(Path(__file__).resolve().parents[1])))
    return json.loads((base / "plan" / "frozen_model.json").read_text())


def predict_pre(t: pd.DataFrame, coef: dict | None = None) -> np.ndarray:
    c = coef or load_coef()["pre"]
    return c["const"] + sum(c[k] * t[k].to_numpy(float) for k in PRE_TERMS)


def directional(p: float, est: float, sign: int) -> tuple[float, bool]:
    return p, bool(np.sign(est) == sign)


def predict_m2(t: pd.DataFrame, coef: dict | None = None) -> np.ndarray:
    c = coef or load_coef()["m2"]
    return (c["const"] + c["pred_cov"] * t.pred_cov_mean.to_numpy(float) + c["V"] * t.V.to_numpy(float)
            + c["binary"] * t.binary.to_numpy(float) + c["signal"] * t.signal_mean.to_numpy(float)
            + c["log10n"] * t.log10n.to_numpy(float))


def family_pvalues(t: pd.DataFrame) -> tuple[dict, dict, dict]:
    """(p values, direction ok, family) for every test; families: primary, secondary, exploratory."""
    ps, ok, fam = {}, {}, {}

    def add(name, p, good, f):
        ps[name], ok[name], fam[name] = float(p), bool(good), f

    # PRIMARY (Holm over 2)
    rho, p = L.spearman_one_sided(t.pred_pre, t.gap_mf_mean)
    add("H-pred", p, rho > 0, "primary")
    b1 = t[[f"bal1_mf_{fm}" for fm in L.FMS]].mean(axis=1)  # FM-mean balance part at m = 1
    add("H-cov1", L.tost_p(b1, MARGIN_COV1), True, "primary")
    # SECONDARY (Holm over 13)
    for fm in ("fm1", "fm5"):
        x = t[f"bal_mf_{fm}"]; add(f"H-bal.{fm}", L.wilcoxon_p(x), x.mean() > 0, "secondary")
    x = (t.bal_mf_fm1 + t.bal_mf_fm5) / 2 - (t.bal_mf_fm2 + t.bal_mf_fm3) / 2
    add("H-bal.order", L.wilcoxon_p(x), x.mean() > 0, "secondary")
    imb = t[t.r <= IMBALANCED_R]
    for fm in L.FMS:
        x = t[f"gap_ba_{fm}"]; add(f"H-BA.{fm}", L.wilcoxon_p(x), x.mean() > 0, "secondary")
        x = imb[f"gap_acc_{fm}"]; add(f"H-acc.{fm}", L.wilcoxon_p(x), x.mean() < 0, "secondary")
    rho, p = L.spearman_one_sided(t.pred_cov_mean, t.cov_mf_mean)
    add("H-miss.cost", p, rho > 0, "secondary")
    rho, p = L.spearman_one_sided(predict_m2(t), t.gap_mf_mean)
    add("H-pred.M2", p, rho > 0, "secondary")
    # H-rank: secondary if the pilot gives >= 0.8 power at <= 35 tables (PLAN.txt), else exploratory
    x = t.flip_prot / t.clear_prot.clip(lower=1) - t.flip_noise / t.clear_noise.clip(lower=1)
    add("H-rank", L.wilcoxon_p(x), x.mean() > 0, RANK_FAMILY)
    # EXPLORATORY / descriptive (no confirmatory weight)
    for fm in L.FMS:
        add(f"D-cov1.{fm}", L.tost_p(t[f"bal1_mf_{fm}"], MARGIN_COV1), True, "descriptive")
        x = t[f"gap_mf_{fm}"]; add(f"D-gap.{fm}", L.wilcoxon_p(x), x.mean() > 0, "descriptive")
    return ps, ok, fam


def adjusted(ps: dict, fam: dict) -> dict:
    out = {}
    for f in set(fam.values()):
        if f in ("primary", "secondary"):
            out.update(L.holm({k: v for k, v in ps.items() if fam[k] == f}))
        else:
            out.update({k: v for k, v in ps.items() if fam[k] == f})
    return out
