"""Real-TabPFN contamination check for the fm1 arm (EXPLORATORY, post hoc, no fits).

fm1 = TabPFNClassifier.create_default_for_version(ModelVersion.V2) under tabpfn
9.2.0, i.e. tabpfn-v2-classifier-finetuned-zk73skhh.ckpt, the Real-TabPFN
checkpoint (Garg et al. 2025, arXiv:2507.03971): TabPFN v2 with continued
pre-training on 71 real tables (their Appendix A, copied in TRAIN below).

1. Intersects that list with DEV (20), T1' (52) and the HELD-OUT ID list (IDs
   and names from data/pool_i2.yaml only; no HELD-OUT data or outputs are read).
2. Recomputes the pre-registered (i) contrasts (stratified - random macro-F1,
   mean over m in {1,3,5,10}) without the overlapping tables, with the frozen
   report code (reports/study7_report.py, simaudit.analysis.study12), plus the
   three secondary FMs from the stored cell files of fm_extra.

Writes out/overlap.csv, out/sensitivity.csv, out/realtabpfn_numbers.tex.
Run: nice -n 15 uv run python paper/p1/exploratory/realtabpfn_check/realtabpfn_check.py
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from simaudit.analysis import study12 as a12  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "reports"))
import study7_report as r7  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
FMX = PROJECT_ROOT / "paper" / "p1" / "exploratory" / "fm_extra" / "out"
MULTS = (1, 3, 5, 10)

# Real-TabPFN Appendix A (arXiv:2507.03971v1): 37 OpenML tables (ID from the
# paper's link) and 34 Kaggle tables (no OpenML ID).
TRAIN_OPENML = {
    4135: "Amazon_employee_access", 1459: "artificial-characters",
    44234: "Bank_marketing_data_set_UCI", 251: "BNG(breast-w)", 137: "BNG(tic-tac-toe)",
    41434: "Click_prediction_small", 4154: "CreditCardSubset", 40668: "connect_4",
    1471: "eeg-eye-state", 151: "electricity", 846: "elevators",
    43551: "Employee-Turnover-at-TECHCO", 1044: "eye_movements",
    41787: "FOREX_eurpln-hour-High", 1477: "gas-drift-different-concentrations",
    1476: "gas-drift", 23512: "higgs", 821: "house_16H", 843: "house_8L",
    44201: "Intersectional-Bias-Assessment-(Training-Data)",
    43904: "law-school-admission-binary", 40679: "magic", 1120: "MagicTelescope",
    43617: "Medical-Appointment", 41671: "microaggregation2", 901: "fried",
    1046: "mozilla4", 43923: "mushroom", 44226: "NewspaperChurn", 1568: "nursery",
    45067: "okcupid_stem", 32: "pendigits", 4534: "PhishingWebsites", 44122: "pol",
    46676: "WBCAtt", 1461: "Bank Marketing", 43039: "Internet Firewall Data",
}
TRAIN_KAGGLE = (
    "aam_avaliacao_dataset", "Air Traffic Data", "ansible-defects-prediction",
    "AV Healthcare Analytics II", "Candidate Selection", "Cardio Disease",
    "CC_Fraud_Dataset", "Classification - Crop Damages in India (2015-2019)",
    "CSGO Round Winner Classification", "Flower Type Prediction Machine Hack",
    "Horse Racing - Tipster Bets", "How severe the accident could be",
    "HR Analysis Case Study", "HR analysis", "hr-comma-sep",
    "ip-network-traffic-flows-labeled-with-87-apps", "Janatahack cross-sell prediction",
    "JanataHack Machine Learning for Banking", "L&T Vehicle Loan Default Prediction",
    "League of Legends Diamond Games (First 15 Minutes)",
    "Multiple target variable classification - Hackathon", "Online News Popularity",
    "Online Shopper's Intention", "Phishing website Detector", "Phishing websites Data",
    "Pump it Up Data Mining the Water Table",
    "Richter's Predictor Modeling Earthquake Damage", "Server Logs - Suspicious",
    "Sloan Digital Sky Survey DR14", "Sloan Digital Sky Survey DR16",
    "Success of Bank Telemarketing Data", "Term Deposit Prediction Data Set",
    "trajectory-based-ship-classification", "Travel Insurance",
)
assert len(TRAIN_OPENML) + len(TRAIN_KAGGLE) == 71

# Manual source-level judgement (name/ID matching alone misses derived tables).
# "overlap": same task and feature space as a training table.
# "related": same source family, different table (not removed in the primary run).
JUDGED = {
    ("t1p", "tic_tac_toe"): ("overlap", "BNG(tic-tac-toe), OpenML 137: Bayes-net resample of the same 9-square task"),
    ("t1p", "car_evaluation"): ("related", "nursery (OpenML 1568): same DEX hierarchical-model family, different table"),
    ("dev", "wdbc"): ("related", "BNG(breast-w)/WBCAtt: Wisconsin breast-cancer family, different table and features"),
}
FLAGGED = {1071: "mw1", 47155: "waveform", 782: "rabe_266"}  # re-review claims


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def overlap_table() -> pd.DataFrame:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())
    t1p = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    train_names = {norm(n) for n in [*TRAIN_OPENML.values(), *TRAIN_KAGGLE]}
    rows = []
    sets = (
        [("dev", int(d["did"]), d["name"]) for d in pool["dev"]]
        + [("t1p", int(d["did"]), d["name"]) for d in t1p["datasets"]]
        + [("heldout", int(d["did"]), d["name"]) for d in pool["heldout"]]
    )
    for split, did, name in sets:
        status, why = JUDGED.get((split, name), ("none", ""))
        rows.append(dict(
            set=split, did=did, name=name,
            id_match=did in TRAIN_OPENML,
            name_match=norm(name) in train_names,
            judged=status, note=why,
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sensitivity
def dev_fm1(drop: tuple[int, ...]) -> dict:
    """Study 7 (i)-fm1 with the frozen report code; Holm over the reduced family."""
    a, b, c = (r7.load(p) for p in "abc")
    t = r7.confirmatory(a, b, c, drop).set_index("id")
    row = t.loc["(i)-fm1"]
    return dict(n=int(row["n"]), mean=row["mean"], lo=row["ci_low"], hi=row["ci_high"],
                p=row["p_wilcoxon"], p_holm=row["p_holm"],
                other_fm_means={k: t.loc[k, "mean"] for k in t.index if k.startswith("(i)-")})


def t1p_family(drop: tuple[str, ...]) -> pd.DataFrame:
    runs = a12.load_runs(settings.results_dir / "study12")
    runs = runs[~runs.dataset.isin(drop)]
    return a12.confirmatory(a12.cell_means(runs, "macro_f1")).set_index("id")


def extra_gap(cells: pd.DataFrame, fm: str, drop) -> dict:
    sub = cells[(cells.learner == fm) & ~cells.dataset.astype(str).isin([str(d) for d in drop])]
    per = sub.groupby("dataset").gap.mean()
    s = a12.summarize(per)
    return dict(n=s["n"], mean=s["mean"], lo=s["ci_low"], hi=s["ci_high"], p=s["p_wilcoxon"])


def main() -> None:
    ov = overlap_table()
    ov.to_csv(OUT / "overlap.csv", index=False)
    t1p_ids = dict(zip(ov[ov.set == "t1p"].name, ov[ov.set == "t1p"].did))
    ttt = f"pmlb_{abs(t1p_ids['tic_tac_toe'])}"
    car = f"pmlb_{abs(t1p_ids['car_evaluation'])}"
    dev_overlap = tuple(int(d) for d in ov[(ov.set == "dev") & (ov.judged == "overlap")].did)

    variants_dev = {
        "all": (),
        "overlap": dev_overlap,                    # empty: no DEV overlap
        "flagged": tuple(FLAGGED),                 # re-review list, refuted
        "broad": (*dev_overlap, 1510),             # + related wdbc
    }
    variants_t1p = {"all": (), "overlap": (ttt,), "broad": (ttt, car)}

    rows = []
    for v, drop in variants_dev.items():
        r = dev_fm1(drop)
        rows.append(dict(set="dev", variant=v, fm="fm1", source="study7/a",
                         **{k: r[k] for k in ("n", "mean", "lo", "hi", "p", "p_holm")}))
    for v, drop in variants_t1p.items():
        fam = t1p_family(drop)
        for cid in ("(i)-fm1", "(i)-fm2", "(i)-fm3", "(i)-fm5", "(ii-a)", "(ii-c)"):
            r = fam.loc[cid]
            rows.append(dict(set="t1p", variant=v, fm=cid, source="study12",
                             n=int(r["n"]), mean=r["mean"], lo=r["ci_low"], hi=r["ci_high"],
                             p=r["p_wilcoxon"], p_holm=r["p_holm"], verdict=r["verdict"]))
    # three secondary FMs (cell files used by the seven-FM checks)
    dev_cells = {"fm6": FMX / "s7a_cells.csv", "fm7": FMX / "s7a_v2_cells.csv", "fm8": FMX / "s7a_cells.csv"}
    t1p_cells = {"fm6": FMX / "t1p_cells.csv", "fm7": FMX / "t1p_cells.csv", "fm8": FMX / "t1p_fm8_cells.csv"}
    for fm in ("fm6", "fm7", "fm8"):
        dc, tc = pd.read_csv(dev_cells[fm]), pd.read_csv(t1p_cells[fm])
        for v, drop in (("all", ()), ("broad", variants_dev["broad"])):
            rows.append(dict(set="dev", variant=v, fm=fm, source=dev_cells[fm].name, **extra_gap(dc, fm, drop)))
        for v, drop in variants_t1p.items():
            rows.append(dict(set="t1p", variant=v, fm=fm, source=t1p_cells[fm].name, **extra_gap(tc, fm, drop)))
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "sensitivity.csv", index=False)

    # direction check on the one overlapping table: fm1 gap vs the other FMs'
    tc_all = pd.concat([pd.read_csv(FMX / "t1p_cells.csv"), pd.read_csv(FMX / "t1p_fm8_cells.csv")])
    per = tc_all.groupby(["learner", "dataset"]).gap.mean().unstack(0)
    ttt_row = per.loc[ttt]
    rank_fm1 = int(ttt_row.rank(ascending=True)["fm1"])  # 1 = smallest gap
    pct_fm1 = float((per["fm1"] < per.loc[ttt, "fm1"]).mean())
    direction = pd.DataFrame(dict(gap_ttt=ttt_row, gap_mean52=per.mean()))
    direction.to_csv(OUT / "direction_tictactoe.csv")
    print(ov[ov.judged != "none"].to_string())
    print(ov[ov.id_match | ov.name_match].to_string())
    print(sens.round(4).to_string())
    print(direction.round(4).to_string(), "\nfm1 rank on tic_tac_toe (1=smallest gap of 7):", rank_fm1,
          "\nshare of T1' tables with smaller fm1 gap:", round(pct_fm1, 3))

    # ------------------------------------------------------------------ macros
    def fmt(x: float, sign: bool = True) -> str:
        s = f"{x:.3f}"
        return ("$+$" + s if x >= 0 else "$-$" + s[1:]) if sign else s

    def pf(p: float) -> str:
        return f"{p:.2g}" if p >= 0.001 else r"$<$0.001"

    word = {"dev": "Dev", "t1p": "Tp", "all": "All", "overlap": "NoOv", "broad": "Broad", "flagged": "NoFlag",
            "fm1": "FMone", "(i)-fm1": "FMone", "(i)-fm2": "FMtwo", "(i)-fm3": "FMthree", "(i)-fm5": "FMfive",
            "(ii-a)": "IIa", "(ii-c)": "IIc", "fm6": "FMsix", "fm7": "FMseven", "fm8": "FMeight"}
    lines = ["% AUTO-GENERATED by paper/p1/exploratory/realtabpfn_check/realtabpfn_check.py. "
             "EXPLORATORY (post hoc sensitivity: tables overlapping the Real-TabPFN training list removed; no fits). Do not edit."]

    def put(name: str, val: str, src: str) -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        lines.append(f"\\newcommand{{\\ExRtp{name}}}{{{val}}} % EXPLORATORY; source: {src}")

    for _, r in sens.iterrows():
        stem = word[r.set] + word[r.fm] + word[r.variant]
        src = f"out/sensitivity.csv ({r.set}, {r.fm}, {r.variant})"
        put(stem + "Mean", fmt(r["mean"]), src)
        put(stem + "Lo", fmt(r.lo), src)
        put(stem + "Hi", fmt(r.hi), src)
        put(stem + "N", str(int(r.n)), src)
        if pd.notna(r.get("p_holm", np.nan)):
            put(stem + "Holm", pf(r.p_holm), src)
    n_ov = ov[ov.judged == "overlap"].groupby("set").size()
    for s in ("dev", "t1p", "heldout"):
        put(f"NOverlap{ {'dev': 'Dev', 't1p': 'Tp', 'heldout': 'Ho'}[s] }".replace(" ", ""), str(int(n_ov.get(s, 0))), "out/overlap.csv")
    put("NTrain", "71", "Garg et al. 2025, App. A")
    put("NTrainOpenML", str(len(TRAIN_OPENML)), "Garg et al. 2025, App. A")
    put("TttGapFMone", fmt(per.loc[ttt, "fm1"]), "out/direction_tictactoe.csv")
    put("TttGapFMoneRank", str(rank_fm1), "out/direction_tictactoe.csv (1 = smallest of 7 FMs)")
    (OUT / "realtabpfn_numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
