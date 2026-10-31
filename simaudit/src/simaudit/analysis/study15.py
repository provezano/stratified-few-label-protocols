"""Study 15 analysis (prereg/STUDY15_SECOND_FRAME.md): test (i) on the second frame.

Primary family (Sec. 3): stratified − random macro-F1 per FM (fm1, fm2, fm3,
fm5), arm T, mean over reps then over m ∈ {1, 3, 5, 10}; Wilcoxon signed-rank,
two-sided; Holm over the 4 FMs. The per-FM summary, verdict rule and
not-evaluable rule are Study 13's (``analysis.study12.summarize`` /
``verdict``; complete case per contrast; more than 10% of tables lost ⇒ not
evaluable). The frame claim (Sec. 4.2) is Study 13's C1 rule.

Secondary (Sec. 5): (ii-a) and (ii-c) as in Study 13, a Holm family of 2;
SENS-INT and SENS-ID (Study 13 flags) for the primary family; the per-budget
and accuracy gaps (unadjusted). Exploratory (Sec. 6): fm5 on the
TabPFN-3.5-safe tables, every FM on the multiclass tables, and the ratio to
the HELD-OUT estimates (descriptive).
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simaudit.analysis import study12 as a12
from simaudit.analysis import study13 as a13
from simaudit.analysis.report import bootstrap_ci, holm
from simaudit.analysis.study7 import wilcoxon_p

FMS = ("fm1", "fm2", "fm3", "fm5")
MULTS = (1, 3, 5, 10)
RANK_MULTS = (3, 5, 10)
LOST_SHARE = 0.10
ALPHA = 0.05
SECONDARY = (
    ("(ii-a)", ("fm2", "random"), ("fm5", "random"), RANK_MULTS, "wilcoxon"),
    ("(ii-c)", ("fm2", "stratified"), ("fm5", "stratified"), RANK_MULTS, "tost"),
)
TEX_NAME = {"fm1": "One", "fm2": "Two", "fm3": "Three", "fm5": "Five"}


def gaps(runs: pd.DataFrame, fm: str, metric: str = "macro_f1") -> pd.Series:
    """Per-table stratified − random of one FM, arm T, mean over MULTS."""
    means = a12.cell_means(a13.arm(runs, "T"), metric)
    return a12.delta(means, (fm, "stratified"), (fm, "random"), MULTS, metric)


def _verdict(cid: str, row: Mapping[Any, Any], floor: float) -> str:
    if float(row["n"]) < floor:
        return "not evaluable"
    return a12.verdict(cid, dict(row))


def primary(runs: pd.DataFrame, n_tables: int) -> pd.DataFrame:
    """The family of 4 (test (i)) with Holm p and verdicts."""
    rows = [
        dict(id=f"(i)-{fm}", **a12.summarize(gaps(runs, fm), "wilcoxon")) for fm in FMS
    ]
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    floor = (1 - LOST_SHARE) * n_tables
    table["evaluable"] = table["n"].astype(float) >= floor
    table["verdict"] = [
        _verdict(str(r["id"]), r, floor) for r in table.to_dict("records")
    ]
    return table


def frame_claim(table: pd.DataFrame) -> dict[str, Any]:
    """Study 13 C1 rule: >= 3 of 4 replicated and none reversed."""
    verdicts = list(table["verdict"])
    replicated = sum(v == "replicated" for v in verdicts)
    reversed_ = sum(v == "reversed" for v in verdicts)
    ok = replicated >= 3 and reversed_ == 0
    return dict(
        replicates=ok,
        replicated=replicated,
        reversed=reversed_,
        text=f"{'replicates' if ok else 'does not replicate'} "
        f"({replicated}/4 replicated, {reversed_} reversed)",
    )


def secondary(runs: pd.DataFrame, n_tables: int) -> pd.DataFrame:
    """(ii-a) and (ii-c) as in Study 13; their own Holm family of 2."""
    means = a12.cell_means(a13.arm(runs, "T"), "macro_f1")
    rows = [
        dict(
            id=cid,
            test=test,
            **a12.summarize(a12.delta(means, a, b, m, "macro_f1"), test),
        )
        for cid, a, b, m, test in SECONDARY
    ]
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    floor = (1 - LOST_SHARE) * n_tables
    table["verdict"] = [
        _verdict(str(r["id"]), r, floor) for r in table.to_dict("records")
    ]
    return table


def describe(values: pd.Series) -> dict[str, Any]:
    """n, mean, median, 95% bootstrap CI, win rate, unadjusted Wilcoxon p."""
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return dict(n=0)
    low, high = bootstrap_ci(arr)
    return dict(
        n=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        ci_low=low,
        ci_high=high,
        win_rate=float(np.mean(arr > 0)),
        p=wilcoxon_p(arr),
    )


def per_budget(runs: pd.DataFrame, metric: str = "macro_f1") -> dict[str, Any]:
    """Unadjusted per-m gap per FM."""
    means = a12.cell_means(a13.arm(runs, "T"), metric)
    return {
        fm: {
            str(m): describe(
                a12.delta(means, (fm, "stratified"), (fm, "random"), (m,), metric)
            )
            for m in MULTS
        }
        for fm in FMS
    }


def subset(
    runs: pd.DataFrame, keys: Sequence[str], fms: Sequence[str]
) -> dict[str, Any]:
    """Descriptive gaps of ``fms`` on the tables ``keys``."""
    sub = runs[runs.dataset.isin(list(keys))]
    return {fm: describe(gaps(sub, fm)) for fm in fms}


def sensitivities(
    runs: pd.DataFrame, screen: pd.DataFrame, n_tables: int
) -> dict[str, Any]:
    """SENS-INT (drop flagged) and SENS-ID (drop identifier-column tables)."""
    out: dict[str, Any] = {}
    for name, col in (("sens_int", "flagged"), ("sens_id", "flag_id")):
        drop = list(screen[screen[col].astype(str).eq("True")]["key"])
        n_left = n_tables - len(drop)
        table = primary(a13.drop_tables(runs, drop), n_left)
        out[name] = dict(dropped=drop, table=table, claim=frame_claim(table))
    return out


def heldout_ratio(table: pd.DataFrame, heldout: Mapping[str, float]) -> pd.Series:
    """Second-frame mean / HELD-OUT mean per FM (descriptive)."""
    return (
        table["id"].map(heldout).rdiv(table["mean"])
        if heldout
        else table["mean"] * np.nan
    )


def heldout_means(path: Path) -> dict[str, float]:
    """Study 13 HELD-OUT (i) means from the joint results (descriptive only)."""
    if not path.exists():
        return {}
    block = json.loads(path.read_text())["study13"]["confirmatory"]
    frame = pd.DataFrame(block["data"], columns=block["columns"])
    return {str(r["id"]): float(r["mean"]) for _, r in frame.iterrows()}


def analyse(
    runs: pd.DataFrame,
    pool: Mapping[str, Any],
    screen: pd.DataFrame | None = None,
    heldout: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Every pre-declared analysis (no file output)."""
    entries = pool["datasets"]
    n_tables = len(entries)
    table = primary(runs, n_tables)
    table["ratio_to_heldout"] = heldout_ratio(table, heldout or {})
    safe = [d["key"] for d in entries if d.get("tabpfn35_safe")]
    multi = [d["key"] for d in entries if int(d["C"]) >= 3]
    out: dict[str, Any] = dict(
        n_tables=n_tables,
        n_analysed={str(r["id"]): int(r["n"]) for r in table.to_dict("records")},
        primary=table,
        claim=frame_claim(table),
        secondary=secondary(runs, n_tables),
        per_budget=per_budget(runs),
        accuracy={fm: describe(gaps(runs, fm, "accuracy")) for fm in FMS},
        balanced_accuracy={
            fm: describe(gaps(runs, fm, "balanced_accuracy")) for fm in FMS
        },
        exploratory_tabpfn35_safe=dict(tables=safe, **subset(runs, safe, ("fm5",))),
        exploratory_multiclass=dict(tables=multi, **subset(runs, multi, FMS)),
        exploratory_binary=subset(
            runs, [d["key"] for d in entries if int(d["C"]) == 2], FMS
        ),
    )
    if set(a13.CLASSICAL) <= set(runs.learner):
        out["s1_classical"] = a13.classical_gaps(runs)  # Study 13 S-1, Holm over 6
    if screen is not None:
        out.update(sensitivities(runs, screen, n_tables))
    return out


# ---------------------------------------------------------------- output
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="split", index=False))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def fmt_num(x: Any, digits: int = 3) -> str:
    """Signed number as LaTeX math, or n/a."""
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    return f"${float(x):+.{digits}f}$"


def fmt_p(p: Any) -> str:
    """Format a p-value: three decimals, or a power of ten below 0.001."""
    if p is None or not np.isfinite(float(p)):
        return "n/a"
    p = float(p)
    if p >= 0.001:
        return f"${p:.3f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def macros(res: Mapping[str, Any]) -> dict[str, str]:
    r"""``\SF...`` macros for P1 (second frame)."""
    out = {"SFnTables": str(res["n_tables"]), "SFClaim": res["claim"]["text"]}
    for r in res["primary"].to_dict("records"):
        name = TEX_NAME[str(r["id"]).split("-")[1]]
        out[f"SFIFM{name}Mean"] = fmt_num(r["mean"])
        out[f"SFIFM{name}CI"] = f"[{fmt_num(r['ci_low'])}, {fmt_num(r['ci_high'])}]"
        out[f"SFIFM{name}Raw"] = fmt_p(r["p"])
        out[f"SFIFM{name}Holm"] = fmt_p(r["p_holm"])
        out[f"SFIFM{name}Verdict"] = str(r["verdict"])
        out[f"SFIFM{name}N"] = str(int(r["n"]))
        out[f"SFIFM{name}Win"] = f"{100 * float(r['win_rate']):.0f}\\%"
    for r in res["secondary"].to_dict("records"):
        tag = "IIa" if r["id"] == "(ii-a)" else "IIc"
        out[f"SF{tag}Mean"] = fmt_num(r["mean"])
        out[f"SF{tag}Holm"] = fmt_p(r["p_holm"])
        out[f"SF{tag}Verdict"] = str(r["verdict"])
    safe = res["exploratory_tabpfn35_safe"]["fm5"]
    out["SFSafeFiveMean"] = fmt_num(safe.get("mean"))
    out["SFSafeFiveN"] = str(safe.get("n", 0))
    return out


def tex_file(m: Mapping[str, str]) -> str:
    r"""Sorted ``\def`` lines with a header."""
    head = [
        "% Generated by simaudit.analysis.study15; do not edit by hand.",
        "% Study 15 second frame; generated "
        + datetime.now(UTC).isoformat(timespec="seconds"),
    ]
    return "\n".join(head + [f"\\def\\{k}{{{v}}}" for k, v in sorted(m.items())]) + "\n"


def main() -> None:  # pragma: no cover - run once after the run completes
    """``python -m simaudit.analysis.study15``: results/study15/ + P1 macros."""
    import yaml

    from simaudit.settings import PROJECT_ROOT, settings

    out_dir = settings.results_dir / "study15" / "second_frame"
    pool = yaml.safe_load(
        (PROJECT_ROOT / "data" / "pool_second_frame.yaml").read_text()
    )
    runs = a13.load_runs(out_dir)
    screen = pd.read_csv(
        out_dir / "integrity_screen.csv", keep_default_na=False, dtype=str
    )
    held = heldout_means(
        settings.results_dir / "joint" / "heldout" / "joint_results.json"
    )
    res = analyse(runs, pool, screen, held)
    report = settings.results_dir / "study15" / "study15_results.json"
    report.write_text(json.dumps(_jsonable(res), indent=1))
    tex = PROJECT_ROOT / "paper" / "p1" / "study15_numbers.tex"
    tex.write_text(tex_file(macros(res)))
    print(
        res["primary"][
            ["id", "n", "mean", "ci_low", "ci_high", "p", "p_holm", "verdict"]
        ]
    )
    print(res["claim"]["text"])


if __name__ == "__main__":  # pragma: no cover
    main()
