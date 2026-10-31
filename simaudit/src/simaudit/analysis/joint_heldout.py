r"""Joint HELD-OUT analysis driver (prereg/JOINT_OPENING_2026-10-30.md).

One command runs every pre-registered analysis of the joint opening on one
set, then the cross-study Holm of the 19 confirmatory p-values::

    P1_HELDOUT_GO=1 nice -n 15 .venv/bin/python \\
        -m simaudit.analysis.joint_heldout --set heldout

``--set dev`` is the dry run on the DEV outputs (Study 13 has none: ``--skip
study13``). Written and committed 2026-10-30, before any HELD-OUT result
existed; it reads no HELD-OUT output until it is run on ``--set heldout``.

Steps, in order:

1. **Completeness** (opening note Sec. 2). From file names and failure logs
   only, every confirmatory cell must be finished or have definitively failed;
   otherwise the driver stops before reading any outcome.
2. **Study 13** (``analysis.study13``): the confirmatory family of 6 with its
   verdicts, the C1 and ranking readings, S-1 to S-14 and the sensitivities.
3. **Study 9** (pinned ``analysis.study9``, SHA-256 checked first): ``analyze``
   and ``report`` unchanged; then the H2a magnitude CI and the FM1
   robustness (as the T1' report), and the descriptive sensitivities of
   Amendments 2-4, all through the frozen module's functions.
4. **Study 10** (``analysis.study10_families``): H1 on HO ∪ T1' with the
   Sec. 8 decision, then the fm2 secondaries.
5. **Cross-study Holm** (``analysis.study13.cross_study_holm``) over the 19
   p-values in the order of the opening note, and the wording rule.
6. Writes ``joint_report.md`` and ``joint_results.json`` to the output folder
   and the P1 macro file (``paper/p1/heldout_numbers.tex`` on HELD-OUT only).

Readings of the preregs (the literal reading where they are silent):

- J-1 Study 9 data traits on HELD-OUT. ``analysis.study9.analyze`` needs the
  tables (n, d, C, IR and the k-means H3 budgets). On HELD-OUT they are read
  through the joint-opening loader only (``study13.SealedLoader`` with the
  label step, after ``study9_heldout.check_go``, the Study 10 FROZEN check
  and ``verify_manifest``); every access goes to
  ``{out}/access_log.jsonl``. Study 10's rescoring secondaries read the
  labels the same way. Nothing else under ``sealed_heldout/`` is read.
- J-2 A Study 9 family member that the frozen ``analyze`` does not return
  enters the cross-study family with p = 1 (opening note Sec. 3). The
  within-study Holm is the frozen module's own (over the tests it returns).
- J-3 Study 13 Sec. 1: a confirmatory contrast with n < 0.9 x the tables of
  the set is "not evaluable" (never replicated). Its computed p still enters
  both Holm families; a contrast with no table enters with p = 1.
- J-4 "Confirmed on HELD-OUT" requires the claim's verdict under the
  cross-study Holm; "replicated (study-wise), not under the cross-study
  correction" if it holds under the within-study Holm only. For Study 9 a
  test is "supported" when its Holm p < 0.05; for Study 10 a contrast is
  "established" when its Holm p < 0.05 and its bootstrap lower bound > -δ.
- J-5 Study 9 Sec. 9 "replication" = the same Holm decision (supported or
  not) as on EXP' = T1' (reports/STUDY9_T1PRIME_OUTPUT.md, constants below).
- J-6 Study 9 sensitivities (Amendments 2-4) rerun the frozen ``analyze`` on
  the reduced table set, with its Holm within the reduced family. A drop set
  that is empty is reported as equal to the primary analysis.
- J-7 Secondary analyses run guarded: an error is written into the report in
  place of that analysis and does not stop the confirmatory output.
- J-8 Macro file: names exactly as paper/p1/heldout_branches.md Sec. 5.
  ``\\HOOpeningRev``, ``\\HONResumed`` and ``\\HOSThreeBstarAgree`` cannot be
  produced by the pre-registered analysis code and are listed, not defined.
  Lists use the analysis module's model names; numbers are LaTeX math.
"""

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from simaudit.analysis import study9 as a9
from simaudit.analysis import study10_families as s10
from simaudit.analysis import study12 as a12
from simaudit.analysis import study13 as a13
from simaudit.analysis.study10 import MARGIN
from simaudit.data.openml_loader import Dataset, load

PINNED_ANALYSIS9 = "aaf0bae1defa10d80b1db235614fe228ebf0aa26e82499d0d29dd51a3f6c50ba"
SETS = ("dev", "heldout")
STUDIES = ("study13", "study9", "study10")
ALPHA = 0.05
S13_IDS = ("(i)-fm1", "(i)-fm2", "(i)-fm3", "(i)-fm5", "(ii-a)", "(ii-c)")
S9_IDS = (
    "H1b",
    "H2a|kmedoids_nested",
    "H2a|lucos_nested",
    "H2a|stratified",
    "H2b|kmedoids_nested",
    "H2b|lucos_nested",
    "H3b",
    "H4|agree",
    "H4|fm_disagree",
)
S10_IDS = ("H1a", "H1b", "H1c", "H1d")
# Study 9 EXP' (T1') Holm p, reports/STUDY9_T1PRIME_OUTPUT.md "Holm family" (J-5).
S9_EXP_HOLM = {
    "H1b": 1.0,
    "H2a|kmedoids_nested": 1.0,
    "H2a|lucos_nested": 1.0,
    "H2a|stratified": 3.138e-07,
    "H2b|kmedoids_nested": 1.0,
    "H2b|lucos_nested": 1.0,
    "H3b": 3.51e-06,
    "H4|agree": 1.0,
    "H4|fm_disagree": 1.0,
}
S9_LEARNERS = {"dev": ("fm2",), "heldout": ("fm2", "fm1", "fm5", "fm3")}
S13_PRIMARY_GROUPS = ("fm2", "fm5", "fm1", "fm3")
NAMES = {
    **a12.FM_NAMES,
    # display names as in the paper (labels only; added after the single run)
    "fm2": "TabICLv2",
    "fm3": "Kumo-Tabular",
    "fm6": "TabDPT",
    "fm7": "Mitra",
    "fm8": "LimiX",
}
NOT_PRODUCED = ("HOOpeningRev", "HONResumed", "HOSThreeBstarAgree")


# ===================================================================== paths
@dataclass
class Layout:
    """Where the inputs of one set live, and where the outputs go."""

    root: Path
    results: Path
    set_name: str
    out: Path
    tex: Path

    def study_dir(self, study: str) -> Path:
        """``results/{study}/{set}``."""
        return self.results / study / self.set_name

    def study10_dirs(self) -> dict[str, Path]:
        """HO ∪ T1' on HELD-OUT (Study 10 Sec. 3.1); the set itself otherwise."""
        base = self.results / "study10"
        if self.set_name == "heldout":
            return {"heldout": base / "heldout", "t1prime": base / "t1prime"}
        return {self.set_name: base / self.set_name}


def pool_ids(root: Path, set_name: str) -> list[int]:
    """Dataset IDs of a pool_i2 set (``dev`` or ``heldout``)."""
    pool = yaml.safe_load((root / "data" / "pool_i2.yaml").read_text())
    return [int(d["did"]) for d in pool[set_name]]


def t1prime_ids(root: Path) -> list[int]:
    """T1' dataset IDs (data/pool_t1prime.yaml)."""
    pool = yaml.safe_load((root / "data" / "pool_t1prime.yaml").read_text())
    return [int(d["did"]) for d in pool["datasets"]]


def dataset_key(did: int) -> str:
    """The runners' file key (``runner.study9.dataset_key``)."""
    return f"openml_{did}" if did >= 0 else f"pmlb_{-did}"


# ===================================================================== step 1
def _failures(directory: Path) -> list[list[Any]]:
    path = directory / "failures.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["job"] for line in path.read_text().splitlines() if line]


def _skipped13(directory: Path) -> set[tuple[int, str]]:
    path = directory / "provenance_amendment1.json"
    if not path.exists():
        return set()
    jobs = json.loads(path.read_text()).get("skipped_jobs", [])
    return {(int(d), str(g)) for d, g in jobs}


def pending_study13(directory: Path, ids: Sequence[int]) -> list[str]:
    """Study 13 jobs neither written nor failed twice (retry once, Sec. 1)."""
    final = {(int(j[0]), str(j[1])) for j in _failures(directory) if int(j[2]) >= 2}
    skipped = _skipped13(directory)
    from simaudit.runner.study13 import GROUPS

    return [
        f"{did}/{g}"
        for g in GROUPS
        for did in ids
        if (did, g) not in skipped
        and (did, g) not in final
        and not (directory / f"{dataset_key(did)}_{g}.parquet").exists()
    ]


def pending_jobs(
    directory: Path, ids: Sequence[int], learners: Sequence[str]
) -> list[str]:
    """Study 9 / Study 10 jobs neither written nor in ``failures.jsonl``."""
    failed = {(int(j[1]), str(j[2])) for j in _failures(directory)}
    return [
        f"{did}/{lr}"
        for lr in learners
        for did in ids
        if (did, lr) not in failed
        and not (directory / f"{dataset_key(did)}_{lr}.parquet").exists()
    ]


def completeness(layout: Layout, skip: Sequence[str]) -> dict[str, list[str]]:
    """Pending confirmatory jobs per study (file names and failure logs only)."""
    ids = pool_ids(layout.root, layout.set_name)
    out: dict[str, list[str]] = {}
    if "study13" not in skip:
        out["study13"] = pending_study13(layout.study_dir("study13"), ids)
    if "study9" not in skip:
        out["study9"] = pending_jobs(
            layout.study_dir("study9"), ids, S9_LEARNERS[layout.set_name]
        )
    if "study10" not in skip:
        pending: list[str] = []
        for name, directory in layout.study10_dirs().items():
            set_ids = t1prime_ids(layout.root) if name == "t1prime" else ids
            pending += [
                f"{name}:{j}"
                for j in pending_jobs(directory, set_ids, (s10.PRIMARY_LEARNER,))
            ]
        out["study10"] = pending
    return out


# ===================================================================== data
@dataclass
class Tables:
    """Table loader of one set; HELD-OUT only through the joint-opening loader."""

    layout: Layout
    cache_dir: Path
    sealed: Path | None = None
    _loader: Any = field(default=None, repr=False)

    def sealed_loader(self) -> Any:
        """Guards, manifest check, then ``SealedLoader`` with the label step (J-1)."""
        if self._loader is None:
            from simaudit.runner import study10 as r10
            from simaudit.runner import study13 as r13
            from simaudit.runner import study9_heldout as r9h

            root = self.layout.root
            r9h.check_go(root)
            r10.check_frozen(root / r10.PREREG)
            sealed = self.sealed or root.parent / "sealed_heldout"
            pool = r9h.heldout_pool(root)
            log = self.layout.out / "access_log.jsonl"
            r13.verify_manifest(sealed, sorted(pool), r13.AccessLog(log))
            self._loader = r13.SealedLoader(sealed, pool, log, normalise_labels=True)
        return self._loader

    def __call__(self, did: int, set_name: str | None = None) -> Dataset:
        """Table ``did``; ``set_name`` = 't1prime' reads the (unsealed) cache."""
        if (set_name or self.layout.set_name) == "heldout":
            data: Dataset = self.sealed_loader()(did, self.cache_dir)
            return data
        return load(did, self.cache_dir)


# ===================================================================== helpers
def guarded(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a secondary analysis; record an error instead of raising (J-7)."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - secondaries must not stop the report
        return f"ERROR in {getattr(fn, '__name__', fn)}: {exc!r}"


def read_csv(path: Path) -> pd.DataFrame | None:
    """A runner CSV (screen, label log) as strings, or None if absent."""
    return pd.read_csv(path, keep_default_na=False) if path.exists() else None


def positive(study: str, verdict: str) -> bool:
    """Whether a per-test verdict is the claim the test was made for."""
    return (
        verdict
        in {
            "study13": ("replicated", "tie established"),
            "study9": ("supported",),
            "study10": ("established",),
        }[study]
    )


def wording(study: str, within: str, cross: str) -> str:
    """Opening note Sec. 3 / Study 13 Sec. 4.4 wording rule (J-4)."""
    if positive(study, within) and cross == within:
        return "confirmed on HELD-OUT"
    if positive(study, within):
        return "replicated (study-wise), not under the cross-study correction"
    return within


# ===================================================================== Study 13
def _s13_verdict(row: Mapping[str, Any], p_holm: float, floor: float) -> str:
    if float(row["n"]) < floor:
        return "not evaluable"
    return a12.verdict(str(row["id"]), dict(row, p_holm=p_holm))


def study13_confirmatory(runs: pd.DataFrame, n_tables: int) -> pd.DataFrame:
    """The family of 6 with verdicts; the Sec. 1 not-evaluable rule (J-3)."""
    table = a13.confirmatory(runs)
    floor = (1 - s10.LOST_SHARE) * n_tables
    table["evaluable"] = table["n"].astype(float) >= floor
    table["verdict"] = [
        _s13_verdict({str(k): v for k, v in r.items()}, float(r["p_holm"]), floor)
        for r in table.to_dict("records")
    ]
    table["reading"] = a13.attenuation(table)
    return table


def study13_part(layout: Layout) -> dict[str, Any]:
    """Every Study 13 analysis of the set (mirrors ``analysis.study13.analyse``)."""
    directory = layout.study_dir("study13")
    runs = a13.load_runs(directory)
    tables = read_csv(directory / "integrity_screen.csv")
    label_log = read_csv(directory / "label_normalisation.csv")
    if tables is None:
        raise FileNotFoundError(directory / "integrity_screen.csv")
    n_tables = len(pool_ids(layout.root, layout.set_name))
    table = study13_confirmatory(runs, n_tables)
    t_runs = a13.arm(runs, "T")
    fm_runs = t_runs[t_runs.learner.isin([*a13.FMS, "fm1_bal", "fm5_bal"])]
    n_classes = tables.set_index("key")["C"].astype(int)
    out: dict[str, Any] = dict(
        n_tables=int(t_runs.dataset.nunique()),
        confirmatory=table,
        overall=a13.overall(table),
        s1_classical=guarded(a13.classical_gaps, runs),
        s2_auc_all=guarded(a12.auc_all_analysis, fm_runs),
        s3_imbalance=guarded(a13.imbalance, runs, tables),
        s4_fixed_split=guarded(a13.fixed_split, runs),
        s5_split=guarded(a12.split_analysis, fm_runs, n_classes),
        s6_meta=guarded(a13.meta_analysis, runs),
        s7_fm_minus_classical=guarded(a13.fm_minus_classical, runs),
        s8_redraw=guarded(a12.redraw_analysis, fm_runs),
        s9_flips=guarded(a12.flip_rate, fm_runs, (*a13.FMS, "fm1_bal", "fm5_bal")),
        s8_infeasible=_infeasible(runs),
        amendments=guarded(a13.amendment_families, runs, tables),
        screen=tables,
    )
    out.update(_s13_sensitivities(runs, table, tables, label_log))
    return out


def _infeasible(runs: pd.DataFrame) -> int:
    if "infeasible" not in runs:
        return 0
    bad = runs[runs.infeasible.astype(bool) & (runs.learner == "fm2")]
    return int(len(bad[["dataset", "mult", "rep"]].drop_duplicates()))


def _s13_sensitivities(
    runs: pd.DataFrame,
    table: pd.DataFrame,
    screen: pd.DataFrame,
    label_log: pd.DataFrame | None,
) -> dict[str, Any]:
    flagged = list(screen[screen.flagged.astype(str).eq("True")]["key"])
    id_tables = list(screen[screen.flag_id.astype(str).eq("True")]["key"])
    out: dict[str, Any] = dict(flagged=flagged, id_tables=id_tables)
    out["sens_int"] = guarded(a13.confirmatory, a13.drop_tables(runs, flagged))
    out["sens_id"] = guarded(a13.confirmatory, a13.drop_tables(runs, id_tables))
    if isinstance(out["sens_int"], pd.DataFrame):
        out["fragile"] = [
            cid
            for cid, a, b in zip(table.id, table.verdict, out["sens_int"].verdict)
            if a != b
        ]
    if label_log is not None:
        out["labels"] = guarded(a13.label_sensitivity, runs, label_log)
    return out


# ===================================================================== Study 9
def check_pinned(root: Path) -> None:
    """Abort unless analysis/study9.py matches its Study 9 freeze hash."""
    import hashlib

    path = root / "src" / "simaudit" / "analysis" / "study9.py"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != PINNED_ANALYSIS9:
        raise SystemExit(f"{path}: SHA-256 {got} differs from Study 9 freeze note 1")


def study9_family(res: Mapping[str, Any]) -> dict[str, float]:
    """The 9 p-values in the opening-note order; missing ones are p = 1 (J-2)."""
    got = res["family"]["p"]
    return {k: float(got[k]) if k in got else 1.0 for k in S9_IDS}


def study9_extras(raw: pd.DataFrame, traits: pd.DataFrame) -> dict[str, Any]:
    """H2a magnitude CI (2000 resamples) and FM1 robustness, frozen functions."""
    curve = a9.curves(raw)
    rnd = a9._xs(curve, a9.PRIMARY, "random")
    out: dict[str, Any] = {}
    for st in ("kmedoids_nested", "lucos_nested", "stratified"):
        if st not in set(curve.index.get_level_values("strategy")):
            continue
        rho = a9._rho_intervals(a9._xs(curve, a9.PRIMARY, st), rnd)[1]
        lo, hi = a9.boot_ci(rho, a9.turnbull_median_ratio)
        keep = (
            hi >= 2.0 and lo <= 3.2
            if st == "stratified"
            else hi >= 1.6 and lo <= 2.5 and lo > 1.2
        )
        out[f"magnitude|{st}"] = dict(
            median=a9.turnbull_median_ratio(rho), ci=(lo, hi), retained=keep
        )
    if a9.REPLICATION in set(raw.learner):
        out["fm1"] = _fm1_robustness(raw, curve, traits)
    return out


def _fm1_robustness(
    raw: pd.DataFrame, curve: pd.DataFrame, traits: pd.DataFrame
) -> dict[str, float]:
    ps: dict[str, float] = {}
    ivs = a9._intervals(curve, "fm1", "random")
    if len(ivs) > 6:
        cov = traits.loc[ivs.index, ["C", "IR", "n", "d"]].astype(float)
        aft = a9.aft_fit(ivs.tolist(), cov.apply(np.log).add_prefix("log_"))
        ps["H1b"] = float(str(aft.loc["log_C", "p_greater"]))
    for st, row in a9.h2(curve, "fm1").items():
        ps[f"H2a|{st}"] = float(row["p_sign"])
        if "p_fade" in row:
            ps[f"H2b|{st}"] = float(row["p_fade"])
    table = a9.stop_table(raw[raw.learner == "fm1"], raw[raw.learner == "fm2"])
    for signal in ("agree", "fm_disagree"):
        if signal in table and table[signal].notna().any():
            ps[f"H4|{signal}"] = a9.mae_ratio(table, signal)["p"]
    return ps


def _s9_reduced(
    raw: pd.DataFrame, traits: pd.DataFrame, drop: Sequence[str]
) -> dict[str, Any]:
    if not drop:
        return dict(note="no table dropped: equals the primary analysis")
    keep = [k for k in traits.index if k not in set(drop)]
    res = a9.analyze(raw[raw.dataset.isin(keep)], traits.loc[keep])
    return dict(dropped=list(drop), n=len(keep), family=res["family"])


def study9_sensitivities(
    raw: pd.DataFrame, traits: pd.DataFrame, directory: Path
) -> dict[str, Any]:
    """Amendments 2-4: merged / unverified labels, SENS-INT, SENS-ID (J-6)."""
    out: dict[str, Any] = {}
    labels = read_csv(directory / "label_normalisation.csv")
    if labels is not None:
        merged = list(labels[labels.merged.astype(str).eq("True")]["key"])
        unverified = list(labels[labels.status != "verified"]["key"])
        out["n_merged"], out["n_unverified"] = len(merged), len(unverified)
        out["merged"] = guarded(_s9_reduced, raw, traits, merged)
        if len(unverified) <= 0.5 * len(labels):
            out["merged_unverified"] = guarded(
                _s9_reduced, raw, traits, merged + unverified
            )
        else:
            out["merged_unverified"] = (
                f"not reported: {len(unverified)} of {len(labels)} tables "
                "unverified, more than half (Study 9 Amendment 3)"
            )
    screen = read_csv(directory / "integrity_screen.csv")
    if screen is not None:
        flagged = list(screen[screen.flagged.astype(str).eq("True")]["key"])
        id_tables = list(screen[screen.flag_id.astype(str).eq("True")]["key"])
        out["sens_int"] = guarded(_s9_reduced, raw, traits, flagged)
        out["sens_id"] = guarded(_s9_reduced, raw, traits, id_tables)
    return out


def replication_vs_exp(within: Mapping[str, float]) -> pd.DataFrame:
    """Study 9 Sec. 9: the same Holm decision as on EXP' = T1' (J-5)."""
    rows = [
        dict(
            test=k,
            exp=s9_word(S9_EXP_HOLM[k]),
            heldout=s9_word(within[k]),
            replicated=s9_word(S9_EXP_HOLM[k]) == s9_word(within[k]),
        )
        for k in S9_IDS
    ]
    return pd.DataFrame(rows)


def study9_part(layout: Layout, tables: Tables) -> dict[str, Any]:
    """The frozen Study 9 analysis on the set (FM2 primary), plus J-2 to J-6."""
    check_pinned(layout.root)
    directory = layout.study_dir("study9")
    raw = a9.load_results(directory)
    done = set(raw.dataset)
    keys = {
        dataset_key(d): d
        for d in pool_ids(layout.root, layout.set_name)
        if dataset_key(d) in done
    }
    from simaudit.settings import settings

    traits = a9.dataset_traits(keys, lambda did: tables(did), settings.base_seed)
    raw = raw[raw.dataset.isin(keys)]
    res = a9.analyze(raw, traits)
    family = study9_family(res)
    within = {k: float(res["family"]["holm"].get(k, 1.0)) for k in S9_IDS}
    title = f"Study 9 analysis, set {layout.set_name} ({len(keys)} datasets)"
    return dict(
        n=len(keys),
        result=res,
        family=family,
        within_holm=within,
        replication=replication_vs_exp(within),
        frozen_report=a9.report(res, title),
        traits=traits,
        extras=guarded(study9_extras, raw, traits),
        sensitivities=guarded(study9_sensitivities, raw, traits, directory),
    )


# ===================================================================== Study 10
def _labels(
    keys: Sequence[str],
    dids: Mapping[str, int],
    set_of: Mapping[str, str],
    tables: Tables,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Permuted labels per dataset (runner order) and the Sec. 7.4 traits."""
    from simaudit.runner import study10 as r10

    labels, traits = {}, []
    for key in keys:
        data = tables(dids[key], set_of[key])
        y = np.asarray(data.y, dtype=np.int64)
        labels[key] = y[r10.row_permutation(len(y), dids[key])]
        counts = np.bincount(y)
        traits.append(
            dict(
                dataset=key,
                n=len(y),
                d=int(data.X.shape[1]),
                C=len(counts),
                IR=float(counts.max()) / max(float(counts.min()), 1.0),
                sortedness=s10.class_sortedness(y),
            )
        )
    return labels, pd.DataFrame(traits).set_index("dataset")


def _stream(
    dirs: Mapping[str, Path], labels: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rescore every job file once: common sets, coverage cells, ring cells."""
    rescored, cover, ring = [], [], []
    pairs = s10.common_pairs()
    for directory in dirs.values():
        for path in s10.result_files(directory):
            frame = pd.read_parquet(path)
            key = str(frame.dataset.iloc[0])
            if key not in labels:
                continue
            rescored.append(s10.rescore_dataset(frame, labels[key], pairs))
            if str(frame.learner.iloc[0]) == s10.PRIMARY_LEARNER:
                cover.append(s10.coverage_cells(frame, labels[key]))
                ring.append(s10.ring_cases(frame, labels[key]))
    return _cat(rescored), _cat(cover), _cat(ring)


def _cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def study10_part(
    layout: Layout, tables: Tables, with_labels: bool = True
) -> dict[str, Any]:
    """H1 on the primary set with the Sec. 8 decision, and the fm2 secondaries."""
    dirs = layout.study10_dirs()
    runs = s10.load_runs(dirs)
    set_of = dict(zip(runs.dataset, runs["set"]))
    ids = pool_ids(layout.root, layout.set_name)
    every = ids + (t1prime_ids(layout.root) if layout.set_name == "heldout" else [])
    dids = {dataset_key(d): d for d in every}
    data = s10.Study10Data(runs, n_primary=len(every))
    label_log = read_csv(
        dirs.get("heldout", Path("/nonexistent")) / "label_normalisation.csv"
    )
    extras: dict[str, Any] = {}
    rescored = None
    if with_labels:
        keys = sorted(set(runs.dataset))
        labels, traits = _labels(keys, dids, set_of, tables)
        landmark = runs[(runs.arm == s10.F1) & runs.landmark_active.astype(bool)]
        traits["landmark"] = traits.index.isin(set(landmark.dataset)).astype(float)
        rescored, cover, ring = _stream(dirs, labels)
        extras = dict(
            traits=traits,
            coverage_decomposition=guarded(s10.coverage_decomposition, cover),
            ring_cases=ring,
            own_complement_check=_own_check(runs, rescored),
        )
    out = s10.analyse(data, label_log, rescored, extras)
    out["n_primary"] = len(every)
    return out


def _own_check(runs: pd.DataFrame, rescored: pd.DataFrame) -> float:
    """Largest |stored − rescored| own-complement metric (sanity, should be 0)."""
    if rescored.empty:
        return float("nan")
    stored = runs.set_index(["learner", "dataset", "arm", "mult", "rep"])
    idx = pd.MultiIndex.from_frame(rescored[["learner", "dataset", "a", "mult", "rep"]])
    worst = 0.0
    for metric in s10.METRICS:
        ref = stored[metric].reindex(idx).to_numpy(float)
        got = rescored[f"own_{metric}_a"].to_numpy(float)
        worst = max(worst, float(np.nanmax(np.abs(ref - got))))
    return worst


# ===================================================================== joint
def joint_family(parts: Mapping[str, Any]) -> pd.DataFrame:
    """The 19 p-values in the opening-note order, both Holm levels, wording."""
    rows: list[dict[str, Any]] = []
    families: dict[str, list[float]] = {}
    if "study13" in parts:
        conf = parts["study13"]["confirmatory"].set_index("id")
        families["study13"] = [_p_or_one(conf.loc[c]) for c in S13_IDS]
    if "study9" in parts:
        families["study9"] = [parts["study9"]["family"][k] for k in S9_IDS]
    if "study10" in parts:
        h1 = parts["study10"]["h1"]
        families["study10"] = [_p_or_one(h1.loc[c]) for c in S10_IDS]
    cross = a13.cross_study_holm(families)
    for study, ps in families.items():
        for i, (test, p) in enumerate(zip(_ids(study), ps)):
            within, verdicts = _within(study, test, parts[study], cross[study][i])
            rows.append(
                dict(
                    study=study,
                    test=test,
                    p=p,
                    p_holm_within=within,
                    p_holm_cross=cross[study][i],
                    verdict_within=verdicts[0],
                    verdict_cross=verdicts[1],
                    wording=wording(study, *verdicts),
                )
            )
    return pd.DataFrame(rows)


def _ids(study: str) -> tuple[str, ...]:
    return {"study13": S13_IDS, "study9": S9_IDS, "study10": S10_IDS}[study]


def _p_or_one(row: pd.Series) -> float:
    return 1.0 if int(float(row["n"])) == 0 or pd.isna(row["p"]) else float(row["p"])


def _within(
    study: str, test: str, part: Mapping[str, Any], p_cross: float
) -> tuple[float, tuple[str, str]]:
    if study == "study13":
        conf = part["confirmatory"].set_index("id")
        row = dict(conf.loc[test], id=test)
        floor = (1 - s10.LOST_SHARE) * len_or(part)
        within = float(row["p_holm"])
        return within, (
            _s13_verdict(row, within, floor),
            _s13_verdict(row, p_cross, floor),
        )
    if study == "study9":
        within = float(part["within_holm"][test])
        return within, (s9_word(within), s9_word(p_cross))
    row = part["h1"].loc[test]
    within = float(row["p_holm"])
    ok = bool(row["evaluable"]) and float(row["lower"]) > -MARGIN
    return within, (s10_word(ok, within), s10_word(ok, p_cross))


def s9_word(p_holm: float) -> str:
    """Study 9 test decision (Sec. 6): Holm p < 0.05."""
    return "supported" if p_holm < ALPHA else "not supported"


def s10_word(ok: bool, p_holm: float) -> str:
    """Study 10 NI decision (Sec. 7): Holm p < 0.05 and lower bound > -δ."""
    return "established" if ok and p_holm < ALPHA else "not established"


def len_or(part: Mapping[str, Any]) -> int:
    """Tables of the set for the Study 13 not-evaluable rule."""
    return int(part.get("n_expected", part.get("n_tables", 0)))


def c1_cross(joint: pd.DataFrame) -> dict[str, Any]:
    """C1 read under each Holm level (count rule of the wording, J-4)."""
    s13 = joint[(joint.study == "study13") & joint.test.str.startswith("(i)")]
    out = {}
    for level in ("within", "cross"):
        verdicts = s13[f"verdict_{level}"]
        rep, rev = int((verdicts == "replicated").sum()), int(
            (verdicts == "reversed").sum()
        )
        out[level] = dict(
            replicated=rep,
            reversed=rev,
            c1="replicates" if rep >= 3 and rev == 0 else "does not replicate",
        )
    return out


# ===================================================================== macros
def fmt_num(x: Any, digits: int = 3, sign: bool = True) -> str:
    """A number as LaTeX math (signed by default)."""
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    spec = f"{'+' if sign else ''}.{digits}f"
    return f"${float(x):{spec}}$".replace("-", "-")


def fmt_p(p: Any) -> str:
    """A p-value: three decimals, or a power of ten below 0.001."""
    if p is None or not np.isfinite(float(p)):
        return "n/a"
    p = float(p)
    if p >= 0.001:
        return f"${p:.3f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def names(fms: Sequence[str]) -> str:
    """Model names joined for running text ('none' if empty)."""
    words = [NAMES.get(f, f) for f in fms]
    if not words:
        return "none"
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


FM_WORD = {"fm1": "FMone", "fm2": "FMtwo", "fm3": "FMthree", "fm5": "FMfive"}
X_WORD = {"fm6": "six", "fm7": "seven", "fm8": "eight"}


def study13_macros(part: Mapping[str, Any], joint: pd.DataFrame) -> dict[str, str]:
    """Study 13 macros of heldout_branches.md Sec. 5."""
    conf = part["confirmatory"].set_index("id")
    j = joint[joint.study == "study13"].set_index("test")
    m: dict[str, str] = {"HON": str(part["n_tables"])}
    c1 = c1_cross(joint)
    fm_rows = conf.loc[[f"(i)-{f}" for f in a13.FMS]]
    m.update(
        HOINRep=str(c1["within"]["replicated"]),
        HOINRepX=str(c1["cross"]["replicated"]),
        HOINRev=str(c1["within"]["reversed"]),
        HOINNotEval=str(int((fm_rows.verdict == "not evaluable").sum())),
        HOINAtten=str(int((fm_rows.reading == "replicated, attenuated").sum())),
        HOIMinMean=fmt_num(fm_rows["mean"].astype(float).min()),
        HOIMaxMean=fmt_num(fm_rows["mean"].astype(float).max()),
    )
    for cid, word in [
        *((f"(i)-{f}", f"I{FM_WORD[f]}") for f in a13.FMS),
        ("(ii-a)", "IIa"),
        ("(ii-c)", "IIc"),
    ]:
        row = conf.loc[cid]
        m.update(
            {
                f"HO{word}Raw": fmt_p(row["p"]),
                f"HO{word}Holm": fmt_p(row["p_holm"]),
                f"HO{word}XHolm": fmt_p(j.loc[cid, "p_holm_cross"]),
                f"HO{word}Verdict": str(row["verdict"]),
                f"HO{word}Mean": fmt_num(row["mean"]),
            }
        )
    iic = conf.loc["(ii-c)"]
    m.update(
        HOIIcLo=fmt_num(iic["ci_low"]),
        HOIIcHi=fmt_num(iic["ci_high"]),
        HOIIcWilcoxonP=fmt_p(iic["p_wilcoxon"]),
        HOIIcLeader=NAMES["fm2"] if float(iic["mean"]) > 0 else NAMES["fm5"],
        HORankingVerdict=part["overall"]["ranking"],
        HOXFamN=str(len(joint)),
    )
    m.update(_integrity_macros(part))
    m.update(_secondary13_macros(part))
    return m


def _integrity_macros(part: Mapping[str, Any]) -> dict[str, str]:
    m = dict(
        HOSensIntNFlag=str(len(part["flagged"])),
        HOSensIntNDrop=str(len(part["flagged"])),
    )
    if isinstance(part.get("sens_int"), pd.DataFrame):
        m["HOSensIntCOneVerdict"] = a13.overall(part["sens_int"])["c1"]
    labels = part.get("labels")
    if isinstance(labels, dict):
        m.update(
            HOSensLabelNDrop=str(len(labels["merged_tables"])),
            HOSensLabelCOneVerdict=a13.overall(labels["sens_label"])["c1"],
            HOLabelNMerged=str(len(labels["merged_tables"])),
            HOLabelNUnverified=str(len(labels["unverified_tables"])),
        )
    m["HONFailedTables"] = str(part.get("n_failed_tables", 0))
    return m


def _sig(table: pd.DataFrame, col: str = "p_holm", mean: str = "mean") -> pd.DataFrame:
    return table[(table[col].astype(float) < ALPHA) & (table[mean].astype(float) > 0)]


def _secondary13_macros(part: Mapping[str, Any]) -> dict[str, str]:
    m: dict[str, str] = {}
    for fn in (_s10_macros, _s11_macros, *SECONDARY_MACROS):
        m.update(guarded_macros(fn, part))
    return m


def guarded_macros(
    fn: Callable[[Mapping[str, Any]], dict[str, str]], part: Mapping[str, Any]
) -> dict[str, str]:
    """Macros of one block; none if its analysis failed (listed in the report)."""
    try:
        return fn(part)
    except Exception:  # noqa: BLE001 - the block's macros stay undefined
        return {}


def _s10_macros(part: Mapping[str, Any]) -> dict[str, str]:
    t = part["amendments"]["s10_new_fms"].set_index("learner")
    floor = (1 - s10.LOST_SHARE) * len_or(part)
    m: dict[str, str] = {}
    means = (
        a12.cell_means(a13.arm(_runs_of(part), "T"), "accuracy")
        if "_runs" in part
        else None
    )
    for fm, w in X_WORD.items():
        m[f"HOSTenFM{w}Mean"] = fmt_num(t.loc[fm, "mean"])
        m[f"HOSTenFM{w}Holm"] = fmt_p(t.loc[fm, "p_holm"])
        if means is not None:
            gap = a12.delta(
                means, (fm, "stratified"), (fm, "random"), a13.MULTS, "accuracy"
            )
            m[f"HOSTenAccFM{w}Mean"] = fmt_num(gap.mean() if len(gap) else np.nan)
    sig = (t.p_holm < ALPHA) & (t["mean"] > 0)
    rev = (t.p_holm < ALPHA) & (t["mean"] < 0)
    noteval = t.n.astype(float) < floor
    m.update(
        HOSTenHolmMax=fmt_p(t.p_holm.max()),
        HOSTenHolmMin=fmt_p(t.p_holm.min()),
        HOSTenMinMean=fmt_num(t["mean"].min()),
        HOSTenMaxMean=fmt_num(t["mean"].max()),
        HOSTenNExtend=str(int(sig.sum())),
        HOSTenExtendList=names(list(t.index[sig])),
        HOSTenNotList=names(list(t.index[~sig])),
        HOSTenNotMean=", ".join(fmt_num(v) for v in t["mean"][~sig]) or "none",
        HOSTenNotHolm=", ".join(fmt_p(v) for v in t.p_holm[~sig]) or "none",
        HOSTenRevList=names(list(t.index[rev])),
        HOSTenRevMean=", ".join(fmt_num(v) for v in t["mean"][rev]) or "none",
        HOSTenRevHolm=", ".join(fmt_p(v) for v in t.p_holm[rev]) or "none",
        HOSTenNotEvalList=names(list(t.index[noteval])),
        HOSTenDisagreeList=names(list(t.index[t["mean"] < 0])),
    )
    return m


def _runs_of(part: Mapping[str, Any]) -> pd.DataFrame:
    runs: pd.DataFrame = part["_runs"]
    return runs


def _s11_macros(part: Mapping[str, Any]) -> dict[str, str]:
    t = part["amendments"]["s11_new_fms_minus_classical"].set_index("learner")
    sig = t.p_holm < ALPHA
    signs = set(np.sign(t["mean"][sig]))
    word = (
        "positive"
        if signs == {1.0}
        else "negative" if signs == {-1.0} else "mixed in sign"
    )
    return dict(
        HOSElevenMinMean=fmt_num(t["mean"].min()),
        HOSElevenMaxMean=fmt_num(t["mean"].max()),
        HOSElevenSignWord=word if sig.any() else "none",
        HOSElevenSigList=names(list(t.index[sig])),
        HOSElevenSigMean=", ".join(fmt_num(v) for v in t["mean"][sig]) or "none",
        HOSElevenSigHolm=", ".join(fmt_p(v) for v in t.p_holm[sig]) or "none",
    )


def _s1_macros(part: Mapping[str, Any]) -> dict[str, str]:
    return dict(HOSOneNSig=str(len(_sig(part["s1_classical"]))))


def _s2_macros(part: Mapping[str, Any]) -> dict[str, str]:
    s2 = part["s2_auc_all"]["auc_all"].set_index("learner")
    sig = (s2.m1_p_holm < ALPHA) & (s2.m1_mean > 0)
    return dict(
        HOSTwoNSig=str(int(sig.sum())), HOSTwoNotList=names(list(s2.index[~sig]))
    )


def _s3_macros(part: Mapping[str, Any]) -> dict[str, str]:
    s3 = part["s3_imbalance"]["table"].set_index("learner").loc["mean"]
    return dict(HOSThreeRho=fmt_num(s3["rho"], 2), HOSThreeP=fmt_p(s3["p_one_sided"]))


def _s4_macros(part: Mapping[str, Any]) -> dict[str, str]:
    s4 = part["s4_fixed_split"]
    arm_diff = s4["t_minus_f"]["mean"].astype(float)
    return dict(
        HOSFourMinMean=fmt_num(s4["gaps"]["mean"].min()),
        HOSFourMaxMean=fmt_num(s4["gaps"]["mean"].max()),
        HOSFourNSig=str(len(_sig(s4["gaps"]))),
        HOSFourArmDiffMin=fmt_num(arm_diff.min()),
        HOSFourArmDiffMax=fmt_num(arm_diff.max()),
    )


def _s5_macros(part: Mapping[str, Any]) -> dict[str, str]:
    s5 = part["s5_split"].set_index("id")
    at1 = s5.loc[[f"(i)-{f} @1C" for f in a13.FMS]]

    def per_fm(col: str, f: Callable[[Any], str]) -> str:
        return ", ".join(f"{NAMES[fm]} {f(v)}" for fm, v in zip(a13.FMS, at1[col]))

    return dict(
        HOSFiveBin=per_fm("binary_mean", fmt_num),
        HOSFiveMulti=per_fm("multi_mean", fmt_num),
        HOSFiveMW=per_fm("mw_p", fmt_p),
        HOSFiveNMulti=str(int(at1["multi_n"].iloc[0])),
    )


def _s6_macros(part: Mapping[str, Any]) -> dict[str, str]:
    meta = part["s6_meta"]
    pi = meta[(meta.metric == "macro_f1") & (meta.m == "all")].set_index("learner")
    lows = {f: float(pi.loc[f, "pi"][0]) for f in a13.FMS}
    highs = {f: float(pi.loc[f, "pi"][1]) for f in a13.FMS}
    return dict(
        HOSSixPIAboveList=names([f for f in a13.FMS if lows[f] > 0]),
        HOSSixPIStraddleList=names([f for f in a13.FMS if lows[f] <= 0 <= highs[f]]),
    )


def _s7_macros(part: Mapping[str, Any]) -> dict[str, str]:
    s7 = part["s7_fm_minus_classical"].set_index("learner")
    return dict(
        HOSSevenFMfiveMean=fmt_num(s7.loc["fm5", "mean"]),
        HOSSevenFMfiveHolm=fmt_p(s7.loc["fm5", "p_holm"]),
        HOSSevenFMtwoMean=fmt_num(s7.loc["fm2", "mean"]),
        HOSSevenFMtwoHolm=fmt_p(s7.loc["fm2", "p_holm"]),
    )


def _s8_macros(part: Mapping[str, Any]) -> dict[str, str]:
    s8 = part["s8_redraw"]
    sig = [str(i).split()[-1] for i in _sig(s8["table"])["id"]]
    rank = s8["rank"]
    return dict(
        HOSEightSigList=names(sig),
        HOSEightRank=f"{fmt_num(rank['mean'])} (Wilcoxon $p=$ {fmt_p(rank['p'])})",
        HOSEightNInfeasible=str(part["s8_infeasible"]),
    )


def _s9_macros(part: Mapping[str, Any]) -> dict[str, str]:
    s9 = part["s9_flips"]
    return dict(
        HOSNineF=fmt_num(s9["rate"], sign=False),
        HOSNineFNoise=fmt_num(s9["noise_rate"], sign=False),
        HOSNineFDiff=fmt_num(s9["diff"]),
        HOSNineFDiffLo=fmt_num(s9["diff_ci"][0]),
        HOSNineFDiffHi=fmt_num(s9["diff_ci"][1]),
    )


def _s12_macros(part: Mapping[str, Any]) -> dict[str, str]:
    fam = part["amendments"]["s12_kappa_mcc"]
    return dict(HOSTwelveNSig=str(len(_sig(fam))))


def _s13_macros(part: Mapping[str, Any]) -> dict[str, str]:
    fam = part["amendments"]["s13_auc_all_minus_seen"]["table"]
    return dict(HOSThirteenNSig=str(len(_sig(fam))))


def _s14_macros(part: Mapping[str, Any]) -> dict[str, str]:
    fam = part["amendments"]["s14_citl_imbalanced"]["table"]
    return dict(HOSFourteenNSig=str(len(_sig(fam))))


SECONDARY_MACROS = (
    _s1_macros, _s2_macros, _s3_macros, _s4_macros, _s5_macros, _s6_macros,
    _s7_macros, _s8_macros, _s9_macros, _s12_macros, _s13_macros, _s14_macros,
)  # fmt: skip


def study9_macros(part: Mapping[str, Any], joint: pd.DataFrame) -> dict[str, str]:
    """Study 9 macros of heldout_branches.md Sec. 5."""
    j = joint[joint.study == "study9"].set_index("test")
    st = "H2a|stratified"
    others = [k for k in S9_IDS if k not in (st, "H3b")]
    supported = [k for k in others if j.loc[k, "verdict_within"] == "supported"]
    m = dict(
        HOSnineN=str(part["n"]),
        HOSnineRawHTwoaStrat=fmt_p(j.loc[st, "p"]),
        HOSnineHolmHTwoaStrat=fmt_p(j.loc[st, "p_holm_within"]),
        HOSnineXHolmHTwoaStrat=fmt_p(j.loc[st, "p_holm_cross"]),
        HOSnineHTwoaVerdict=str(j.loc[st, "verdict_within"]),
        HOSnineHolmHThreeb=fmt_p(j.loc["H3b", "p_holm_within"]),
        HOSnineHThreebTau=fmt_num(part["result"]["h3"]["h3b_tau"], 2),
        HOSnineNOtherSupported=str(len(supported)),
        HOSnineOtherSupportedList=", ".join(supported) if supported else "none",
    )
    extras = part["extras"]
    if isinstance(extras, dict) and "magnitude|stratified" in extras:
        mag = extras["magnitude|stratified"]
        m.update(
            HOSnineRhoOneStrat=fmt_num(mag["median"], 2, sign=False),
            HOSnineRhoOneStratLo=fmt_num(mag["ci"][0], 2, sign=False),
            HOSnineRhoOneStratHi=fmt_num(mag["ci"][1], 2, sign=False),
            HOSnineMagWord="retained" if mag["retained"] else "not retained",
        )
    if isinstance(extras, dict) and "fm1" in extras and st in extras["fm1"]:
        m["HOSnineFMoneStratP"] = fmt_p(extras["fm1"][st])
    return m


def now() -> str:
    """UTC time stamp, seconds."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def tex_file(macros: Mapping[str, str], set_name: str) -> str:
    r"""``\def`` lines, sorted; the names that cannot be produced as comments."""
    lines = [
        "% Generated by simaudit.analysis.joint_heldout; do not edit by hand.",
        f"% set: {set_name}; generated {now()}",
    ]
    lines += [f"\\def\\{k}{{{v}}}" for k, v in sorted(macros.items())]
    lines.append("% Not produced by the pre-registered analysis (J-8):")
    lines += [f"%   \\{k}" for k in NOT_PRODUCED]
    return "\n".join(lines) + "\n"


# ===================================================================== output
def jsonable(obj: Any) -> Any:
    """Convert analysis output (frames, numpy values, tuples) to JSON types."""
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="split", default_handler=str))
    if isinstance(obj, pd.Series):
        return jsonable(obj.to_dict())
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def md(obj: Any) -> str:
    """A result block as markdown (tables as tables, the rest as JSON)."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_markdown() if len(obj) else "(empty)"
    if isinstance(obj, str):
        return obj
    return "```json\n" + json.dumps(jsonable(obj), indent=1)[:20000] + "\n```"


def report(result: Mapping[str, Any]) -> str:
    """The joint report: completeness, the 19-test table, then every study."""
    lines = [
        f"# Joint HELD-OUT analysis, set {result['set']}",
        "",
        f"Generated {result['generated']} by simaudit.analysis.joint_heldout.",
        "",
        "## Completeness (opening note Sec. 2)",
        "",
        md(result["pending"]),
        "",
        "## Cross-study Holm family (opening note Sec. 3)",
        "",
        md(result["joint"]),
        "",
        md(result.get("c1", {})),
        "",
    ]
    for study in STUDIES:
        part = result.get(study)
        if part is None:
            continue
        lines += [f"## {study}", ""]
        for key, value in part.items():
            if key.startswith("_"):
                continue
            lines += [f"### {study}: {key}", "", md(value), ""]
    lines += ["## Readings (driver docstring J-1 to J-8; Study 10 R10-x)", ""]
    lines += [
        md(list(s10.READINGS)),
        "",
        f"Macros not produced: {', '.join(NOT_PRODUCED)}",
    ]
    return "\n".join(lines) + "\n"


def run(
    layout: Layout,
    tables: Tables,
    skip: Sequence[str] = (),
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Steps 1-6 of the module docstring; returns the result and writes files."""
    pending = completeness(layout, skip)
    if any(pending.values()) and not allow_incomplete:
        raise SystemExit(
            "confirmatory cells still pending, no outcome read: "
            + json.dumps(
                {
                    k: v[:10] + (["..."] if len(v) > 10 else [])
                    for k, v in pending.items()
                }
            )
        )
    layout.out.mkdir(parents=True, exist_ok=True)
    parts: dict[str, Any] = {}
    if "study13" not in skip:
        parts["study13"] = study13_part(layout)
        parts["study13"]["n_expected"] = len(pool_ids(layout.root, layout.set_name))
        parts["study13"]["n_failed_tables"] = len(
            {
                j.split("/")[0]
                for j in pending.get("study13", [])
                if j.split("/")[1] in S13_PRIMARY_GROUPS
            }
            | _failed13(layout)
        )
    if "study9" not in skip:
        parts["study9"] = study9_part(layout, tables)
    if "study10" not in skip:
        parts["study10"] = study10_part(layout, tables)
    joint = joint_family(parts)
    result: dict[str, Any] = dict(
        set=layout.set_name,
        generated=now(),
        pending=pending,
        joint=joint,
        **parts,
    )
    if "study13" in parts:
        result["c1"] = c1_cross(joint)
    macros: dict[str, str] = {}
    if "study13" in parts:
        parts["study13"]["_runs"] = a13.load_runs(layout.study_dir("study13"))
        macros.update(study13_macros(parts["study13"], joint))
    if "study9" in parts:
        macros.update(study9_macros(parts["study9"], joint))
    result["macros"] = macros
    (layout.out / "joint_report.md").write_text(report(result))
    clean = {k: v for k, v in result.items()}
    if "study13" in clean:
        clean["study13"] = {
            k: v for k, v in clean["study13"].items() if not k.startswith("_")
        }
    (layout.out / "joint_results.json").write_text(
        json.dumps(jsonable(clean), indent=1)
    )
    layout.tex.parent.mkdir(parents=True, exist_ok=True)
    layout.tex.write_text(tex_file(macros, layout.set_name))
    return result


def _failed13(layout: Layout) -> set[str]:
    return {
        str(j[0])
        for j in _failures(layout.study_dir("study13"))
        if int(j[2]) >= 2 and str(j[1]) in S13_PRIMARY_GROUPS
    }


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    """``python -m simaudit.analysis.joint_heldout --set {dev,heldout}``."""
    from simaudit.settings import PROJECT_ROOT, settings

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=SETS, required=True, dest="set_name")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tex", type=Path, default=None)
    parser.add_argument("--skip", default="", help="comma list of studies")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    out = args.out or settings.results_dir / "joint" / args.set_name
    tex = args.tex or (
        PROJECT_ROOT / "paper" / "p1" / "heldout_numbers.tex"
        if args.set_name == "heldout"
        else out / "heldout_numbers_dev.tex"
    )
    layout = Layout(PROJECT_ROOT, settings.results_dir, args.set_name, out, tex)
    skip = [s for s in args.skip.split(",") if s]
    result = run(
        layout, Tables(layout, settings.data_cache_dir), skip, args.allow_incomplete
    )
    print(result["joint"].to_string())
    print(f"wrote {out / 'joint_report.md'}, {out / 'joint_results.json'}, {tex}")


if __name__ == "__main__":  # pragma: no cover
    main()
