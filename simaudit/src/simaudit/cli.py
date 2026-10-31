"""Command line: pools, pilot, studies 2-9 and the pilot report."""

from pathlib import Path
from typing import Any

import typer
import yaml

from simaudit.analysis.report import build_report
from simaudit.data.openml_loader import load
from simaudit.autoresearch import harness, select_harness
from simaudit.runner.build_pool import build_pool
from simaudit.runner import build_pool_ext, build_pool_pmlb, build_pool_uci, t1prime
from simaudit.runner import screen_integrity
from simaudit.runner.extension import EXT_MODELS, run_extension_dataset_m
from simaudit.runner.pilot import run_pilot
from simaudit.runner import study2_b, study4, study7, study9, study10
from simaudit.runner.study2 import GRIDS, run_stage_a
from simaudit.settings import PROJECT_ROOT, settings

app = typer.Typer(no_args_is_help=True)
POOL = PROJECT_ROOT / "data" / "pool_i2.yaml"
PILOT_DIR = settings.results_dir / "pilot_i2"
EXT_DIR = settings.results_dir / "pilot_i2_ext"
STUDY2_DIR = settings.results_dir / "study2_stage_a"
STUDY4_DIR = settings.results_dir / "study4_stage_a"
STUDY7_DIR = settings.results_dir / "study7"
STUDY9_DIR = settings.results_dir / "study9"
STUDY10_DIR = settings.results_dir / "study10"
POOL_EXT = PROJECT_ROOT / "data" / "pool_ext.yaml"
POOL_T1PRIME = PROJECT_ROOT / "data" / "pool_t1prime.yaml"
UCI_SNAPSHOT = PROJECT_ROOT / "data" / "uci_api_snapshot.json.gz"
# Study 9 Sec. 1: learners per set and expansion tier.
STUDY9_LEARNERS = {
    "dev": ("fm2",),
    1: ("fm1", "fm2", "fm3", "fm5"),
    2: ("fm1", "fm2"),
    3: ("fm1", "fm2"),
    # Study 9 Amendment 1: EXP' = T1', all four FMs, cheapest first.
    "t1prime": ("fm2", "fm1", "fm5", "fm3"),
}
SEARCH_CACHE = settings.data_cache_dir / "search"
LEDGER = settings.results_dir / "autoresearch" / "ledger.jsonl"
NULL_LEDGER = settings.results_dir / "autoresearch" / "null_ledger.jsonl"
SELECT_DIR = settings.results_dir / "autoresearch" / "workstream_b"
SELECT_FM_CACHE = settings.data_cache_dir / "select_fm"


@app.command("build-pool")
def build_pool_cmd() -> None:
    """Query OpenML, validate candidates and freeze data/pool_i2.yaml."""
    pool = build_pool(
        settings.data_cache_dir,
        PROJECT_ROOT / "data" / "priorsel_excluded_ids.txt",
        POOL,
        settings.base_seed,
    )
    typer.echo(
        f"valid unique: {pool['n_valid_unique']}, DEV {len(pool['dev'])}, "
        f"HELD-OUT {len(pool['heldout'])}, rejected {len(pool['rejected'])}",
    )


@app.command("build-pool-ext")
def build_pool_ext_cmd() -> None:
    """Study 8: draw the stratified expansion pool into data/pool_ext.yaml."""
    data = PROJECT_ROOT / "data"
    pool = build_pool_ext.build_pool_ext(
        settings.data_cache_dir,
        POOL,
        data / "priorsel_excluded_ids.txt",
        data / "pool_ext.yaml",
        data / "pool_ext_audit.csv",
        meta=build_pool_ext.fetch_metadata(),
        suites=build_pool_ext.fetch_suite_ids(),
        pmlb_path=data / "pool_ext_pmlb.yaml",
    )
    for name, row in pool["strata"].items():
        typer.echo(f"{name}: {row}")
    typer.echo(
        f"datasets {len(pool['datasets'])}, PMLB removed {len(pool['pmlb_removed'])}, "
        f"post-load rejected {len(pool['post_load_rejected'])}"
    )


@app.command("build-pool-pmlb")
def build_pool_pmlb_cmd() -> None:
    """Study 8 Amendment 1: PMLB Tiers 2a and 3 into data/pool_ext_pmlb.yaml."""
    data = PROJECT_ROOT / "data"
    cache = settings.data_cache_dir
    pool = build_pool_pmlb.build_pool_pmlb(
        cache,
        POOL,
        data / "pool_ext_pmlb.yaml",
        data / "pool_ext_pmlb_audit.csv",
        summary=build_pool_pmlb.fetch_summary(cache),
        fetch=build_pool_pmlb.fetcher(cache),
        dev_loader=lambda did: load(did, cache),
    )
    for tier, counts in pool["counts"].items():
        typer.echo(f"Tier {tier}: {sum(counts.values())} {counts}")


@app.command("snapshot-uci")
def snapshot_uci_cmd() -> None:
    """Study 8 Amendment 2: freeze the UCI API metadata snapshot (no data files)."""
    snap = build_pool_uci.build_snapshot(
        build_pool_uci.http_json, build_pool_uci.http_text
    )
    sha = build_pool_uci.save_snapshot(snap, UCI_SNAPSHOT)
    typer.echo(f"{len(snap['datasets'])} datasets, missing {snap['missing']}, {sha}")


@app.command("build-pool-uci")
def build_pool_uci_cmd() -> None:
    """Study 8 Amendment 2: UCI Tier 4 into data/pool_ext_uci.yaml."""
    data = PROJECT_ROOT / "data"
    cache = settings.data_cache_dir
    sha = build_pool_uci.SNAPSHOT_SHA256
    pool = build_pool_uci.build_pool_uci(
        cache,
        POOL,
        data / "pool_ext_pmlb.yaml",
        data / "pool_ext_uci.yaml",
        data / "pool_ext_uci_audit.csv",
        snapshot=build_pool_uci.load_snapshot(UCI_SNAPSHOT, sha),
        snapshot_sha=sha,
        fetch=build_pool_uci.fetcher(),
        loader=lambda did: load(did, cache),
    )
    typer.echo(f"Tier 4: {sum(pool['counts'].values())} {pool['counts']}")


@app.command("screen-integrity")
def screen_integrity_cmd() -> None:
    """Study 8 Amendment 2 (A2.6): integrity screens on PMLB and UCI tiers."""
    data = PROJECT_ROOT / "data"
    cache = settings.data_cache_dir
    rows = screen_integrity.screen_pools(
        [data / "pool_ext_pmlb.yaml", data / "pool_ext_uci.yaml"],
        lambda did: load(did, cache),
        data / "pool_ext_integrity_audit.csv",
        {
            f"dev:{e['did']}": load(int(e["did"]), cache)
            for e in yaml.safe_load(POOL.read_text())["dev"]
        },
    )
    for r in rows:
        if r["decision"] != "keep" or r["class_sorted"]:
            typer.echo(f"{r['key']}: {r['decision']} sorted={r['class_sorted']}")
    typer.echo(f"screened {len(rows)}")


@app.command("build-t1prime")
def build_t1prime_cmd() -> None:
    """Study 8 Amendment 2: assemble T1' into data/pool_t1prime.yaml."""
    data = PROJECT_ROOT / "data"
    out = t1prime.build_t1prime(
        data / "pool_ext_pmlb.yaml",
        data / "pool_ext_uci.yaml",
        data / "pool_t1prime.yaml",
        data / "pool_ext_integrity_audit.csv",
    )
    for name, row in out["counts"].items():
        typer.echo(f"{name}: {row}")
    typer.echo(
        f"T1' {out['n_t1prime']} (balanced {out['n_balanced']}), sum n {out['sum_n']}"
    )


@app.command("run-pilot")
def run_pilot_cmd(workers: int = 6, limit: int = 0) -> None:
    """Run pilot I2 on the DEV datasets (resumable)."""
    dev = [d["did"] for d in yaml.safe_load(POOL.read_text())["dev"]]
    dev = dev[:limit] if limit else dev
    run_pilot(dev, settings.data_cache_dir, PILOT_DIR, settings.base_seed, workers)


@app.command("run-extension")
def run_extension_cmd(model: str = "tabfm", workers: int = 3, limit: int = 0) -> None:
    """Run one Amendment 2 model (tabfm | v35) on the pilot splits (resumable)."""
    if model not in EXT_MODELS:
        raise typer.BadParameter(f"model must be one of {sorted(EXT_MODELS)}")
    dev = [d["did"] for d in yaml.safe_load(POOL.read_text())["dev"]]
    dev = dev[:limit] if limit else dev
    run_pilot(
        dev,
        settings.data_cache_dir,
        EXT_DIR / model,
        settings.base_seed,
        workers,
        job=run_extension_dataset_m,
        job_args=(model,),
    )


@app.command("study2-a")
def study2_a_cmd(workers: int = 12, methods: str = "all", limit: int = 0) -> None:
    """Study 2 Stage A: SSL methods x 16 configurations on the pilot splits."""
    chosen = list(GRIDS) if methods == "all" else methods.split(",")
    unknown = sorted(set(chosen) - set(GRIDS))
    if unknown:
        raise typer.BadParameter(
            f"unknown methods {unknown}; choose from {sorted(GRIDS)}"
        )
    dev = [d["did"] for d in yaml.safe_load(POOL.read_text())["dev"]]
    dev = dev[:limit] if limit else dev
    run_stage_a(
        dev, chosen, settings.data_cache_dir, STUDY2_DIR, settings.base_seed, workers
    )


def _dev_ids() -> list[int]:
    return [d["did"] for d in yaml.safe_load(POOL.read_text())["dev"]]


@app.command("study2-b")
def study2_b_cmd(workers: int = 8, methods: str = "all", limit: int = 0) -> None:
    """Study 2 Stage B: honest CV selection (R2) for the pairs left open."""
    chosen = set(methods.split(","))
    pairs = [p for p in study2_b.STAGE_B_PAIRS if methods == "all" or p[0] in chosen]
    dev = _dev_ids()
    dev = dev[:limit] if limit else dev
    study2_b.run_stage_b(
        pairs,
        dev,
        settings.data_cache_dir,
        settings.results_dir / "study2_stage_b",
        settings.base_seed,
        workers,
    )


def _study4_strategies(names: str) -> tuple[str, ...]:
    chosen = tuple(names.split(",")) if names else study4.STAGE_A
    unknown = sorted(set(chosen) - set(study4.ORDER))
    if unknown:
        raise typer.BadParameter(f"unknown strategies {unknown}")
    return chosen


@app.command("study4-a")
def study4_a_cmd(
    workers: int = 4, limit: int = 0, learner: str = "fm5", strategies: str = ""
) -> None:
    """Study 4: label-selection strategies x budgets (Stage A: fm5; B: fm1-3).

    ``--strategies lucos`` runs only that subset (Amendment 2) into separate
    ``{did}[_{learner}]_lucos.parquet`` files; the default is the Stage A set.
    """
    chosen = _study4_strategies(strategies)
    dev = _dev_ids()
    dev = dev[:limit] if limit else dev
    out = STUDY4_DIR if learner == "fm5" else STUDY4_DIR.with_name("study4_stage_b")
    study4.run_stage_a(
        dev, settings.data_cache_dir, out, settings.base_seed, workers, learner, chosen
    )


def _study7_plan(plan: str) -> tuple[tuple[str, str], ...]:
    if not plan:
        return study7.PLAN
    chosen = tuple(
        (part, learner)
        for part, _, learner in (item.partition(":") for item in plan.split(","))
    )
    bad = [f"{p}:{lr}" for p, lr in chosen if lr not in study7.PART_LEARNERS.get(p, ())]
    if bad:
        raise typer.BadParameter(f"unknown part:learner {bad}")
    return chosen


def _study7_drop(drop: str) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for item in filter(None, drop.split(",")):
        did, _, col = item.partition(":")
        out[int(did)] = (*out.get(int(did), ()), int(col))
    return out


@app.command("study7")
def study7_cmd(
    workers: int = 3,
    limit: int = 0,
    plan: str = "",
    datasets: str = "",
    drop: str = "",
    out_name: str = "study7",
) -> None:
    """Study 7 (DEV): protocol re-run (a), balance option (b), dose-response (c).

    ``--plan c:fm2,b:fm1_bal`` runs those jobs in that order; the default is the
    pre-registered order. ``--datasets 694 --drop 694:1 --out-name study7_s4``
    re-runs a DEV dataset without a feature column (Addendum 1, S4). Outputs go
    to ``results/{out-name}/{part}/``, resumable.
    """
    dev = _dev_ids()
    chosen = [int(d) for d in datasets.split(",")] if datasets else dev
    if not set(chosen) <= set(dev):
        raise typer.BadParameter("Study 7 runs DEV datasets only")
    chosen = chosen[:limit] if limit else chosen
    study7.run(
        chosen,
        settings.data_cache_dir,
        settings.results_dir / out_name,
        settings.base_seed,
        workers,
        _study7_plan(plan),
        _study7_drop(drop),
    )


def _study9_jobs(set_name: str) -> list[tuple[list[int], tuple[str, ...]]]:
    """(dataset IDs, learners) groups for a Study 9 set; HELD-OUT is not exposed."""
    if set_name == "dev":
        return [(_dev_ids(), STUDY9_LEARNERS["dev"])]
    if set_name == "t1prime":
        pool = yaml.safe_load(POOL_T1PRIME.read_text())
        ids = [d["did"] for d in pool["datasets"]]
        return [(ids, STUDY9_LEARNERS["t1prime"])]
    tiers = {"ext": (1, 2), "tier3": (3,)}.get(set_name)
    if tiers is None:
        raise typer.BadParameter("set must be dev, t1prime, ext or tier3")
    pool = yaml.safe_load(POOL_EXT.read_text())
    return [
        ([d["did"] for d in pool["datasets"] if d["tier"] == t], STUDY9_LEARNERS[t])
        for t in tiers
    ]


@app.command("study9")
def study9_cmd(
    set_name: str = typer.Option("dev", "--set"),
    workers: int = 2,
    limit: int = 0,
    learners: str = "",
) -> None:
    """Study 9 (Workstream C): nested label budgets 1C-30C, stored probabilities.

    ``--set dev`` is the FM2 dry-run; ``t1prime`` is EXP' (all four FMs,
    FM2 first); ``ext`` runs Tiers 1 and 2 of
    data/pool_ext.yaml with their pre-registered learners; ``tier3`` the
    secondary tier. ``--learners fm2`` restricts the learners.
    """
    chosen = set(learners.split(",")) if learners else None
    for ids, group in _study9_jobs(set_name):
        ids = ids[:limit] if limit else ids
        run_with = [lr for lr in group if chosen is None or lr in chosen]
        if ids and run_with:
            study9.run(
                set_name,
                ids,
                run_with,
                settings.data_cache_dir,
                STUDY9_DIR / set_name,
                settings.base_seed,
                workers,
            )


def _study10_ids(set_name: str) -> tuple[list[int], dict[int, str] | None]:
    """Dataset IDs of a Study 10 set and, for HELD-OUT, the pool content hashes."""
    if set_name == "dev":
        return _dev_ids(), None
    if set_name == "t1prime":
        pool = yaml.safe_load(POOL_T1PRIME.read_text())
        # Largest first (jobs are independent): the n = 5000 jobs, with 20
        # ZCore calls each, do not end up last on one worker.
        largest = sorted(pool["datasets"], key=lambda d: -int(d["n"]))
        return [d["did"] for d in largest], None
    if set_name == "heldout":
        held = yaml.safe_load(POOL.read_text())["heldout"]
        return [d["did"] for d in held], {d["did"]: d["sha256"] for d in held}
    raise typer.BadParameter("set must be dev, heldout or t1prime")


@app.command("study10")
def study10_cmd(
    set_name: str = typer.Option("dev", "--set"),
    learners: str = ",".join(study10.LEARNERS),
    workers: int = 4,
    limit: int = 0,
) -> None:
    """Study 10: confirmation of the consensus-core family (F1 primary).

    Learners run in the order given (FM2 first). ``--set heldout`` refuses to
    run unless prereg/STUDY10_SELECTOR_CONFIRMATION.md is FROZEN, and checks
    every dataset against its pool hash.
    """
    ids, hashes = _study10_ids(set_name)
    ids = ids[:limit] if limit else ids
    chosen = [lr for lr in learners.split(",") if lr]
    unknown = sorted(set(chosen) - set(study10.LEARNERS))
    if unknown:
        raise typer.BadParameter(f"unknown learners {unknown}")
    out = STUDY10_DIR / set_name
    extra: dict[str, Any] = {}
    if set_name == "heldout":  # the joint-opening loader, guards first (Sec. 3.5)
        sealed = PROJECT_ROOT.parent / "sealed_heldout"
        extra["loader"] = study10.heldout_loader(PROJECT_ROOT, sealed, out)
    study10.run(
        set_name,
        ids,
        chosen,
        settings.data_cache_dir,
        out,
        workers,
        PROJECT_ROOT,
        hashes,
        **extra,
    )


@app.command("search-cache")
def search_cache_cmd(workers: int = 6) -> None:
    """Study 3: cache FM outputs for DEV search cells (seeds disjoint from pilot)."""
    n = harness.build_search_cache(
        _dev_ids(), settings.data_cache_dir, SEARCH_CACHE, settings.base_seed, workers
    )
    typer.echo(f"search cells cached: {n}")


@app.command("evaluate")
def evaluate_cmd(
    candidate: Path,
    workers: int = 8,
    note: str = "",
    null: bool = False,
    tag: str = "",
) -> None:
    """Study 3: score a candidate on the DEV search cells and log it.

    ``--null`` scores against permuted hidden labels (null control loop);
    ``--tag v2`` writes to ``v2_ledger.jsonl`` / ``v2_null_ledger.jsonl``.
    """
    ledger = NULL_LEDGER if null else LEDGER
    if tag:
        ledger = ledger.with_name(f"{tag}_{ledger.name}")
    entry = harness.evaluate(
        candidate,
        _dev_ids(),
        settings.data_cache_dir,
        SEARCH_CACHE,
        ledger,
        workers,
        note,
        null=null,
    )
    by_m = ", ".join(f"m={m}: {v:.4f}" for m, v in entry["by_m"].items())
    acc = entry["secondary"]["accuracy"]
    acc_m = ", ".join(f"m={m}: {v:.4f}" for m, v in acc.items())
    typer.echo(f"score {entry['score']:.4f} ({by_m}); failed {entry['n_failed']}")
    typer.echo(f"accuracy {sum(acc.values()) / len(acc):.4f} ({acc_m})")
    if entry["first_error"]:
        typer.echo(entry["first_error"])


@app.command("select-eval")
def select_eval_cmd(
    candidate: Path,
    workers: int = 4,
    note: str = "",
    null: bool = False,
    ledger_name: str = "",
    trusted: bool = False,
    rescore: bool = False,
) -> None:
    """Study 5 (Workstream B loop): score a label selector on DEV and log it.

    Ledgers: ``ledger.jsonl`` (main), ``null_ledger.jsonl`` (``--null``) or
    ``<ledger-name>.jsonl`` (for example the literature references).
    ``--trusted`` skips the forbidden-token check, for references that call
    project code (LUCoS); it is allowed only with ``--ledger-name reference``
    and never with ``--null``. ``--rescore`` uses fresh seeds (namespace 9,
    5 reps) for the final re-score of branch bests.
    """
    if trusted and (ledger_name != "reference" or null):
        raise typer.BadParameter(
            "--trusted requires --ledger-name reference and no --null"
        )
    name = ledger_name or ("null_ledger" if null else "ledger")
    entry = select_harness.evaluate(
        candidate,
        _dev_ids(),
        settings.data_cache_dir,
        SELECT_FM_CACHE,
        SELECT_DIR / f"{name}.jsonl",
        workers,
        note,
        null=null,
        base_seed=settings.base_seed,
        trusted=trusted,
        rescore=rescore,
    )
    for metric in ("macro_f1", "accuracy", "coverage"):
        values = entry["by_m"] if metric == "macro_f1" else entry["secondary"][metric]
        per = ", ".join(f"{m}C: {v:.4f}" for m, v in values.items())
        mean = sum(values.values()) / len(values)
        typer.echo(f"{metric} {mean:.4f} ({per})")
    typer.echo(f"failed {entry['n_failed']}; {entry['seconds']:.0f} s")
    if entry["first_error"]:
        typer.echo(entry["first_error"])


@app.command("report")
def report_cmd(results: Path = PILOT_DIR, extension: Path = EXT_DIR) -> None:
    """Write and print the pilot report; extension results feed the secondary part."""
    typer.echo(build_report(results, extension))


if __name__ == "__main__":
    app()
