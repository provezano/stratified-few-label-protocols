"""Study 15: Study 13 test (i) replicated on a second, post-FM frame.

Prereg: prereg/STUDY15_SECOND_FRAME.md. The design is Study 13's, reused
unchanged through :mod:`simaudit.runner.study13`: the same row permutation
(namespace 35), cell seeds (namespace 34, arms T = 0 and F = 2), redraw
seeds, fixed split (namespace 36), strategies, reps, budgets, learners and
per-fit metrics. Only the data differ: the 20 tables of
``data/pool_second_frame.yaml``, loaded by
:class:`simaudit.data.second_frame.SecondFrameLoader`, which re-checks each
table's frozen content hash. Their IDs (source offset + repository ID) are
disjoint from every OpenML, PMLB and UCI ID, so every seed is new.

Guard: ``--set second_frame`` refuses to run unless the prereg carries a
``Status: FROZEN`` line and is committed unchanged against HEAD. No
HELD-OUT path is read.
"""

import json
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from simaudit.data import second_frame as sf
from simaudit.runner import study13 as s13

SET_NAME = "second_frame"
PREREG = "STUDY15_SECOND_FRAME.md"
GROUPS = (*s13.FMS, "classical")  # fm2, fm5, fm1, fm3, then the classical group
POOL_FILE = "pool_second_frame.yaml"


def check_frozen(root: Path) -> Path:
    """Raise unless the Study 15 prereg is FROZEN and committed unchanged."""
    path = root / "prereg" / PREREG
    if not path.exists():
        raise PermissionError(f"{path} does not exist")
    status = re.search(r"^Status:\s*\**\s*(\w+)", path.read_text(), re.MULTILINE)
    if status is None or status.group(1) != "FROZEN":
        raise PermissionError(f"{path} is not FROZEN")
    rel = path.relative_to(root)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=root,
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise PermissionError(f"{rel} is not committed")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(rel)], cwd=root
    ).returncode:
        raise PermissionError(f"{rel} differs from HEAD")
    return path


def pool_hashes(pool: Mapping[str, Any]) -> dict[int, str]:
    """Frozen content hash per table ID."""
    return {int(d["did"]): str(d["sha256"]) for d in pool["datasets"]}


def run_job(
    did: int,
    group: str,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    *,
    loader: s13.Loader,
    predict: s13.Predictor | None = None,
) -> Path:
    """Study 13 ``run_job`` with the second-frame key (spec key) in file names."""
    key = sf.dataset_key(did)
    out = s13.output_path(out_dir, key, group)
    if out.exists():
        return out
    data = loader(did, cache_dir)
    perm = s13.row_permutation(len(data.y), did, base_seed)
    X, y = data.X[perm], data.y[perm]
    chosen = s13.selections(key, did, X, y, base_seed, out_dir)
    ctx = dict(
        X=s13.zscore_all(X) if group == "classical" else X,
        y=y,
        perm=perm,
        test=s13.test_positions(out_dir, key),
    )
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows = s13._resume(partial)
    done = {(r["learner"], r["_key"]) for r in rows}
    fits = {k: r for r in rows if (k := s13._random_key(r)) is not None}
    predictors = s13._predictors(group, predict)
    todo = [
        (cell, sel, name, fn)
        for cell, sel in s13._cells(s13.job_selections(chosen, group))
        for name, fn in predictors.items()
        if (name, cell) not in done
    ]
    try:
        for count, (cell, sel, name, fn) in enumerate(todo, start=1):
            base = fits.get((name, int(sel["mult"]), int(sel["rep"])))
            meta, record, reused = s13._fit_cell(sel, fn, ctx, base)
            row = dict(meta, learner=name, reused=reused, _key=cell, **record)
            rows.append(row)
            if (k := s13._random_key(row)) is not None:
                fits[k] = row
            if count % s13.CHECKPOINT_EVERY == 0:
                s13._atomic_write(pd.DataFrame(rows), partial)
    finally:
        s13._close(predictors)
    frame = pd.DataFrame(rows).drop(columns="_key")
    frame.insert(0, "dataset", key)
    s13._atomic_write(frame, out)
    partial.unlink(missing_ok=True)
    return out


def screen_all(ids: list[int], loader: s13.Loader, out_dir: Path) -> pd.DataFrame:
    """Study 13 integrity screen (flag-only), keyed by the second-frame key."""
    path = out_dir / "integrity_screen.csv"
    if path.exists():
        return pd.read_csv(path, keep_default_na=False, dtype=str)
    rows = []
    for did in ids:
        row = s13.screen_table(loader(did, Path()))
        row["key"] = sf.dataset_key(did)
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _write_provenance(
    out_dir: Path, base_seed: int, ids: list[int], groups: list[str], root: Path
) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    prereg = root / "prereg" / PREREG
    here = Path(__file__).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(
        json.dumps(
            dict(
                set=SET_NAME,
                prereg_sha256=s13.file_sha256(prereg) if prereg.exists() else None,
                runner_sha256=s13.file_sha256(here),
                study13_runner_sha256=s13.file_sha256(Path(s13.__file__).resolve()),
                loader_sha256=s13.file_sha256(Path(sf.__file__).resolve()),
                git_head=head,
                host=os.uname().nodename,
                pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                base_seed=base_seed,
                namespaces=dict(
                    seed=s13.SEED_NAMESPACE,
                    perm=s13.PERM_NAMESPACE,
                    split=s13.SPLIT_NAMESPACE,
                ),
                mults=s13.MULTS,
                reps=s13.REPS,
                groups=groups,
                datasets=ids,
                started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
            indent=2,
        )
    )


def run(
    ids: list[int],
    groups: list[str],
    out_dir: Path,
    base_seed: int,
    workers: int,
    root: Path,
    *,
    loader: s13.Loader,
    predict: s13.Predictor | None = None,
) -> list[Path]:
    """Screen, then every (group, table) job; failures retried once."""
    screen_all(ids, loader, out_dir)  # before any fit
    _write_provenance(out_dir, base_seed, ids, groups, root)
    os.environ.setdefault("OMP_NUM_THREADS", str(s13.TORCH_THREADS))
    jobs = [(did, g) for g in groups for did in ids]
    done: list[Path] = []
    for attempt in (1, 2):
        finished, jobs = _attempt(
            jobs, attempt, workers, out_dir, base_seed, loader, predict
        )
        done.extend(finished)
        if not jobs:
            break
    return done


def _attempt(
    jobs: list[tuple[int, str]],
    attempt: int,
    workers: int,
    out_dir: Path,
    base_seed: int,
    loader: s13.Loader,
    predict: s13.Predictor | None,
) -> tuple[list[Path], list[tuple[int, str]]]:
    done: list[Path] = []
    failed: list[tuple[int, str]] = []
    common: dict[str, Any] = dict(loader=loader)
    if workers == 1:
        for job in jobs:
            try:
                done.append(
                    run_job(*job, Path(), out_dir, base_seed, predict=predict, **common)
                )
            except Exception as exc:  # noqa: BLE001 - logged; retried once
                s13._log_failure(out_dir, (*job, attempt), exc)
                failed.append(job)
        return done, failed
    from concurrent.futures import ProcessPoolExecutor, as_completed  # pragma: no cover

    with ProcessPoolExecutor(  # pragma: no cover - multiprocess path
        max_workers=workers, initializer=s13._init_worker
    ) as pool:
        futures = {
            pool.submit(run_job, *job, Path(), out_dir, base_seed, **common): job
            for job in jobs
        }
        for future in as_completed(futures):
            try:
                done.append(future.result())
                print(f"done {futures[future]} ({len(done)}/{len(jobs)})", flush=True)
            except Exception as exc:  # noqa: BLE001 - logged; retried once
                s13._log_failure(out_dir, (*futures[future], attempt), exc)
                failed.append(futures[future])
    return done, failed


def main() -> None:  # pragma: no cover - command-line entry
    """``python -m simaudit.runner.study15 --set second_frame [--workers 6]``."""
    import argparse

    import yaml

    from simaudit.settings import PROJECT_ROOT, settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=(SET_NAME,), default=SET_NAME)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--groups", default=",".join(GROUPS))
    args = parser.parse_args()
    check_frozen(PROJECT_ROOT)
    pool = yaml.safe_load((PROJECT_ROOT / "data" / POOL_FILE).read_text())
    ids = [int(d["did"]) for d in pool["datasets"]]
    loader = sf.SecondFrameLoader(
        settings.data_cache_dir / "second_frame" / "npz", pool_hashes(pool)
    )
    run(
        ids,
        args.groups.split(","),
        settings.results_dir / "study15" / SET_NAME,
        settings.base_seed,
        args.workers,
        PROJECT_ROOT,
        loader=loader,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
