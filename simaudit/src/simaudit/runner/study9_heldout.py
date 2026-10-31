"""Study 9 Sec. 9 HELD-OUT replication: a thin sealed driver (joint opening).

``runner/study9.py`` and ``analysis/study9.py`` are hash-frozen (Study 9
Freeze note 1) and are not changed. Their ``run`` passes no loader, so on
HELD-OUT IDs the default ``openml_loader.load`` would download the tables from
OpenML and bypass the seal. This driver calls ``study9.run_job`` with
``loader=study13.SealedLoader`` instead, after these checks, in this order,
before any sealed file is touched:

1. ``study13.check_heldout_go``: ``P1_HELDOUT_GO=1`` and the Study 13 prereg
   frozen and committed. HELD-OUT is opened jointly for Studies 9 and 13
   (owner decision 2026-10-27, research_notes/WORKSTREAMS.md), so
   Study 9 cannot open it alone;
2. the Study 9 prereg reads FROZEN and is committed unchanged;
3. the frozen Study 9 runner and analysis match their pinned SHA-256;
4. ``study13.verify_manifest``: MANIFEST.json IDs equal the pool_i2 HELD-OUT
   IDs and every npz's bytes match before any npz is parsed.

Every access is logged to ``results/study9/heldout/access_log.jsonl``.

Study 9 Amendment 2 (2026-10-28, before any HELD-OUT access): the loader runs
the label-normalisation step of Study 13 Amendment 1 (A)
(``SealedLoader(normalise_labels=True)``, ``data.labels``), so both studies of
the joint opening use the same labels; ``label_normalisation.csv`` is written
before any selection. The pinned Study 9 runner and analysis are unchanged.

Study 9 Amendment 4 (2026-10-28, before any HELD-OUT access): after the label
step and before any fit, the Study 8 A2.7 screen runs flag-only through
``study13.screen_all`` (``integrity_screen.csv``, flags in the access log).
No table is excluded and no column is dropped; the flags define the
sensitivities SENS-INT and SENS-ID, as in Study 13.
"""

import json
import os
import re
import subprocess
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simaudit.runner import study9, study13

SET_NAME = "heldout"
LEARNERS = ("fm2", "fm1", "fm5", "fm3")  # Study 9 Sec. 9: all four FMs
PREREG = Path("prereg") / "STUDY9_HOW_MANY_LABELS.md"
# Study 9 Freeze note 1 (commit R003).
PINNED = {
    Path("src/simaudit/runner/study9.py"): (
        "6fe40d008265fff82a64c8b98321774dc30736c075e89163acdcf176787dd4c0"
    ),
    Path("src/simaudit/analysis/study9.py"): (
        "aaf0bae1defa10d80b1db235614fe228ebf0aa26e82499d0d29dd51a3f6c50ba"
    ),
}


def _committed_unchanged(root: Path, rel: Path) -> bool:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=root,
        capture_output=True,
    )
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel)], cwd=root)
    return tracked.returncode == 0 and clean.returncode == 0


def check_study9_frozen(root: Path, pinned: Mapping[Path, str] = PINNED) -> None:
    """Raise unless the Study 9 prereg is FROZEN and committed and code is pinned."""
    prereg = root / PREREG
    status = re.search(r"^Status:\s*\**\s*(\w+)", prereg.read_text(), re.MULTILINE)
    if status is None or status.group(1) != "FROZEN":
        raise PermissionError(f"{PREREG} is not FROZEN")
    if not _committed_unchanged(root, PREREG):
        raise PermissionError(f"{PREREG} is not committed unchanged")
    for rel, expected in pinned.items():
        if study13.file_sha256(root / rel) != expected:
            raise PermissionError(f"{rel} differs from its Study 9 freeze hash")


def check_go(root: Path, env: Mapping[str, str] | None = None) -> Path:
    """All preconditions of the joint opening for Study 9 (module docstring)."""
    frozen13 = study13.check_heldout_go(root, env)
    check_study9_frozen(root)
    return frozen13


def heldout_pool(root: Path) -> dict[int, str]:
    """HELD-OUT IDs and content hashes from the non-sealed ``pool_i2.yaml``."""
    import yaml

    pool = yaml.safe_load((root / "data" / "pool_i2.yaml").read_text())
    return {int(d["did"]): str(d["sha256"]) for d in pool["heldout"]}


def screen_flags(
    ids: list[int], loader: study13.Loader, cache_dir: Path, out_dir: Path, log: Path
) -> dict[int, dict[str, Any]]:
    """Study 9 Amendment 4: the Study 13 A2.7 screen, flag-only, before any fit.

    Writes ``integrity_screen.csv`` (``study13.screen_all``) and logs each
    table's flags. No table is excluded and no column is dropped; the flags
    define the sensitivities SENS-INT (``flagged``) and SENS-ID (``flag_id``).
    """
    rows = study13.screen_all(ids, loader, cache_dir, out_dir)
    access = study13.AccessLog(log)
    for did, row in rows.items():
        access(
            out_dir / "integrity_screen.csv",
            "integrity screen (flag-only)",
            dict(
                did=did,
                flagged=str(row["flagged"]),
                flag_id=str(row["flag_id"]),
                decision=str(row["decision"]),
            ),
        )
    return rows


def _write_provenance(
    out_dir: Path, root: Path, ids: list[int], learners: list[str], base_seed: int
) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance_heldout.json").write_text(
        json.dumps(
            dict(
                driver="runner/study9_heldout.py",
                driver_sha256=study13.file_sha256(Path(__file__).resolve()),
                prereg=str(PREREG),
                prereg_sha256=study13.file_sha256(root / PREREG),
                pinned={str(k): v for k, v in PINNED.items()},
                git_head=head,
                base_seed=base_seed,
                seed_namespace=study9.SEED_NAMESPACE,
                perm_namespace=study9.PERM_NAMESPACE,
                grid=study9.GRID,
                reps=study9.REPS,
                learners=learners,
                datasets=ids,
                started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
            indent=2,
        )
    )


def _log_failure(out_dir: Path, job: tuple[Any, ...], exc: Exception) -> None:
    with (out_dir / "failures.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(job=list(job), error=repr(exc))) + "\n")
    print(f"FAILED {job}: {exc!r}", flush=True)


def run(
    root: Path,
    sealed: Path,
    out_dir: Path,
    cache_dir: Path,
    base_seed: int,
    workers: int,
    learners: list[str],
    *,
    env: Mapping[str, str] | None = None,
    job_kwargs: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Guard, verify the manifest, then run every (learner, table) job.

    ``job_kwargs`` (``predict``, ``embed``) are for tests on synthetic data.
    Failures are logged and the other jobs go on, as in ``study9.run``.
    """
    check_go(root, env)
    pool = heldout_pool(root)
    log_path = out_dir / "access_log.jsonl"
    study13.verify_manifest(sealed, sorted(pool), study13.AccessLog(log_path))
    loader = study13.SealedLoader(sealed, pool, log_path, normalise_labels=True)
    ids = sorted(pool)
    study13.label_log(ids, loader, out_dir)  # Study 9 Amendment 2, before any draw
    screen_flags(ids, loader, cache_dir, out_dir, log_path)  # Amendment 4
    _write_provenance(out_dir, root, ids, learners, base_seed)
    os.environ.setdefault("OMP_NUM_THREADS", str(study9.TORCH_THREADS))
    jobs = [(SET_NAME, did, lr) for lr in learners for did in ids]
    kwargs = dict(job_kwargs or {}, loader=loader)
    done: list[Path] = []
    if workers == 1:
        for job in jobs:
            try:
                done.append(
                    study9.run_job(*job, cache_dir, out_dir, base_seed, **kwargs)
                )
            except Exception as exc:  # noqa: BLE001 - logged; the rest go on
                _log_failure(out_dir, job, exc)
        return done
    with ProcessPoolExecutor(  # pragma: no cover - multiprocess path
        max_workers=workers, initializer=study9._init_worker
    ) as pool_exec:
        futures = {
            pool_exec.submit(
                study9.run_job, *job, cache_dir, out_dir, base_seed, loader=loader
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            try:
                done.append(future.result())
                print(f"done {futures[future]} ({len(done)}/{len(jobs)})", flush=True)
            except Exception as exc:  # noqa: BLE001 - logged; the rest go on
                _log_failure(out_dir, futures[future], exc)
    return done


def main() -> None:  # pragma: no cover - command-line entry
    """``python -m simaudit.runner.study9_heldout [--workers 4] [--learners ...]``."""
    import argparse

    from simaudit.settings import PROJECT_ROOT, settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learners", default=",".join(LEARNERS))
    args = parser.parse_args()
    check_go(PROJECT_ROOT)  # before anything sealed is touched
    run(
        PROJECT_ROOT,
        PROJECT_ROOT.parent / "sealed_heldout",
        settings.results_dir / "study9" / SET_NAME,
        settings.data_cache_dir,
        settings.base_seed,
        args.workers,
        args.learners.split(","),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
