"""P1 protocol corrections (exploratory, DEV): new FM2 runs for Correction 4.

Plan: reports/P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md, Sec. 1.2-1.3. For each
multiclass DEV dataset and mult in {1, 3, 5, 10} (B = mult * C):

* ``design``: 100,000 multivariate-hypergeometric compositions are sorted by
  (number of missed classes, D = ||N/B - u||^2) and cut into K = 6 equal-mass
  strata; 4 compositions are drawn per stratum, each fitted on rows drawn
  uniformly within classes.
* ``random``: 40 extra realistic random contexts (uniform B-subsets).

Learner: FM2 (TabICL) through the Study 7 wrapper and scoring. Seed namespace
21. Outputs: ``results/p1_corrections/{kind}/{did}_fm2.parquet`` (atomic,
skipped when present), then ``results/p1_corrections/DONE``. DEV only.

Run: ``nice -n 15 uv run python reports/p1_corrections_runs.py --workers 4``
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.data.openml_loader import load
from simaudit.runner import study7
from simaudit.settings import settings

SEED_NAMESPACE = 21
MULTS = (1, 3, 5, 10)
N_MC = 100_000
N_STRATA = 6
REPS_PER_STRATUM = 4
EXTRA_RANDOM = 40
LEARNER = "fm2"
MULTICLASS_DEV = (42544, 40496, 47155, 694, 46336, 41511)
OUT_DIR = settings.results_dir / "p1_corrections"


def imbalance(compositions: NDArray[np.int64]) -> NDArray[np.float64]:
    """D = squared distance of the context shares from uniform, per row."""
    budget = compositions.sum(axis=1, keepdims=True)
    n_classes = compositions.shape[1]
    return np.asarray(((compositions / budget - 1 / n_classes) ** 2).sum(axis=1))


def composition_strata(
    counts: NDArray[np.int64], budget: int, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """MC compositions and their equal-mass stratum labels (0..N_STRATA-1)."""
    comps = rng.multivariate_hypergeometric(counts, budget, size=N_MC).astype(np.int64)
    missed = (comps == 0).sum(axis=1)
    order = np.lexsort((imbalance(comps), missed))
    labels = np.empty(N_MC, dtype=np.int64)
    labels[order] = np.arange(N_MC) * N_STRATA // N_MC
    return comps, labels


def design_contexts(y: NDArray[np.int64], did: int, mult: int) -> list[dict[str, Any]]:
    """The (stratum, rep, composition, seed) designs of one (dataset, mult)."""
    counts = np.bincount(y).astype(np.int64)
    budget = mult * len(counts)
    rng = np.random.default_rng([settings.base_seed, did, mult, SEED_NAMESPACE, 0])
    comps, labels = composition_strata(counts, budget, rng)
    designs = []
    for stratum in range(N_STRATA):
        members = np.flatnonzero(labels == stratum)
        for rep, pick in enumerate(rng.choice(members, REPS_PER_STRATUM, False)):
            seed = np.random.SeedSequence(
                [settings.base_seed, did, mult, stratum, rep, SEED_NAMESPACE, 1]
            ).generate_state(1)[0] % (2**31 - 1)
            designs.append(
                dict(
                    stratum=stratum,
                    rep=rep,
                    counts=comps[pick].tolist(),
                    seed=int(seed),
                    weight=1 / N_STRATA,
                )
            )
    return designs


def random_contexts(n: int, did: int, mult: int, budget: int) -> list[dict[str, Any]]:
    """EXTRA_RANDOM uniform B-subsets (realistic protocol)."""
    out = []
    for rep in range(EXTRA_RANDOM):
        seed = np.random.SeedSequence(
            [settings.base_seed, did, mult, rep, SEED_NAMESPACE, 2]
        ).generate_state(1)[0] % (2**31 - 1)
        rows = np.random.default_rng(int(seed)).choice(n, budget, replace=False)
        out.append(dict(rep=rep, seed=int(seed), labeled=np.sort(rows).tolist()))
    return out


def run_job(kind: str, did: int) -> Path:
    """Fit FM2 on every context of one (kind, dataset); skipped if done."""
    out = OUT_DIR / kind / f"{did}_{LEARNER}.parquet"
    if out.exists():
        return out
    import torch

    torch.set_num_threads(1)
    data = load(did, settings.data_cache_dir)
    X, y = data.X, data.y
    n_classes = int(y.max()) + 1
    predict = study7.predictor(LEARNER)
    rows = []
    for mult in MULTS:
        if kind == "design":
            contexts = design_contexts(y, did, mult)
            for ctx in contexts:
                ctx["labeled"] = study7.draw_context(
                    y, np.asarray(ctx["counts"], dtype=np.int64), ctx["seed"]
                ).tolist()
        else:
            contexts = random_contexts(len(y), did, mult, mult * n_classes)
        for ctx in contexts:
            labeled = np.asarray(ctx.pop("labeled"), dtype=np.int64)
            _, record = study7._fit(predict, X, y, labeled, ctx["seed"])
            rows.append(
                dict(
                    dataset=did,
                    learner=LEARNER,
                    kind=kind,
                    mult=mult,
                    **ctx,
                    budget=len(labeled),
                    **record,
                )
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(kind, did) for kind in ("design", "random") for did in MULTICLASS_DEV]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, *job): job for job in jobs}
        for future in as_completed(futures):
            print(f"done {futures[future]} -> {future.result()}", flush=True)
    (OUT_DIR / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
