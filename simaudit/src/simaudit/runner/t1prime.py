"""Study 8 Amendment 2 (A2.5): the OpenML-free confirmation set T1'.

T1' is drawn from the frozen PMLB tiers (2a, 3) and the UCI Tier 4 with
Tier 1's quotas, seed rule and reallocation (Sec. 4-5). Within a stratum the
numeric-only tiers (2a, 4) are queued before Tier 3.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from simaudit.runner import build_pool_ext as ext
from simaudit.runner.screen_integrity import read_screens

Screens = dict[str, tuple[str, tuple[int, ...]]]

SEED = (*ext.SEED, 2)
NUMERIC = frozenset({"2a", "4"})
TARGET = ext.QUOTA * len(ext.STRATA)

Entry = dict[str, Any]


def queues(entries: Sequence[Entry]) -> dict[str, list[str]]:
    """Per stratum: numeric keys (sorted, permuted), then Tier 3 keys likewise."""
    rng = np.random.default_rng(list(SEED))
    out: dict[str, list[str]] = {}
    for s in ext.STRATA:
        out[s] = []
        for numeric in (True, False):
            keys = sorted(
                str(e["key"])
                for e in entries
                if e["stratum"] == s and (e["subtier"] in NUMERIC) == numeric
            )
            out[s] += [str(k) for k in rng.permutation(np.array(keys, dtype=object))]
    return out


def assemble(
    entries: Sequence[Entry], screens: Screens | None = None
) -> dict[str, Any]:
    """Draw T1' (quota 8 per stratum, reallocation to 72) and its balanced core.

    Datasets the integrity screens exclude (A2.6) never enter a queue; the
    columns they drop are recorded per member.
    """
    screens = screens or {}
    excluded = sorted(
        str(e["key"])
        for e in entries
        if screens.get(str(e["key"]), ("keep", ()))[0].startswith("exclude")
    )
    entries = [e for e in entries if str(e["key"]) not in excluded]
    q = queues(entries)
    balanced = {k for s in ext.STRATA for k in q[s][: ext.QUOTA]}
    ids = {str(e["key"]): i for i, e in enumerate(entries)}
    int_queues = {s: [ids[k] for k in q[s]] for s in ext.STRATA}
    picked, _ = ext.draw(int_queues, ext.QUOTA, lambda did, s: None)
    chosen = [entries[i] for s in ext.STRATA for i in picked[s]]
    datasets = [
        dict(
            key=e["key"],
            did=e["did"],
            name=e["name"],
            subtier=e["subtier"],
            stratum=e["stratum"],
            n=e["n"],
            numeric=e["subtier"] in NUMERIC,
            balanced=e["key"] in balanced,
            drop_columns=list(screens.get(str(e["key"]), ("keep", ()))[1]),
        )
        for e in chosen
    ]
    counts = {
        s: dict(
            available=len(q[s]),
            t1prime=len(picked[s]),
            balanced=min(len(q[s]), ext.QUOTA),
            quota=ext.QUOTA,
        )
        for s in ext.STRATA
    }
    return dict(
        seed=list(SEED),
        quota=ext.QUOTA,
        target=TARGET,
        heldout_signature_check="pending",
        integrity_excluded=excluded,
        n_t1prime=len(datasets),
        n_balanced=sum(d["balanced"] for d in datasets),
        sum_n=sum(int(d["n"]) for d in datasets),
        counts=counts,
        datasets=datasets,
    )


def build_t1prime(
    pmlb_pool: Path, uci_pool: Path, out_path: Path, integrity: Path | None = None
) -> dict[str, Any]:
    """Assemble T1' from the two frozen pool files and write it as YAML."""
    entries = [
        e
        for path in (pmlb_pool, uci_pool)
        for e in yaml.safe_load(path.read_text())["datasets"]
    ]
    screens = read_screens(integrity) if integrity else None
    result = assemble(entries, screens)
    out_path.write_text(yaml.safe_dump(result, sort_keys=False))
    return cast(dict[str, Any], yaml.safe_load(out_path.read_text()))
