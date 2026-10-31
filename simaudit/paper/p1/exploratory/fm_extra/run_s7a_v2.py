"""DEV Part (a) re-run of fm7 (Mitra-v2) with per-fit seeding (s7a_v2).

Identical to ``run_s7a.py`` (same stored labelled sets, seeds, protocol and
adapter settings) except that AutoGluon's own ``set_seed(seed)`` is called with
the cell seed before every Mitra fit, as in the Study 13 worker
(``src/simaudit/runner/fm_extra_worker.py``, Study 13 Amendment 1 (B.2)).
AutoGluon 1.6 no longer seeds per fit, so the original s7a fm7 fits drew
Mitra's random preprocessing from the unseeded global numpy RNG and are not
reproducible. The old outputs in ``out/s7a`` are kept unchanged.

Usage (from simaudit/, in the Mitra venv, one process = one worker):
    OMP_NUM_THREADS=1 PYTHONPATH=src nice -n 10 .venv_fm_extra_mitra/bin/python \
      paper/p1/exploratory/fm_extra/run_s7a_v2.py [--datasets 1510,42544]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adapters as ad  # noqa: E402
import run_s7a as base  # noqa: E402

OUT = HERE / "out" / "s7a_v2"


def seeded_mitra(X_fit, y_fit, X_query, seed):  # type: ignore[no-untyped-def]
    """``adapters.mitra_predict`` after AutoGluon's ``set_seed(seed)``."""
    from autogluon.tabular.models.mitra._internal.utils.set_seed import set_seed

    set_seed(int(seed))
    return ad.mitra_predict(X_fit, y_fit, X_query, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    a = ap.parse_args()
    ad.PREDICTORS["fm7"] = ("Mitra-v2 (seeded)", seeded_mitra)
    dids = [int(x) for x in a.datasets.split(",")] if a.datasets else base.dev_ids()
    for did in dids:
        t = time.perf_counter()
        try:
            base.run_table("fm7", did, base.CORE, OUT)
            print(f"done fm7 {did} {time.perf_counter() - t:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - logged, next table goes on
            print(f"FAILED fm7 {did}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
