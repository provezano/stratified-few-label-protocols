"""EXPLORATORY: fm8 LimiX-16M on T1' with Study 12's draws (plan: limix_t1p/PLAN.md).

Identical to ``run_t1p.py`` (fm6 TabDPT / fm7 Mitra-v2 on T1'): same 52 tables, row
permutation, stored Study 12 stratified/random selections and seeds, panel E = first
min(n, 2000) permuted rows, metrics on E \\ L, missing classes as in
``runner.study9.predict_on``. The only differences:

- learner = ``adapters.limix_predict`` (LimiX-16M @ 3a7c18ac, default no-retrieval
  classification config, fp32, CPU, one torch thread), exactly the DEV wrapper of
  ``out/s7a/*_fm8.parquet``;
- the checkpoint is loaded with ``torch.load(..., weights_only=True)`` (the DEV run used
  ``weights_only=False``; the file is a plain ``{"state_dict", "config"}`` dict, so the
  tensors are identical);
- outputs go to ``out/t1p_fm8/`` (not ``out/t1p/``), so the fm6/fm7 summary is untouched.

Usage (from simaudit/, in the LimiX venv):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
      PYTHONDONTWRITEBYTECODE=1 nice -n 15 .venv_fm_extra_limix/bin/python \
      paper/p1/exploratory/fm_extra/run_t1p_fm8.py --workers 6 [--datasets k1,k2]

Resumable and stoppable exactly as run_t1p.py (``out/t1p_fm8/STOP``). Writes
``out/t1p_fm8/DONE`` when every requested table has finished (or failed; see log).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adapters as ad  # noqa: E402
import run_t1p as base  # noqa: E402

# Output dir: env FM8_OUT_NAME (default t1p_fm8; the timing pilot used t1p_fm8_pilot).
# Set via env so spawned workers see it at import time.
OUT = HERE / "out" / os.environ.get("FM8_OUT_NAME", "t1p_fm8")
base.OUT = OUT  # run_job / stop_requested read the module global


@lru_cache(maxsize=1)
def _limix_ckpt_safe():  # type: ignore[no-untyped-def]
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ad.LIMIX_REPO, ad.LIMIX_FILE, revision=ad.LIMIX_REVISION,
                           local_files_only=True)  # no download: weights already cached
    return path, torch.load(path, map_location="cpu", weights_only=True)


ad._limix_ckpt = _limix_ckpt_safe
_orig_predictor = base.predictor
base.predictor = lambda fm: ad.limix_predict if fm == "fm8" else _orig_predictor(fm)


def run_job(did: int, min_free_gb: float) -> str:
    return base.run_job("fm8", did, min_free_gb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default="", help="comma-separated keys (default: all 52)")
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DONE").unlink(missing_ok=True)
    tables = base.t1prime()
    if a.datasets:
        wanted = set(a.datasets.split(","))
        tables = [t for t in tables if base.dataset_key(t[0]) in wanted]
    sizes = {did: len(base.load(did, base.settings.data_cache_dir).y) for did, _ in tables}
    tables.sort(key=lambda t: -min(sizes[t[0]], base.PANEL_MAX) * sizes[t[0]])
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_job, did, a.min_free_gb): did for did, _ in tables}
        for future in as_completed(futures):
            did = futures[future]
            try:
                print(f"fm8 {future.result()} [{time.strftime('%H:%M:%S')}]", flush=True)
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                print(f"FAILED fm8 {base.dataset_key(did)}: {exc!r}", flush=True)
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
