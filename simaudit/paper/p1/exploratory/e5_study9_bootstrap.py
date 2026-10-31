"""E5 (exploratory): Study 9 analysis with the pre-registered number of bootstrap resamples.

The prereg (STUDY9_HOW_MANY_LABELS.md, Sec. 3 H1-dist) fixes 2000 dataset-bootstrap
resamples. The frozen ``analysis.study9.h1`` calls ``distribution_summary`` with
its default ``n_boot = 500`` (H1-dist for every learner x strategy and the
selection-ceiling distributions). Every other bootstrap in the module
(``boot_ci``: ICC, H2a magnitudes; ``mae_ratio``: H4) already uses N_BOOT = 2000.

This script runs the frozen ``analyze`` twice on the stored results, unchanged
(500) and with ``distribution_summary`` bound to n_boot = 2000 (monkeypatched in
this process only; src/ is not edited), for T1' (the confirmatory set) and DEV
(dry-run), and diffs every quantity. The SHA-256 of the frozen module is checked
first. Outputs only under paper/p1/exploratory/out/.
"""

from __future__ import annotations

import hashlib
import sys
from functools import partial
from math import inf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import common as cm  # noqa: E402
from simaudit.analysis import study9 as a9  # noqa: E402
from simaudit.runner.study9 import dataset_key  # noqa: E402

FROZEN = "aaf0bae1defa10d80b1db235614fe228ebf0aa26e82499d0d29dd51a3f6c50ba"
ORIG = a9.distribution_summary


def keys_for(set_name: str) -> dict[str, int]:
    if set_name == "dev":
        return {dataset_key(d): d for d in cm.dev_ids()}
    rows = yaml.safe_load((cm.PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]
    return {dataset_key(int(r["did"])): int(r["did"]) for r in rows}


def flatten(res: dict) -> dict[str, float]:
    out: dict[str, float] = {}

    def walk(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}/{k}", v)
        elif isinstance(obj, (tuple, list)) and all(isinstance(x, (int, float, np.floating, np.integer)) for x in obj):
            for i, v in enumerate(obj):
                out[f"{prefix}[{i}]"] = float(v)
        elif isinstance(obj, (int, float, np.floating, np.integer, bool, np.bool_)):
            out[prefix] = float(obj)
        elif isinstance(obj, pd.DataFrame):
            for (r, c), v in obj.stack().items():
                if isinstance(v, (int, float, np.floating)):
                    out[f"{prefix}/{r}/{c}"] = float(v)

    walk("", res)
    return out


def run(set_name: str) -> tuple[pd.DataFrame, str, str]:
    keys = keys_for(set_name)
    raw = a9.load_results(cm.settings.results_dir / "study9" / set_name)
    keys = {k: d for k, d in keys.items() if k in set(raw.dataset)}
    traits = a9.dataset_traits(keys, cm.dev_data if set_name == "dev" else
                               (lambda did: cm.load(did, cm.settings.data_cache_dir)), cm.settings.base_seed)
    raw = raw[raw.dataset.isin(keys)]
    a9.distribution_summary = ORIG
    r500 = a9.analyze(raw, traits)
    a9.distribution_summary = partial(ORIG, n_boot=a9.N_BOOT)
    r2000 = a9.analyze(raw, traits)
    a9.distribution_summary = ORIG
    f5, f2 = flatten(r500), flatten(r2000)
    rows = []
    for k in sorted(set(f5) | set(f2)):
        a, b = f5.get(k, np.nan), f2.get(k, np.nan)
        same = (a == b) or (np.isnan(a) and np.isnan(b)) or (np.isinf(a) and a == b)
        rows.append(dict(set=set_name, quantity=k, boot500=a, boot2000=b, diff=b - a if not same else 0.0,
                         changed=not same))
    title = f"Study 9 analysis, set {set_name} ({len(keys)} datasets)"
    return pd.DataFrame(rows), a9.report(r500, title + ", frozen n_boot=500"), a9.report(r2000, title + ", n_boot=2000")


def main() -> None:
    got = hashlib.sha256((cm.PROJECT_ROOT / "src/simaudit/analysis/study9.py").read_bytes()).hexdigest()
    if got != FROZEN:
        raise SystemExit("analysis/study9.py differs from freeze note 1")
    frames = []
    for set_name in ("t1prime", "dev"):
        df, rep500, rep2000 = run(set_name)
        frames.append(df)
        (cm.OUT / f"e5_study9_{set_name}_boot500.md").write_text(rep500)
        (cm.OUT / f"e5_study9_{set_name}_boot2000.md").write_text(rep2000)
    diff = pd.concat(frames, ignore_index=True)
    diff.to_csv(cm.OUT / "e5_diff.csv", index=False)
    ch = diff[diff.changed]
    # reproduction of the committed Part A dist table (500) for T1'
    committed = (cm.PROJECT_ROOT / "reports" / "STUDY9_T1PRIME_OUTPUT.md").read_text()
    ours = (cm.OUT / "e5_study9_t1prime_boot500.md").read_text()
    dist_lines = [ln for ln in ours.splitlines() if ln.startswith("| fm")]
    reproduced = sum(ln in committed for ln in dist_lines)
    ci = ch[ch.quantity.str.contains(r"\[1\]|\[2\]")]
    L = ["# E5: Study 9 with the pre-registered 2000 bootstrap resamples (exploratory re-analysis)", "",
         "Frozen `analysis/study9.py` (SHA-256 checked) run twice on the stored results: as frozen (H1-dist and "
         "selection-ceiling CIs with `distribution_summary`'s default 500 resamples) and with n_boot = 2000 "
         "(Sec. 3 of the prereg). Every other bootstrap in the module already used 2000. Nothing in reports/ or "
         "results/ is touched.", "",
         f"- Reproduction: {reproduced} of {len(dist_lines)} H1-dist rows of the 500-resample T1' run are verbatim in "
         "reports/STUDY9_T1PRIME_OUTPUT.md (Part A).",
         f"- Quantities compared: {len(diff)}; changed: {len(ch)}, all of them bootstrap CI bounds "
         f"({len(ci)} CI-bound entries; {int(ch.quantity.str.contains('dist|ceiling').sum())} in H1-dist/ceiling).",
         f"- Largest absolute change of a CI bound: {ch['diff'].abs().max():.3f}; median {ch['diff'].abs().median():.3f}.",
         f"- Point estimates, medians/IQR cells, AFT, ICC, H2, H3, H4 and the Holm family: "
         f"{'unchanged' if not ch.quantity.str.contains('family|aft|icc|h2|h3|h4').any() else 'CHANGED — see e5_diff.csv'}.", ""]
    for set_name in ("t1prime", "dev"):
        s = ch[ch.set == set_name].copy()
        s["abs"] = s["diff"].abs()
        L += [f"## {set_name}: changed quantities (largest 25)", "",
              s.sort_values("abs", ascending=False).head(25)[["quantity", "boot500", "boot2000", "diff"]].round(4).to_markdown(index=False), ""]
    prim = ch[(ch.set == "t1prime") & ch.quantity.str.contains("dist\\|fm2\\|random")]
    L += ["## Primary H1-dist (T1', FM2, random)", "", prim[["quantity", "boot500", "boot2000"]].round(3).to_markdown(index=False), ""]
    (cm.OUT / "E5_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
