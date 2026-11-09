"""Freeze the frame: plan/frame.csv (eligible rows of harvest/curation/build_report.csv), plan/units.txt (table x FM,
largest estimated cost first), plan/predictions.csv (code/predict.py), and Section 9 of plan/PLAN.txt.
Content hash of each npz is re-checked against the build report. No model is fitted."""
import csv
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, "simaudit/src")
from simaudit.data.openml_loader import content_hash  # noqa: E402

OFF = {"zenodo": 10**10, "figshare": 2 * 10**10, "kaggle": 3 * 10**10, "dataverse": 4 * 10**10, "mendeley": 5 * 10**10}
FMS = ("fm3", "fm5", "fm1", "fm2")  # cost order (pilot shares 0.43 / 0.27 / 0.21 / 0.09)
SHARE = dict(fm3=0.43, fm5=0.27, fm1=0.21, fm2=0.09)

rep = pd.read_csv(BASE / "harvest/curation/build_report.csv", dtype=str, keep_default_na=False)
el = rep[rep.excluded == ""].copy()
rows = []
for _, r in el.iterrows():
    rec = r.record
    num = int(rec) if rec.isdigit() else zlib.crc32(rec.encode())
    did = OFF[r.source.lower()] + num
    with np.load(BASE / "harvest/npz" / f"{r.key}.npz", allow_pickle=False) as z:
        h = content_hash(np.asarray(z["X"], dtype=np.float64), np.asarray(z["y"]).astype(np.int64))
    assert h == r.sha256, r.key
    rows.append(dict(key=r.key, did=did, sha256=h, class_counts=r.class_counts, n=int(r.n), d=int(r.d), C=int(r.C),
                     source=r.source, record=rec, first_public=r.first_public, first_public_how=r.first_public_how,
                     licence=r.licence, target=r.target, url=r.url))
fr = pd.DataFrame(rows).sort_values(["first_public", "source", "record"]).reset_index(drop=True)
assert fr.did.is_unique and fr.sha256.is_unique
fr.to_csv(BASE / "plan/frame.csv", index=False)
fr["est_s"] = 9.9 * fr.n + 2408
units = sorted(((r.est_s * SHARE[fm], r.key, fm) for _, r in fr.iterrows() for fm in FMS), reverse=True)
(BASE / "plan/units.txt").write_text("".join(f"{k} {fm}\n" for _, k, fm in units))
subprocess.run([sys.executable, "-I", str(HERE / "predict.py")], check=True, env=dict(os.environ),
               stdout=subprocess.DEVNULL)
pr = pd.read_csv(BASE / "plan/predictions.csv")
t = fr.merge(pr[["key", "r", "V", "qmc_bar", "pred_pre", "pred_pre92"]], on="key")
tot_h = fr.est_s.sum() / 3600
lines = [f"{len(fr)} tables; sources: " + ", ".join(f"{k} {v}" for k, v in fr.source.value_counts().items())
         + f"; C = 2: {(fr.C == 2).sum()}, C ≥ 3: {(fr.C >= 3).sum()}; n {fr.n.min()}–{fr.n.max()} (median "
         f"{int(fr.n.median())}); first public {fr.first_public.min()} to {fr.first_public.max()}. Imbalanced (r ≤ 0.6): "
         f"{(t.r <= 0.6).sum()}.",
         "",
         f"Compute estimate (pilot cost model, arm T, four FMs): {tot_h:.0f} CPU-hours, {4 * len(fr)} units; largest "
         f"unit ≈ {units[0][0] / 3600:.1f} h ({units[0][1]}, {units[0][2]}). With 14 jobs on host B and 4 local: about "
         f"{tot_h / 16:.0f}–{tot_h / 12:.0f} h of wall time.",
         "",
         f"Frozen predicted FM-mean gap (pre-data model): mean {t.pred_pre.mean():+.3f}, range {t.pred_pre.min():+.3f} to "
         f"{t.pred_pre.max():+.3f}.",
         "",
         "| # | key | source | record | first public (how verified) | licence | target | n | d | C | class counts | r | predicted gap |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in t.iterrows():
    lines.append(f"| {i + 1} | {r.key} | {r.source} | {r.record} | {r.first_public} ({r.first_public_how}) | {r.licence} | "
                 f"`{r.target}` | {r.n} | {r.d} | {r.C} | {r.class_counts.replace(';', '/')} | {r.r:.2f} | {r.pred_pre:+.3f} |")
plan = (BASE / "plan/PLAN.txt").read_text()
head = plan.split("## 9. Frame and predictions")[0]
(BASE / "plan/PLAN.txt").write_text(head + "## 9. Frame and predictions\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines[:6]))
