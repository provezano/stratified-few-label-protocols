"""(a) Review item 2: subtypes of the label-using draws among stated-draw few-shot papers.

Usage: python a_review_subtypes.py [CODING_CSV]
Default CODING_CSV = ../human_check/data/final_coding.csv (the final review coding; a byte-identical
copy of the internal coding file, sha256 in human_check/ADJUDICATION.md).
A' subtypes come from a_subtype_lookup.csv (model coding of the stored quotes/notes);
an A' paper missing from the lookup is counted as 'unsubtyped'. EXPLORATORY.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rv_common import HERE, OUT, Macros

DEFAULT = HERE.parent / "human_check" / "data" / "final_coding.csv"
ROOT = HERE.parents[3]  # simaudit/
SUBS = ("prop", "prop_min", "min_fill", "reject", "other", "unsubtyped")
SUBW = dict(prop="Prop", prop_min="PropMin", min_fill="MinFill", reject="Reject", other="Other",
            unsubtyped="Unsub")


def norm_cat(c: str) -> str:
    c = str(c).strip().replace("′", "'").replace("’", "'")
    return c.upper().replace("A'", "A'")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(path) -> None:
    fc = pd.read_csv(path, keep_default_na=False)
    fc["cat"] = fc.category.map(norm_cat)
    fs = fc[fc.few_shot.str.strip().str.lower().eq("yes")]
    st = fs[fs.cat.isin(["A", "A'", "B", "C"])].copy()
    look = pd.read_csv(HERE / "a_subtype_lookup.csv")
    st = st.merge(look[["paper_id", "subtype"]], on="paper_id", how="left")
    st.loc[st.cat != "A'", "subtype"] = ""
    st.loc[(st.cat == "A'") & st.subtype.isna(), "subtype"] = "unsubtyped"
    n = len(st)
    rows = []
    for lab, k in [("A", (st.cat == "A").sum()), ("A'", (st.cat == "A'").sum()),
                   ("B", (st.cat == "B").sum()), ("C", (st.cat == "C").sum())] + \
                  [(f"A':{s}", (st.subtype == s).sum()) for s in SUBS]:
        lo, hi = wilson(int(k), n)
        rows.append(dict(group=lab, k=int(k), n=n, share=k / n, wilson_lo=lo, wilson_hi=hi))
    tab = pd.DataFrame(rows)
    path = Path(path).resolve()
    tab["coding_file"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    tab.to_csv(OUT / "a_subtypes.csv", index=False)
    st[["paper_id", "cat", "subtype", "coverage_forced", "quote"]].to_csv(OUT / "a_papers.csv", index=False)
    # coverage forced within each subtype
    cov = st[st.cat == "A'"].groupby("subtype").coverage_forced.value_counts().unstack(fill_value=0)
    cov.to_csv(OUT / "a_subtype_coverage.csv")
    print(tab.to_string(index=False))
    print(cov)

    mac = Macros("a")
    src = f"a_subtypes.csv ({path.name}, few-shot, stated draw)"
    get = dict(zip(tab.group, tab.k))
    mac.put("RevStatedN", str(n), src)
    mac.put("RevCatA", str(get["A"]), src)
    mac.put("RevCatAprime", str(get["A'"]), src)
    mac.put("RevCatB", str(get["B"]), src)
    mac.put("RevCatC", str(get["C"]), src)
    for s in SUBS:
        mac.put(f"RevAprime{SUBW[s]}", str(get[f"A':{s}"]), src + f", A' subtype {s} (model-coded)")
    cov_only = get["A':min_fill"] + get["A':reject"]
    mac.put("RevAprimeCovOnly", str(cov_only), src + ", min_fill + reject")
    mac.put("RevAprimePropAll", str(get["A':prop"] + get["A':prop_min"]), src + ", prop + prop_min")
    a = get["A"]
    lo, hi = wilson(a, n)
    mac.put("RevCatALo", f"{lo:.2f}", src + ", Wilson 95%")
    mac.put("RevCatAHi", f"{hi:.2f}", src + ", Wilson 95%")
    mac.write()


if __name__ == "__main__":
    from pathlib import Path
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
