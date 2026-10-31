"""Sensitivity of the review headline (A among few-shot studies with a stated draw) to the
few-shot definition (EXPLORATORY; read-only inputs).

The coding manual (01_CODING_MANUAL.md) codes few_shot = yes if the paper frames the setting as
few-shot / k-shot / cold start, OR any main reduced-label experiment uses <= 100 labelled rows in
total or <= 10 per class. The two parts are not coded separately, so we rebuild each from stored
fields:
  (a) numeric rule only: parse smallest_budget. "<k> per class" -> per class; "<k> total", "<k>
      shots", "<k> labeled", "<k> initial", "<k> queried" -> total. Fractions and unparsed entries
      do not meet the rule (they cannot be checked without each dataset's size); we report how many.
  (b) self-description only: the title or the first 4,000 characters of the extracted full text
      (title, abstract, start of the introduction) contain few-shot, k-shot, n-shot, low-shot,
      cold start or cold-start (case-insensitive). As a looser variant, the whole text.
Frame of the headline: category in A, A', B, C (the draw is described).
Inputs: research_notes/systematic_review_2026-10/coding/final_coding.csv and included_papers.csv,
downloads/text/<id>.txt (all read-only).
Writes out/r7_fewshot_def.csv, out/rv_r7fs.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r7_fewshot_def.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rv_common import OUT, Macros

AUD = Path(__file__).resolve().parents[5] / "research_notes" / "systematic_review_2026-10"
STATED = {"A", "A'", "B", "C"}
TERMS = re.compile(r"few[\s-]?shot|\b[kn][\s-]shot|low[\s-]shot|cold[\s-]start", re.I)


def numeric_rule(s: str) -> str:
    """'yes' / 'no' / 'unknown' for the <=100 total or <=10 per class rule."""
    s = s.lower().replace(",", "")
    if not s or "not stated" in s:
        return "unknown"
    nums = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%|per class|total|shots?|labeled|labelled|initial|queried)", s):
        k, unit = float(m.group(1)), m.group(2)
        if unit == "%":
            continue
        nums.append(k <= 10 if unit == "per class" else k <= 100)
    if not nums:
        return "unknown"
    return "yes" if any(nums) else "no"


def main() -> None:
    f = pd.read_csv(AUD / "coding" / "final_coding.csv", dtype=str).fillna("")
    inc = pd.read_csv(AUD / "coding" / "included_papers.csv", dtype=str).fillna("")
    f = f.merge(inc[["paper_id", "title"]], on="paper_id", how="left")
    assert len(f) == 161

    def text(pid: str) -> str:
        p = AUD / "downloads" / "text" / f"{pid}.txt"
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

    texts = {pid: text(pid) for pid in f.paper_id}
    f["n_missing_text"] = [texts[p] == "" for p in f.paper_id]
    f["num"] = f.smallest_budget.map(numeric_rule)
    f["self_head"] = [bool(TERMS.search(t + " " + texts[p][:4000])) for p, t in zip(f.paper_id, f.title)]
    f["self_full"] = [bool(TERMS.search(t + " " + texts[p])) for p, t in zip(f.paper_id, f.title)]
    f["fs"] = f.few_shot.str.lower().eq("yes")
    cat = f.category.str.strip()
    defs = {
        "Coded": f.fs,
        "Num": f.num.eq("yes"),
        "Self": f.self_head,
        "SelfFull": f.self_full,
        "Both": f.num.eq("yes") & f.self_head,
    }
    rows = []
    for name, mask in defs.items():
        frame = mask & cat.isin(STATED)
        rows.append(dict(definition=name, n_fewshot=int(mask.sum()), n_stated=int(frame.sum()),
                         A=int((frame & (cat == "A")).sum()), Ap=int((frame & (cat == "A'")).sum())))
    df = pd.DataFrame(rows)
    df["share_A"] = df.A / df.n_stated
    df.to_csv(OUT / "r7_fewshot_def.csv", index=False)
    coded = df.iloc[0]
    assert (coded.n_stated, coded.A) == (100, 40), coded  # reproduces \SysAuditFsCatA / \SysAuditFsStatedN

    m = Macros("r7fs")
    src = "r7_fewshot_def.py; research_notes/systematic_review_2026-10/coding/final_coding.csv"
    for _, r in df.iterrows():
        w = r.definition
        m.put(f"RsevFs{w}N", str(r.n_fewshot), src + f" ({w}: few-shot papers)")
        m.put(f"RsevFs{w}Stated", str(r.n_stated), src + f" ({w}: with a stated draw)")
        m.put(f"RsevFs{w}A", str(r.A), src + f" ({w}: category A)")
        m.put(f"RsevFs{w}Pct", f"{100 * r.share_A:.0f}\\%", src + f" ({w}: A share)")
    unk = f.num.eq("unknown")
    m.put("RsevFsNumUnknown", str(int(unk.sum())), src + " (budget given as a fraction or not parsed)")
    m.put("RsevFsNoText", str(int(f.n_missing_text.sum())), src + " (no extracted text)")
    m.write()
    print(df.round(3).to_string())
    print("numeric rule:", f.num.value_counts().to_dict(), "| missing text:", int(f.n_missing_text.sum()))
    print(pd.crosstab(f.fs, [f.num, f.self_head]))


if __name__ == "__main__":
    main()
