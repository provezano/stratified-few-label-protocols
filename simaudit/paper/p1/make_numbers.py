"""Generate numbers_p1.tex: every number quoted in the P1 draft.

Run from anywhere:  uv run python paper/p1/make_numbers.py

The script only READS the generated reports (which are themselves produced by
scripts from results/), research notes, and (for the reproducibility numbers)
the provenance files and the per-fit "seconds" column of results/. It never touches prereg/, reports/,
results/ or sealed_heldout/. Each value is parsed from its source file; the few
values that are impractical to parse are constants with a comment that gives
their source file and line.

Macro names contain letters only (LaTeX rule), so digits are spelled out.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[1]  # .../simaudit
REP = SIM / "reports"

OUT: dict[str, str] = {}
SOURCES: dict[str, str] = {}

DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def macro_name(raw: str) -> str:
    name = "".join(DIGITS.get(c, c) for c in raw)
    name = re.sub(r"[^A-Za-z]", "", name)
    if not name:
        raise ValueError(raw)
    return name


def put(name: str, value, src: str) -> None:
    name = macro_name(name)
    if name in OUT:
        raise KeyError(f"duplicate macro {name}")
    OUT[name] = str(value)
    SOURCES[name] = src


def f(x: float, nd: int = 3, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}\\%"


def pfmt(p: float) -> str:
    """p-value for text: scientific below 1e-3."""
    if p == 0:
        return "$<10^{-300}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def section(text: str, start: str, end_pat: str = r"\n## ") -> str:
    i = text.index(start)
    m = re.search(end_pat, text[i + len(start):])
    return text[i: i + len(start) + (m.start() if m else len(text))]


def md_rows(block: str) -> list[list[str]]:
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def num(s: str) -> float:
    s = s.replace("−", "-").replace("*", "").strip()
    return float(s)


CI_RE = re.compile(r"([+\-−]?\d*\.\d+)\s*\[([+\-−]?\d*\.\d+),\s*([+\-−]?\d*\.\d+)\]")

# ---------------------------------------------------------------- Study 7 ---
S7 = REP / "STUDY7_REPORT.md"
s7 = read(S7)
FM_NAME = {"fm1": "FMone", "fm2": "FMtwo", "fm3": "FMthree", "fm5": "FMfive"}


def study7_family(block: str, prefix: str, src: str) -> None:
    for r in md_rows(block):
        cid = r[0]
        if not cid.startswith("(") or len(r) < 10:
            continue
        key = {"(i)-fm1": "IFMone", "(i)-fm2": "IFMtwo", "(i)-fm3": "IFMthree", "(i)-fm5": "IFMfive",
               "(ii-a)": "IIa", "(ii-b)": "IIb", "(ii-c)": "IIc"}.get(cid)
        holm = num(r[8])
        if key is None:
            # dose-response rows: (iii)-fm1-majority
            m = re.match(r"\(iii\)-(fm\d)-(majority|minority)", cid)
            if m and prefix == "Ssev":
                k = f"Ssev{FM_NAME[m.group(1)]}{m.group(2).title()}"
                put(k + "JTz", re.search(r"z=([\d.]+)", r[2]).group(1), src)
                put(k + "Holm", pfmt(holm), src)
            continue
        mean, lo, hi = (num(x) for x in CI_RE.search(r[3]).groups())
        put(prefix + key + "Mean", f(mean, sign=True), src)
        put(prefix + key + "Lo", f(lo, sign=True), src)
        put(prefix + key + "Hi", f(hi, sign=True), src)
        put(prefix + key + "Win", f(num(r[5]), 2), src)
        put(prefix + key + "Holm", pfmt(holm), src)
        put(prefix + key + "Tost", f(num(r[6]), 2), src)
        b = r[9].split("/")
        if len(b) == 3:
            put(prefix + key + "Rope", f(num(b[1]), 2), src)


fam = section(s7, "## Confirmatory contrasts")
study7_family(fam, "Ssev", f"{S7.name} (confirmatory table)")
# max Holm p of (i) contrasts
holms = [num(r[8]) for r in md_rows(fam) if r[0].startswith("(i)-")]
put("SsevIMaxHolm", pfmt(max(holms)), S7.name)
means_i = [num(CI_RE.search(r[3]).group(1)) for r in md_rows(fam) if r[0].startswith("(i)-")]
put("SsevIMinMean", f(min(means_i), sign=True), S7.name)
put("SsevIMaxMean", f(max(means_i), sign=True), S7.name)
wins = [num(r[5]) for r in md_rows(fam) if r[0].startswith("(i)-")]
put("SsevIMinWin", f(min(wins), 2), S7.name)
put("SsevIMaxWin", f(max(wins), 2), S7.name)
jtp = [num(r[8]) for r in md_rows(fam) if r[0].startswith("(iii)")]
put("SsevIIIMaxHolm", pfmt(max(jtp)), S7.name)
# share of the reversal explained by the balance option
iia = num(CI_RE.search([r for r in md_rows(fam) if r[0] == "(ii-a)"][0][3]).group(1))
iib = num(CI_RE.search([r for r in md_rows(fam) if r[0] == "(ii-b)"][0][3]).group(1))
put("SsevReversalSharePrior", pct(1 - iib / iia), S7.name + " (1 - (ii-b)/(ii-a))")

# sensitivity families
for tag, start in [("SsevSOne", "### S1:"), ("SsevSTwo", "### S2:"), ("SsevSFour", "### S4:")]:
    blk = section(s7, start, r"\n### ")
    study7_family(blk, tag, f"{S7.name} ({start})")

# per-budget contrasts
for metric, tag in [("macro_f1", "F"), ("accuracy", "Acc"), ("balanced_accuracy", "BA")]:
    blk = section(s7, f"## Per-budget contrasts, {metric} ")
    for r in md_rows(blk)[1:]:
        name = r[0]
        m = re.match(r"strat − random (fm\d)(_bal)?$", name)
        m2 = re.match(r"(fm\d)(_bal)? − (fm\d)(_bal)?, (random|stratified)$", name)
        for j, mult in enumerate([1, 3, 5, 10]):
            val = num(r[j + 1].split("(")[0])
            pv = num(re.search(r"\(p=([\d.]+)", r[j + 1]).group(1))
            words = {1: "One", 3: "Three", 5: "Five", 10: "Ten"}[mult]
            if m:
                k = f"SsevGap{tag}{FM_NAME[m.group(1)]}{'Bal' if m.group(2) else ''}M{words}"
                put(k, f(val, sign=True), S7.name)
                put(k + "P", f(pv, 3), S7.name)
            elif m2 and metric == "macro_f1":
                k = (f"SsevDiff{FM_NAME[m2.group(1)]}{'Bal' if m2.group(2) else ''}"
                     f"{FM_NAME[m2.group(3)]}{'Bal' if m2.group(4) else ''}{m2.group(5).title()}M{words}")
                put(k, f(val, sign=True), S7.name)
                put(k + "P", f(pv, 3), S7.name)

# range of the 1C and 10C macro-F1 gap over the four FMs
blk = section(s7, "## Per-budget contrasts, macro_f1 ")
g1, g10 = [], []
for r in md_rows(blk)[1:]:
    if re.match(r"strat − random fm\d$", r[0]):
        g1.append(num(r[1].split("(")[0]))
        g10.append(num(r[4].split("(")[0]))
put("SsevGapOneMin", f(min(g1), sign=True), S7.name)
put("SsevGapOneMax", f(max(g1), sign=True), S7.name)
put("SsevGapTenMin", f(min(g10), sign=True), S7.name)
put("SsevGapTenMax", f(max(g10), sign=True), S7.name)
blk = section(s7, "## Per-budget contrasts, accuracy ")
accp = []
for r in md_rows(blk)[1:]:
    if re.match(r"strat − random fm\d$", r[0]):
        accp += [num(re.search(r"\(p=([\d.]+)", c).group(1)) for c in r[1:5]]
put("SsevAccGapMinP", f(min(accp), 3), S7.name + " (min unadjusted p, accuracy, 4 FMs x 4 budgets)")

# rank reversal
rr = section(s7, "## Rank reversal between protocols")
mf = rr.split("### accuracy")[0]
taus = [float(t.replace("−", "-")) for t in re.findall(r"Kendall τ = ([−\-]?\d\.\d+)", mf)]
put("SsevTauMin", f(min(taus), 2), S7.name)
put("SsevTauMax", f(max(taus), 2), S7.name)
fl = re.findall(r"flip winner: (\d+)/(\d+) = ([\d.]+)%", rr)
put("SsevFlipF", f"{fl[0][2]}\\%", S7.name)
put("SsevFlipFNum", fl[0][0], S7.name)
put("SsevFlipFDen", fl[0][1], S7.name)
put("SsevFlipAcc", f"{fl[1][2]}\\%", S7.name)

# balance option check
dp = re.findall(r"max \|Δp\| = ([\d.e\-]+)", s7)
mx = max(float(x) for x in dp)
m_, e_ = f"{mx:.1e}".split("e")
put("SsevBalMaxDp", f"${m_}\\times10^{{{int(e_)}}}$", S7.name)

# Part (c) dose-response table
pc = section(s7, "## Part (c): dose-response", r"\n#### ")
drops = {}
for r in md_rows(pc)[1:]:
    lr, arm = r[0], r[1]
    drop, pcdrop = num(r[2]), num(r[8])
    drops[(lr, arm)] = (drop, pcdrop, r[12])
    k = f"Ssev{FM_NAME[lr]}{arm.title()}"
    put(k + "Drop", f(drop), S7.name)
    put(k + "PcDrop", f(pcdrop), S7.name)
    put(k + "Removed", pct(1 - pcdrop / drop), S7.name + " (1 - pc_drop/drop)")
    put(k + "Verdict", r[12], S7.name)
for arm in ["majority", "minority"]:
    ds = [drops[(lr, arm)][0] for lr in FM_NAME]
    put(f"Ssev{arm.title()}DropMin", f(min(ds), 2), S7.name)
    put(f"Ssev{arm.title()}DropMax", f(max(ds), 2), S7.name)
rem_pfn = [1 - drops[(lr, a)][1] / drops[(lr, a)][0] for lr in ["fm1", "fm5"] for a in ["majority", "minority"]]
put("SsevPFNRemovedMin", pct(min(rem_pfn)), S7.name)
put("SsevPFNRemovedMax", pct(max(rem_pfn)), S7.name)
rem_min = [1 - drops[(lr, "minority")][1] / drops[(lr, "minority")][0] for lr in ["fm2", "fm3"]]
put("SsevICLMinorityRemovedMin", pct(min(rem_min)), S7.name)
put("SsevICLMinorityRemovedMax", pct(max(rem_min)), S7.name)
n_beyond = sum(v[2] == "beyond prior shift" for v in drops.values())
put("SsevNBeyond", n_beyond, S7.name)
put("SsevNUnresolved", sum(v[2] == "unresolved" for v in drops.values()), S7.name)
put("SsevNPriorShift", sum(v[2] == "prior shift" for v in drops.values()), S7.name)
pcj = [num(r[10]) for r in md_rows(pc)[1:]]
put("SsevPcJTMax", pfmt(max(pcj)) if max(pcj) > 0 else "$<10^{-57}$",
    S7.name + " (pc_jt_p column is rounded to 0; the bound 1e-57 is from the Reading, line 30)")

# exploratory within-random imbalance correlation
ex = section(s7, "## Exploratory (not pre-registered): context imbalance")
for r in md_rows(ex)[1:]:
    put(f"SsevRho{FM_NAME[r[0]]}", f(num(r[1]), 2), S7.name)

# C2 selectors (exploratory DEV): count significant cells at 3C and 5C
c2 = section(s7, "## C2 (exploratory): label-free selectors − random, macro_f1")
rows = md_rows(c2)[1:]
for j, words in [(1, "One"), (2, "Three"), (3, "Five"), (4, "Ten")]:
    sig = [r for r in rows if num(re.search(r"p=([\d.]+)", r[j]).group(1)) < 0.05]
    put(f"SsevCTwoSig{words}", len(sig), S7.name + " (C2 table, Holm p within FM x mult < 0.05)")
    if words == "One":
        sel1 = sorted({re.match(r"\('(\w+)'", r[0]).group(1) for r in sig})
        put("SsevCTwoSigOneSelectors", ", ".join(sel1).replace("lucos", "LUCoS"), S7.name + " (C2 table, 1C)")
put("SsevCTwoN", len(rows), S7.name)

# 694 with / without ID column
t694 = section(s7, "694 macro-F1 (mean over mult and reps)", r"\n### ")
infl = [num(r[1]) - num(r[2]) for r in md_rows(t694)[1:]]
put("SsevIDInflMin", f(min(infl), 2), S7.name)
put("SsevIDInflMax", f(max(infl), 2), S7.name)
# S3 row order
s3 = section(s7, "### S3: row order", r"\n#### accuracy")
vals, ps = [], []
for r in md_rows(s3)[1:]:
    if "permuted − original" in r[0]:
        for c in r[1:5]:
            vals.append(abs(num(c.split("(")[0])))
            ps.append(num(re.search(r"\(p=([\d.]+)", c).group(1)))
put("SsevSThreeMaxAbs", f(max(vals)), S7.name)
put("SsevSThreeMinP", f(min(ps), 3), S7.name)


# critical-difference mean ranks of the learners (Friedman), per protocol
for proto, tag in [("stratified", "Strat"), ("random", "Rand")]:
    ln = re.search(rf"- Learners, {proto}: Friedman p = ([\d.e\-]+); CD = ([\d.]+); ranks: ([^(]+)\(", s7)
    put(f"SsevCD{tag}FriedP", pfmt(float(ln.group(1))), S7.name)
    put("SsevCDWidth" if tag == "Strat" else "SsevCDWidthRand", f(float(ln.group(2)), 2), S7.name)
    for item in ln.group(3).split(","):
        lr, rk = item.split()
        put(f"SsevCD{tag}{FM_NAME[lr[:3]]}{'Bal' if lr.endswith('_bal') else ''}", f(float(rk), 2), S7.name)

# Study 7 design constants (prereg)
P7 = SIM / "prereg" / "STUDY7_PROTOCOL_MECHANISM.md"
p7 = read(P7)
put("SsevRandomReps", re.search(r"`random` \(\*\*(\d+) reps\*\*\)", p7).group(1), P7.name)
put("SsevSelReps", re.search(r"`lucos` \(\*\*(\d+) reps\*\* each\)", p7).group(1), P7.name)
put("SsevDoseReps", re.search(r"\*\*(\d+) reps\*\* per \(dataset", p7).group(1), P7.name)
put("SsevNContrasts", re.search(r"one family, Holm across all (\d+)", p7).group(1), P7.name)

# ---------------------------------------------------------------- Study 4 ---
S4A = REP / "STUDY4_STAGE_A.md"
s4 = read(S4A)
tab = section(s4, "### FM5: Δ vs realistic random (macro_f1)", r"\n### ")
for r in md_rows(tab)[1:]:
    if r[0] == "stratified":
        for j, w in enumerate(["One", "Three", "Five", "Ten"]):
            put(f"SfourStratGapM{w}", f(num(r[j + 1].split("[")[0]), sign=True), S4A.name)
abs_t = section(s4, "### Absolute means over datasets (FM5)", r"\n### ")
rows = md_rows(abs_t)
hdr = rows[0]
for r in rows[1:]:
    if r[0] == "random":
        put("SfourRandCovOne", pct(num(r[hdr.index("('coverage', 1)")])), S4A.name)
        put("SfourRandCovThree", pct(num(r[hdr.index("('coverage', 3)")])), S4A.name)
hr = section(s4, "### RQ-L4 headroom", r"\n### ")
for r in md_rows(hr)[1:]:
    if r[0] == "1":
        put("SfourHeadroomOne", f(num(r[4])), S4A.name)
    if r[0] == "10":
        put("SfourHeadroomTen", f(num(r[4])), S4A.name)

S4B = REP / "STUDY4_BASELINES.md"
s4b = read(S4B)
blk = section(s4b, "## Selector − random, macro_f1: mean over the four FMs", r"\n### ")
for r in md_rows(blk)[1:]:
    nm = r[0]
    if nm in ("lucos", "usl", "kmedoids_raw", "zcore", "kmedoids", "typiclust"):
        for j, w in enumerate(["One", "Three", "Five", "Ten"]):
            put(f"SfourSel{nm.replace('_', '').title()}M{w}", f(num(r[j + 1].split("(")[0]), sign=True), S4B.name)

# ------------------------------------------------------------- Theory -------
TC = REP / "THEORY_CHECKS.md"
tc = read(TC)
alld = section(tc, "### All datasets", r"\n### ")
rows = md_rows(alld)
hdr = rows[0]
for r in rows[1:]:
    w = {"1": "One", "3": "Three", "5": "Five", "10": "Ten"}[r[0]]
    for col, k in [("f1_gap", "FGap"), ("f1_miss_obs", "FCovObs"), ("f1_cov_pred", "FCovPred"),
                   ("f1_bal_obs", "FBalObs"), ("f1_bal_pred", "FBalPred"), ("f1_total_pred", "FTotPred"),
                   ("acc_gap", "AccGap"), ("acc_miss_obs", "AccCovObs"), ("acc_cov_pred", "AccCovPred")]:
        put(f"Th{k}M{w}", f(num(r[hdr.index(col)]), sign=True), TC.name)
byk = section(tc, "### By pool type", r"\n### ")
rows = md_rows(byk)
hdr = rows[0]
for r in rows[1:]:
    if r[0] == "exact" and r[1] == "1":
        put("ThBinFCovObs", f(num(r[hdr.index("f1_miss_obs")])), TC.name)
        put("ThBinFCovPred", f(num(r[hdr.index("f1_cov_pred")])), TC.name)
        put("ThBinAccCovObs", f(num(r[hdr.index("acc_miss_obs")]), 4), TC.name)
        put("ThBinAccCovPred", f(num(r[hdr.index("acc_cov_pred")]), 4), TC.name)
put("ThASevenMaxErr", "$" + re.search(r"max \|observed - formula\| = ([\d.e\-]+)", tc).group(1).rstrip(".").replace("e-", "\\times10^{-") + "}$",
    TC.name)
cor = section(tc, "### Correlation across datasets", r"\n### ")
for r in md_rows(cor)[1:]:
    w = {"1": "One", "3": "Three", "5": "Five", "10": "Ten"}[r[0]]
    put(f"ThRFCovM{w}", f(num(r[1]), 2), TC.name)
imb = section(tc, "### Imbalance within covered random draws", r"\nSame, FM1")
for r in md_rows(imb)[1:]:
    w = {"3": "Three", "5": "Five", "10": "Ten"}[r[0]]
    put(f"ThPrFDM{w}", f(num(r[3]), 2), TC.name)
    put(f"ThPrFDpiM{w}", f(num(r[4]), 2), TC.name)
    put(f"ThPrAccDM{w}", f(num(r[5]), 2), TC.name)
    put(f"ThPrAccDpiM{w}", f(num(r[6]), 2), TC.name)
l1 = section(tc, "### Lemma 1 check", r"\n### ")
for r in md_rows(l1)[1:]:
    w = {"3": "Three", "5": "Five", "10": "Ten"}[r[0]]
    put(f"ThLemInt{w}", f(num(r[1])), TC.name)
    put(f"ThLemIntLo{w}", f(num(r[2])), TC.name)
    put(f"ThLemIntHi{w}", f(num(r[3])), TC.name)
put("ThMCrossMedian", f(float(re.search(r"Median m_cross = ([\d.]+)", tc).group(1)), 1), TC.name)
put("ThKappa", f(float(re.search(r"kappa \(median LODO slope at 3C\) = (\d+\.\d+)", tc).group(1)), 2), TC.name)
b4 = section(tc, "### Prop. B4 endpoint conditions", r"\n### ")
r = md_rows(b4)[1]
put("ThBFourCellsPi", r[0], TC.name)
put("ThBFourViolPi", r[1], TC.name)
put("ThBFourCellsU", r[2], TC.name)
put("ThBFourViolU", r[3], TC.name)
put("ThBFourBetween", r[4], TC.name)
sg = re.search(r"BA ([\d.]+), accuracy ([\d.]+), macro-F1 ([\d.]+) \((\d+) cells", tc)
put("ThSignBA", f(float(sg.group(1)), 2), TC.name)
put("ThSignAcc", f(float(sg.group(2)), 2), TC.name)
put("ThSignF", f(float(sg.group(3)), 2), TC.name)
put("ThSignCells", sg.group(4), TC.name)
lp = section(tc, "### Learned plug-in", r"\n## ")
for r in md_rows(lp)[1:]:
    if r[0].startswith("C=2 pi=(.25,.75)") and r[1] == "10":
        put("ThPlugGapF", f(num(r[2])), TC.name)
        put("ThPlugGapFPc", f(num(r[5])), TC.name)
        put("ThPlugGapAcc", f(num(r[4])), TC.name)
        put("ThPlugGapAccPc", f(num(r[6])), TC.name)
lam = section(tc, "## 5. Label equivalence", r"\Z")
for r in md_rows(lam)[1:]:
    w = {"1": "One", "3": "Three", "5": "Five", "10": "Ten"}[r[0]]
    put(f"ThLamMed{w}", f(num(r[4]), 2), TC.name)
    put(f"ThLamCovMed{w}", f(num(r[5]), 2), TC.name)
cb = section(tc, "## 2. Coverage budgets", r"\nMonte Carlo")
for r in md_rows(cb)[1:]:
    if r[0] in ("DEV 1071", "DEV 694", "DEV 40496") and r[1] == "1":
        put(f"ThBudget{r[0].split()[1]}", r[2], TC.name)
mc = section(tc, "Monte Carlo P(full coverage) at the exact r = 1 budget", r"\n## ")
pm = [num(r[2]) for r in md_rows(mc)[1:]]
put("ThCovMCMin", f(min(pm)), TC.name)
put("ThCovMCMax", f(max(pm)), TC.name)

# ---------------------------------------------------------------- Study 9 ---
S9 = REP / "STUDY9_T1PRIME_OUTPUT.md"
s9 = read(S9)
S9R = REP / "STUDY9_T1PRIME_REPORT.md"
s9r = read(S9R)
put("SnineN", re.search(r"T1′, (\d+) datasets", s9r).group(1), S9R.name)
put("HeldOutN", re.search(r"lists (\d+) IDs", s9r).group(1), S9R.name)
put("SnineMaxOverlap", re.search(r"maximum over all .* is \*\*([\d.]+)\*\*", s9r).group(1), S9R.name)
put("SnineNExcludedScreens", re.search(r"(\d+) candidates were excluded before the draw", s9r).group(1), S9R.name)

b1 = section(s9, "## B1. H1-dist", r"\nPost-stratified")
F_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CELL_RE = re.compile(r"([\d.]+) \[([\d.]+), ([\d.]+)\]")
for r in md_rows(b1)[1:]:
    tagmap = {"EXP' all": "All", "eps = 0.005": "EpsLow", "eps = 0.02": "EpsHigh", "accuracy": "Acc",
              "C2": "CTwo", "C3-4": "CThreeFour", "C5-10": "CFiveTen", "fm2, stratified": "Strat",
              "fm1, random": "FMone", "fm5, random": "FMfive", "fm3, random": "FMthree",
              "fm2, kmedoids_nested": "KMn", "fm2, lucos_nested": "LUn", "balanced core": "Core",
              "numeric-only (2a + 4)": "Numeric"}
    t = tagmap.get(r[0])
    if not t:
        continue
    put(f"SnineHOne{t}N", r[1], S9.name)
    put(f"SnineHOne{t}Median", "$" + r[2].replace("inf", "\\infty") + "$", S9.name)
    put(f"SnineHOne{t}IQR", "$" + r[3].replace("–", "$--$").replace("inf", "\\infty") + "$", S9.name)
    for col, k in [(4, "Pfive"), (5, "Pten"), (6, "Ptwenty")]:
        m = CELL_RE.search(r[col])
        put(f"SnineHOne{t}{k}", f(float(m.group(1)), 2), S9.name)
        put(f"SnineHOne{t}{k}Lo", f(float(m.group(2)), 2), S9.name)
        put(f"SnineHOne{t}{k}Hi", f(float(m.group(3)), 2), S9.name)
# grid of F-hat for FM2 random (used in a table)
allrow = [r for r in md_rows(b1) if r[0] == "EXP' all"][0]
for g, v in zip(F_GRID, allrow[7].split()):
    put(f"SnineFhat{g}", v, S9.name)

aft = section(s9, "AFT (interval-censored log-normal), FM2, random:", r"\nH1a")
for r in md_rows(aft)[1:]:
    if r[0] == "log_C":
        put("SnineBetaC", f(num(r[1])), S9.name)
        put("SnineBetaCSE", f(num(r[2])), S9.name)
        put("SnineBetaCP", f(num(r[4]), 2), S9.name)
icc = re.search(r"H1a ICC: \{'n': (\d+), 'icc': ([\d.]+), 'ci_low': ([\d.]+), 'ci_high': ([\d.]+)", s9)
put("SnineICCN", icc.group(1), S9.name)
put("SnineICC", f(float(icc.group(2)), 2), S9.name)
put("SnineICCLo", f(float(icc.group(3)), 2), S9.name)
put("SnineICCHi", f(float(icc.group(4)), 2), S9.name)
icc4 = re.search(r"Four-FM ICC\(2,1\), full-grid datasets: \{n: (\d+), icc: ([\d.]+), ci_low: ([\d.]+), ci_high: ([\d.]+)", s9)
put("SnineICCFour", f(float(icc4.group(2)), 2), S9.name)
put("SnineICCFourLo", f(float(icc4.group(3)), 2), S9.name)
put("SnineICCFourHi", f(float(icc4.group(4)), 2), S9.name)

h2 = section(s9, "## H2: label equivalence (FM2)", r"\n## ")
for strat, t in [("kmedoids_nested", "KMn"), ("lucos_nested", "LUn"), ("stratified", "Strat")]:
    line = [ln for ln in h2.splitlines() if ln.startswith(f"- {strat}:")][0]
    share = float(re.search(r"'share_gt1': (?:np.float64\()?([\d.]+)", line).group(1))
    p = float(re.search(r"'p_sign': ([\d.e\-]+)", line).group(1))
    n = int(re.search(r"'n': (\d+)", line).group(1))
    put(f"SnineRho{t}Share", f(share, 2), S9.name)
    put(f"SnineRho{t}Count", round(share * n), S9.name)
    put(f"SnineRho{t}P", pfmt(p), S9.name)
    meds = re.search(r"'medians': \{([^}]*)\}", line).group(1)
    for kv in meds.split(","):
        mm, vv = kv.split(":")
        put(f"SnineRho{t}MedM{int(mm.strip())}", f(float(vv), 2), S9.name)
    fd = re.search(r"'fade_greater': (\d+), 'fade_decided': (\d+), 'p_fade': ([\d.e\-]+)", line)
    if fd:
        put(f"SnineFade{t}", f"{fd.group(1)}/{fd.group(2)}", S9.name)
        put(f"SnineFade{t}P", f(float(fd.group(3)), 2), S9.name)
b6 = section(s9, "## B6. H2a magnitude", r"\n## ")
for strat, t in [("kmedoids_nested", "KMn"), ("lucos_nested", "LUn"), ("stratified", "Strat")]:
    m = re.search(rf"- {strat}: median ([\d.]+) \[([\d.]+), ([\d.]+)\]", b6)
    put(f"SnineRhoOne{t}", f(float(m.group(1)), 2), S9.name)
    put(f"SnineRhoOne{t}Lo", f(float(m.group(2)), 2), S9.name)
    put(f"SnineRhoOne{t}Hi", f(float(m.group(3)), 2), S9.name)
    lc = re.search(rf"- {strat}: .*left-censored (\d+)", b6).group(1)
    put(f"SnineRhoOne{t}LeftCens", lc, S9.name)
put("SnineRhoOneStratApprox", f"{round(float(re.search(r'- stratified: median ([\d.]+)', b6).group(1)))}",
    S9.name + " (rounded point estimate)")
for strat, t in [("kmedoids_nested", "KMn"), ("lucos_nested", "LUn"), ("stratified", "Strat")]:
    line = [ln for ln in b6.splitlines() if ln.startswith(f"- {strat}: {{n")][0]
    put(f"SnineAccRho{t}Share", f(float(re.search(r"share_gt1: ([\d.]+)", line).group(1)), 2), S9.name)
    # half-up on the source string (1.025 -> 1.03), as in STUDY9_T1PRIME_REPORT.md
    from decimal import ROUND_HALF_UP as _RHU, Decimal as _Dec
    _mo = re.search(r"medians: \{1: ([\d.]+)", line).group(1)
    put(f"SnineAccRho{t}MedOne", str(_Dec(_mo).quantize(_Dec("0.01"), rounding=_RHU)), S9.name + " (half-up)")
    put(f"SnineAccRho{t}P", pfmt(float(re.search(r"p_sign: ([\d.e\-]+)", line).group(1))), S9.name)

h3 = re.search(r"\{'h3a': \{'n': \d+, 'share': ([\d.]+), 'ci_low': ([\d.]+), 'ci_high': ([\d.]+).*'h3b_tau': ([\-\d.]+), 'h3b_p': ([\d.e\-]+)", s9)
put("SnineHThreeShare", f(float(h3.group(1)), 2), S9.name)
put("SnineHThreeLo", f(float(h3.group(2)), 2), S9.name)
put("SnineHThreeHi", f(float(h3.group(3)), 2), S9.name)
put("SnineHThreeTau", f(float(h3.group(4)), 2), S9.name)
b8 = section(s9, "## B8. H3 descriptives", r"\n- p_cov")
for r in md_rows(b8)[1:]:
    t = {"IR<2": "Low", "IR 2-5": "Mid", "IR>5": "High"}[r[0]]
    put(f"SnineUnsafe{t}", f(num(r[2]), 2), S9.name)
    put(f"SnineUnsafe{t}N", r[1], S9.name)
    put(f"SnineUnsafe{t}Pcov", f(num(r[4]), 2), S9.name)
h4 = re.search(r"'agree': \{[^}]*'ratio': ([\d.]+), 'ci_low': ([\d.]+), 'ci_high': ([\d.]+), 'p': ([\d.]+)\}, "
               r"'fm_disagree': \{[^}]*'ratio': ([\d.]+), 'ci_low': ([\d.]+), 'ci_high': ([\d.]+), 'p': ([\d.]+)", s9)
for i, k in enumerate(["AgreeR", "AgreeLo", "AgreeHi", "AgreeP", "DisR", "DisLo", "DisHi", "DisP"]):
    put(f"SnineHFour{k}", f(float(h4.group(i + 1)), 3), S9.name)
holm = section(s9, "## Holm family", r"\n\| dataset")
for ln in holm.splitlines():
    mm = re.match(r"\| (\S+) \| ([\d.e\-]+) \| ([\d.e\-]+) \|", ln)
    if not mm:
        continue
    r = [mm.group(1), mm.group(2), mm.group(3)]
    t = {"H1b": "HOneb", "H2a|stratified": "HTwoaStrat", "H3b": "HThreeb", "H2a|kmedoids_nested": "HTwoaKMn",
         "H2a|lucos_nested": "HTwoaLUn", "H4|agree": "HFourA", "H4|fm_disagree": "HFourD",
         "H2b|kmedoids_nested": "HTwobKMn", "H2b|lucos_nested": "HTwobLUn"}.get(r[0])
    if t:
        put(f"SnineHolm{t}", pfmt(float(r[2])), S9.name)
        put(f"SnineRaw{t}", pfmt(float(r[1])), S9.name)  # unadjusted p (Table hyp)
b7 = re.search(r"H2 fm1 stratified: \{n: \d+, share_gt1: ([\d.]+), p_sign: ([\d.e\-]+)", s9)
put("SnineFMoneStratShare", f(float(b7.group(1)), 2), S9.name)
put("SnineFMoneStratP", pfmt(float(b7.group(2))), S9.name)
b7d = re.search(r"H4 fm1 primary, fm_disagree: \{n_datasets: \d+, ratio: ([\d.]+), ci_low: ([\d.]+), ci_high: ([\d.]+)", s9)
put("SnineFMoneDisR", f(float(b7d.group(1)), 3), S9.name)
b5 = re.search(r"Floor: m\* >= m95 decided yes for (\d+), decided no for (\d+), undecided (\d+)", s9)
put("SnineFloorYes", b5.group(1), S9.name)
put("SnineFloorNo", b5.group(2), S9.name)
put("SnineFloorUnd", b5.group(3), S9.name)
b2 = section(s9, "## B2. Censoring", r"\n## ")
for r in md_rows(b2)[1:]:
    t = {"C2": "CTwo", "C3-4": "CThreeFour", "C5-10": "CFiveTen"}[r[0]]
    put(f"SnineBin{t}N", r[1], S9.name)
put("SnineMultiN", sum(int(r[1]) for r in md_rows(b2)[1:] if r[0] != "C2"), S9.name)
# Turnbull CI resample counts: frozen h1() default vs prereg text
P9 = SIM / "prereg" / "STUDY9_HOW_MANY_LABELS.md"
p9 = read(P9)
put("SnineBootPrereg", re.search(r"dataset-bootstrap 95% CI \((\d+) resamples\)", p9).group(1), P9.name)
put("SnineBootFrozen", re.search(r"default of (\d+) resamples", s9r).group(1), S9R.name)
put("SnineEps", re.search(r"ε = (0\.\d+)\.", p9).group(1), P9.name)

# --------------------------------------------------------------- Study 12 ---
# Source: reports/STUDY12_REPORT.md, generated by reports/study12_report.py from results/study12/
# (the driver writes no other machine-readable file). Only the generated tables are parsed.
S12 = REP / "STUDY12_REPORT.md"
s12 = read(S12)
_s12 = S12.name
_hdr = re.search(r"T1′ = (\d+) tables \((\d+) binary, (\d+) multiclass", s12)
put("StwelveN", _hdr.group(1), _s12 + " header (Erratum 1)")
put("StwelveBinN", _hdr.group(2), _s12 + " header (Erratum 1)")
put("StwelveMultiN", _hdr.group(3), _s12 + " header (Erratum 1)")
_fam = section(s12, "\n## Confirmatory family")
_key12 = {"(i)-fm1": "IFMone", "(i)-fm2": "IFMtwo", "(i)-fm3": "IFMthree", "(i)-fm5": "IFMfive",
          "(ii-a)": "IIa", "(ii-c)": "IIc"}
_i_means, _i_holm, _i_win, _i_ratio = [], [], [], []
for r in md_rows(_fam):
    k = _key12.get(r[0])
    if not k:
        continue
    mean, lo, hi = (num(x) for x in CI_RE.search(r[4]).groups())
    put(f"Stwelve{k}Mean", f(mean, sign=True), _s12 + " (confirmatory table)")
    put(f"Stwelve{k}Lo", f(lo, sign=True), _s12 + " (confirmatory table)")
    put(f"Stwelve{k}Hi", f(hi, sign=True), _s12 + " (confirmatory table)")
    put(f"Stwelve{k}Win", f(num(r[6]), 2), _s12 + " (confirmatory table)")
    put(f"Stwelve{k}Holm", pfmt(num(r[8])), _s12 + " (confirmatory table)")
    put(f"Stwelve{k}Raw", pfmt(num(r[7])), _s12 + " (confirmatory table, unadjusted p)")
    put(f"Stwelve{k}Tost", f(num(r[9]), 2), _s12 + " (confirmatory table)")
    put(f"Stwelve{k}Rope", f(num(r[10].split("/")[1]), 2), _s12 + " (confirmatory table, Bayes ROPE)")
    put(f"Stwelve{k}Verdict", r[13].replace("*", ""), _s12 + " (confirmatory table)")
    put(f"Stwelve{k}Dev", f(num(re.match(r"([+\-−]?[\d.]+)", r[12]).group(1)), sign=True), _s12 + " (DEV mean column)")
    if r[0].startswith("(i)"):
        _i_means.append(mean); _i_holm.append(num(r[8])); _i_win.append(num(r[6]))
put("StwelveIMinMean", f(min(_i_means), sign=True), _s12)
put("StwelveIMaxMean", f(max(_i_means), sign=True), _s12)
put("StwelveIMaxHolm", pfmt(max(_i_holm)), _s12)
put("StwelveIMinWin", f(min(_i_win), 2), _s12)
put("StwelveIMaxWin", f(max(_i_win), 2), _s12)
_ov = re.search(r"\*\*Overall \(Sec\. 3\.1 rules\):\*\* C1 replicates \((\d)/(\d) FMs\); ranking claim \*\*([a-z ]+)\*\*", s12)
put("StwelveCOneK", _ov.group(1), _s12 + " (Overall line)")
put("StwelveRankingVerdict", _ov.group(3), _s12 + " (Overall line)")
_ratios = [float(x) for x in re.findall(r"(?:\(i\)-fm\d|\(ii-a\)) ([\d.]+)", section(s12, "Magnitude: T1′ mean / DEV mean", r"\n"))]
put("StwelveRatioMin", f(min(_ratios), 2), _s12 + " (Magnitude line)")
put("StwelveRatioMax", f(max(_ratios), 2), _s12 + " (Magnitude line)")
# per budget, macro-F1
_pb = section(s12, "## Robustness: per budget", r"\n### ")
_pbv = {}
for r in md_rows(_pb):
    if r[0] in _key12:
        _pbv[r[0]] = [float(re.match(r"([+\-−]?[\d.]+)", c).group(1).replace("−", "-")) for c in r[1:5]]
_ifm = [k for k in _pbv if k.startswith("(i)")]
for j, mm in [(0, "One"), (3, "Ten")]:
    put(f"StwelveGap{mm}Min", f(min(_pbv[k][j] for k in _ifm), sign=True), _s12 + " (per-budget macro-F1 table)")
    put(f"StwelveGap{mm}Max", f(max(_pbv[k][j] for k in _ifm), sign=True), _s12 + " (per-budget macro-F1 table)")
# per FM at ten labels per class (T1'), with the unadjusted per-budget p
_pbp = {}
for r in md_rows(_pb):
    if r[0] in _ifm and r[0] not in _pbp:
        _pbp[r[0]] = float(re.search(r"p=([\d.]+)", r[4]).group(1))
_FMW12 = {"(i)-fm1": "FMone", "(i)-fm2": "FMtwo", "(i)-fm3": "FMthree", "(i)-fm5": "FMfive"}
for k in _ifm:
    put(f"StwelveGapTen{_FMW12[k]}", f(_pbv[k][3], sign=True), _s12 + f" (per-budget macro-F1 table, {k}, 10C)")
    if _pbp[k] >= 0.01:  # the report rounds smaller p to 0.000
        put(f"StwelveGapTen{_FMW12[k]}P", f"{_pbp[k]:.2f}", _s12 + f" (per-budget macro-F1 table, {k}, 10C, unadjusted p)")
_sig = [k for k in _ifm if _pbp[k] < 0.05]
put("StwelveGapTenSigMin", f(min(_pbv[k][3] for k in _sig), sign=True), _s12 + " (10C, FMs with p<0.05)")
put("StwelveGapTenSigMax", f(max(_pbv[k][3] for k in _sig), sign=True), _s12 + " (10C, FMs with p<0.05)")
put("StwelveGapTenSigN", str(len(_sig)), _s12 + " (10C, FMs with p<0.05)")
# secondary metrics of the (i) contrasts
_sec = section(s12, "### The six contrasts in secondary metrics", r"\n## ")
for r in md_rows(_sec):
    if r[0] in ("accuracy", "balanced_accuracy"):
        vals = [num(CI_RE.search(c).group(1)) for c in r[1:5]]
        t = {"accuracy": "Acc", "balanced_accuracy": "BA"}[r[0]]
        put(f"StwelveI{t}Min", f(min(vals), sign=True), _s12 + " (secondary metrics table)")
        put(f"StwelveI{t}Max", f(max(vals), sign=True), _s12 + " (secondary metrics table)")
        ps = [float(re.search(r"p=([\d.e\-]+)", c).group(1)) for c in r[1:5]]
        put(f"StwelveI{t}PMin", pfmt(min(ps)), _s12 + " (secondary metrics table, unadjusted)")
# S-1 redraw
_s1 = section(s12, "## S-1: redraw protocol")
_cov = re.search(r"1C ([\d.]+), 3C ([\d.]+), 5C ([\d.]+), 10C ([\d.]+)", _s1)
put("StwelveCovShareOne", f(float(_cov.group(1)), 2), _s12 + " (S-1)")
_rd = {}
for r in md_rows(_s1):
    m = re.match(r"strat−redraw (fm\d)", r[0])
    if m:
        _rd[m.group(1)] = (num(CI_RE.search(r[2]).group(1)), num(r[5]))
        put(f"StwelveRedraw{FM_NAME[m.group(1)]}", f(_rd[m.group(1)][0], sign=True), _s12 + " (S-1)")
        put(f"StwelveRedraw{FM_NAME[m.group(1)]}Holm", pfmt(_rd[m.group(1)][1]), _s12 + " (S-1)")
    if r[0].startswith("fm2 − fm5 under redraw"):
        mean, lo, hi = (num(x) for x in CI_RE.search(r[2]).groups())
        put("StwelveRedrawRank", f(mean, sign=True), _s12 + " (S-1)")
        put("StwelveRedrawRankLo", f(lo, sign=True), _s12 + " (S-1)")
        put("StwelveRedrawRankHi", f(hi, sign=True), _s12 + " (S-1)")
_chk = [float(pp) for _, pp in re.findall(r"fm\d ([+\-−]?[\d.]+) \(p=([\d.]+)\)", re.search(r"Check at m = 1.*", _s1).group(0))]
put("StwelveRedrawCheckPMin", f(min(_chk), 2), _s12 + " (S-1 check at m = 1)")
# S-2 AUC
_s2 = section(s12, "## S-2: AUC")
_all, _seen = _s2.split("**auc (Study 7 convention")
def _auc_m1(block):
    out = []
    for r in md_rows(block):
        if re.match(r"fm\d$", r[0]):
            out.append((num(CI_RE.search(r[1]).group(1)), num(r[3])))
    return out
_a = _auc_m1(_all); _sn = _auc_m1(_seen.split("Share of fits")[0])
put("StwelveAucAllOneMin", f(min(x[0] for x in _a), sign=True), _s12 + " (S-2, auc_all, m=1)")
put("StwelveAucAllOneMax", f(max(x[0] for x in _a), sign=True), _s12 + " (S-2, auc_all, m=1)")
put("StwelveAucAllOneHolm", pfmt(max(x[1] for x in _a)), _s12 + " (S-2, auc_all, m=1, max Holm)")
put("StwelveAucSeenOneMin", f(min(x[0] for x in _sn), sign=True), _s12 + " (S-2, seen-class auc, m=1)")
put("StwelveAucSeenOneMax", f(max(x[0] for x in _sn), sign=True), _s12 + " (S-2, seen-class auc, m=1)")
put("StwelveAucSeenOneHolmMin", pfmt(min(x[1] for x in _sn)), _s12 + " (S-2, seen-class auc, m=1, min Holm)")
_und = [r for r in md_rows(_seen) if r[0] == "random"][0]
put("StwelveOneSeenShare", pct(float(_und[1])), _s12 + " (S-2, share of random 1C fits with one seen class)")
# S-3 flips
_s3 = section(s12, "## S-3: ranking flip rate")
for metric, nlearn, t in [("macro_f1", 6, "F"), ("accuracy", 6, "Acc"), ("macro_f1", 4, "FFM")]:
    ln = re.search(rf"\*\*{metric}, {nlearn} (?:learners|FMs)\.\*\* Protocol flip rate ([\d.]+) \[([\d.]+), ([\d.]+)\] "
                   rf"\((\d+)/(\d+) decisive pairs\); noise baseline [^)]*\) ([\d.]+) \[([\d.]+), ([\d.]+)\].*?difference "
                   rf"([+\-][\d.]+) \[([+\-][\d.]+), ([+\-][\d.]+)\]", _s3)
    g = ln.groups()
    put(f"StwelveFlip{t}", pct(float(g[0]), 1), _s12 + " (S-3)")
    put(f"StwelveFlip{t}Num", g[3], _s12 + " (S-3)")
    put(f"StwelveFlip{t}Den", g[4], _s12 + " (S-3)")
    put(f"StwelveFlip{t}Noise", pct(float(g[5]), 1), _s12 + " (S-3, random-vs-random baseline)")
    put(f"StwelveFlip{t}Diff", f(float(g[8]), 2, sign=True), _s12 + " (S-3)")
    put(f"StwelveFlip{t}DiffLo", f(float(g[9]), 2, sign=True), _s12 + " (S-3)")
    put(f"StwelveFlip{t}DiffHi", f(float(g[10]), 2, sign=True), _s12 + " (S-3)")
# S-4 binary vs multiclass
_s4 = section(s12, "## S-4: binary vs multiclass")
for r in md_rows(_s4):
    k = {"(ii-a)": "IIa", "(ii-c)": "IIc", "(i)-fm1": "IFMone", "(i)-fm5 @1C": "IFMfiveMOne"}.get(r[0])
    if not k:
        continue
    put(f"StwelveSplit{k}Bin", f(num(CI_RE.search(r[1]).group(1)), sign=True), _s12 + " (S-4)")
    put(f"StwelveSplit{k}Multi", f(num(CI_RE.search(r[2]).group(1)), sign=True), _s12 + " (S-4)")
    put(f"StwelveSplit{k}MW", r[3].strip(), _s12 + " (S-4, Mann-Whitney; as printed)")
_s4i = [r for r in md_rows(_s4) if re.match(r"\(i\)-fm\d$", r[0])]
put("StwelveSplitIMultiLarger", str(sum(num(CI_RE.search(r[2]).group(1)) > num(CI_RE.search(r[1]).group(1)) for r in _s4i)),
    _s12 + " (S-4, count of FMs with a larger multiclass gap)")
# sensitivities
for lab, t in [("SENS-INT", "Int"), ("SENS-ID", "Id")]:
    blk = section(s12, f"#### {lab}", r"\n#### |\Z")
    put(f"StwelveSens{t}N", re.search(r"\(n = (\d+)\)", blk).group(1), _s12 + f" ({lab})")
    ov = re.search(r"Overall: C1 replicates \((\d)/4 FMs\); ranking claim ([a-z ]+)\.", blk)
    assert ov.group(1) == _ov.group(1) and ov.group(2) == _ov.group(3), lab
_cmp = re.search(r"- ([\d,]+) fits \(incl\. constant one-class contexts\), with ([\d.]+) worker-hours", s12)
put("StwelveFits", _cmp.group(1).replace(",", "{,}"), _s12 + " (Compute)")
put("StwelveWorkerHours", _cmp.group(2).split(".")[0], _s12 + " (Compute, truncated)")
# exploratory UCI 257 sensitivity of Study 12 (drop / merge): verdicts and the (i) range
U12 = REP / "STUDY12_UCI257_EXPLORATORY.md"
u12 = read(U12)
for lab, t in [("DROP", "Drop"), ("MERGE", "Merge")]:
    blk = section(u12, f"## {lab}")
    rr = re.search(r"Reading rules: C1 replicates \((\d)/4 FMs\); ranking claim ([a-z ]+)\.", blk)
    assert rr.group(1) == _ov.group(1) and rr.group(2) == _ov.group(3), lab
    ms = [num(CI_RE.search(r[1]).group(1)) for r in md_rows(blk.split("Binary / multiclass split")[0])
          if re.match(r"\(i\)-fm\d$", r[0])]
    put(f"StwelveUci{t}IMin", f(min(ms), sign=True), U12.name + f" ({lab}, (i) means)")
    put(f"StwelveUci{t}IMax", f(max(ms), sign=True), U12.name + f" ({lab}, (i) means)")

# ------------------------------------------------ Systematic audit (SR) -----
# Source: paper/p1/supplementary/audit/ (self-contained, shipped with the paper). Counts and
# Wilson intervals come from analysis_output.json (written by scripts/analyze.py); the few
# values that exist only in the internal review report, the protocol or the human-check sample
# come from audit_numbers.json (written once by supplementary/audit/extract_numbers.py).
import json as _json  # noqa: E402
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402

SR = HERE / "supplementary" / "audit"
srj = _json.loads(read(SR / "analysis_output.json"), parse_float=Decimal)
srn = _json.loads(read(SR / "audit_numbers.json"), parse_float=Decimal)["values"]
_srj = "supplementary/audit/analysis_output.json"
_srn = "supplementary/audit/audit_numbers.json"


def _sv(key: str):
    """Value of audit_numbers.json and its source comment."""
    return srn[key]["value"], f"{_srn} {key} ({srn[key]['source']})"


def _half_up(x, nd: int = 0) -> str:
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def _int(x) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def _share(stem: str, d: dict, key: str, nd: int = 0) -> None:
    """k, n, percentage and Wilson 95% bounds of one proportion of analysis_output.json."""
    src = f"{_srj} {key}"
    put(f"{stem}K", _int(d["k"]), src)
    put(f"{stem}N", _int(d["n"]), src)
    put(f"{stem}Pct", _half_up(d["pct"], nd) + "\\%", src)
    put(f"{stem}Lo", _half_up(d["wilson_lo"], nd) + "\\%", src + " (Wilson 95% lower)")
    put(f"{stem}Hi", _half_up(d["wilson_hi"], nd) + "\\%", src + " (Wilson 95% upper)")


# PRISMA flow
_pr = srj["prisma"]
_id = _pr["identification"]
for nm, key in [("ArxivHits", "arxiv_raw_hits"), ("ArxivUnique", "arxiv_unique"),
                ("SsHits", "s2_raw_hits"), ("SsScope", "s2_in_scope"),
                ("OaHits", "openalex_raw_hits"), ("OaScope", "openalex_in_scope"),
                ("SnowCites", "snowball_raw_citations"), ("SnowTerm", "snowball_with_L_term_in_window"),
                ("SnowScope", "snowball_in_scope"), ("PriorRecords", "prior_audit_records"),
                ("BeforeDedup", "records_before_dedup"), ("Duplicates", "duplicates_removed"),
                ("Unique", "records_after_dedup"), ("OnlyPrior", "only_prior_audit")]:
    put(f"SysAudit{nm}", _int(_id[key]), f"{_srj} prisma.identification.{key}")
for nm, key in [("Screened", "stage1_screened"), ("ScreenExcl", "stage1_excluded"),
                ("ScreenIncl", "stage1_included"), ("Sought", "fulltext_sought"),
                ("NotAccessible", "fulltext_not_accessible"), ("Assessed", "fulltext_assessed"),
                ("FullExcl", "fulltext_excluded"), ("Included", "included"),
                ("PriorOnlyIncluded", "included_from_prior_audit_only")]:
    put(f"SysAudit{nm}", _int(_pr[key]), f"{_srj} prisma.{key}")
put("SysAuditSearchIncluded", _int(_pr["included"] - _pr["included_from_prior_audit_only"]),
    f"{_srj} prisma.included - included_from_prior_audit_only")

# Included papers whose only version is a preprint (venue_type == "preprint").
import csv as _csv_s2  # noqa: E402

_s2_path = SR / "screening" / "stage2_decisions.csv"
with _s2_path.open(newline="", encoding="utf-8") as _fh:
    _s2_inc = [r for r in _csv_s2.DictReader(_fh) if r["decision"] == "include"]
assert len(_s2_inc) == _pr["included"], (len(_s2_inc), _pr["included"])
put("SysAuditPreprintK", _int(sum(r["venue_type"] == "preprint" for r in _s2_inc)),
    f"{_s2_path.relative_to(HERE)} decision == include and venue_type == preprint")
_reason = {"not_classification": "NotCls", "not_tabular": "NotTab", "table_QA_or_text2sql": "TableQA",
           "no_reduced_label_experiment": "NoReduced", "survey_or_position": "Survey",
           "recsys_cold_start": "Recsys", "out_of_window": "Window", "duplicate": "Dup"}
for key, v in _pr["stage1_excluded_by_reason"].items():
    put(f"SysAuditScreenExcl{_reason[key]}", _int(v), f"{_srj} prisma.stage1_excluded_by_reason.{key}")
for key, v in _pr["fulltext_excluded_by_reason"].items():
    put(f"SysAuditFullExcl{_reason[key]}", _int(v), f"{_srj} prisma.fulltext_excluded_by_reason.{key}")
if sum(_pr["stage1_excluded_by_reason"].values()) != _pr["stage1_excluded"]:
    raise ValueError("SR: stage-1 exclusion reasons do not add up")
if sum(_pr["fulltext_excluded_by_reason"].values()) != _pr["fulltext_excluded"]:
    raise ValueError("SR: full-text exclusion reasons do not add up")

# Agreement between the two coders (Cohen's kappa, raw agreement)
_ag = srj["agreement"]
_agname = {"category": "Cat", "coverage_forced": "Cov", "few_shot": "Fs", "missed_class_reported": "Missed",
           "budget_unit": "Budget", "main_metric": "Metric"}
for key, nm in _agname.items():
    put(f"SysAuditRaw{nm}", _half_up(_ag[key]["raw_agreement"], 3), f"{_srj} agreement.{key}.raw_agreement")
    if _ag[key]["kappa"] is not None:
        put(f"SysAuditKappa{nm}", _half_up(_ag[key]["kappa"], 3), f"{_srj} agreement.{key}.kappa")
_kap = [v["kappa"] for v in _ag.values() if v["kappa"] is not None]
put("SysAuditKappaMin", _half_up(min(_kap), 2), f"{_srj} agreement (min kappa over defined variables, 2 d.p.)")
put("SysAuditKappaMax", _half_up(max(_kap), 2), f"{_srj} agreement (max kappa over defined variables, 2 d.p.)")

# Main shares: few-shot (Fs) and all reduced-label papers (All)
_mn = srj["main"]
put("SysAuditFsN", _int(_mn["n_few_shot"]), f"{_srj} main.n_few_shot")
if _mn["n_reduced_label"] != _pr["included"]:
    raise ValueError("SR: included count disagrees with main.n_reduced_label")
_catnm = {"A": "A", "A'": "Aprime", "B": "B", "C": "C", "D": "D", "E": "E"}
for key, nm in _catnm.items():
    put(f"SysAuditCat{nm}", _int(_mn["category_counts_all"][key]), f"{_srj} main.category_counts_all")
    put(f"SysAuditFsCat{nm}", _int(_mn["category_counts_few_shot"][key]), f"{_srj} main.category_counts_few_shot")
_share("SysAuditFsLab", _mn["AorAprime_few_shot"], "main.AorAprime_few_shot")
_share("SysAuditAllLab", _mn["AorAprime_all"], "main.AorAprime_all")
_share("SysAuditFsStated", _mn["AorAprime_among_stated_draw_few_shot"], "main.AorAprime_among_stated_draw_few_shot")
_share("SysAuditAllStated", _mn["AorAprime_among_stated_draw_all"], "main.AorAprime_among_stated_draw_all")
_share("SysAuditFsForced", _mn["coverage_forced_yes_few_shot"], "main.coverage_forced_yes_few_shot")
_share("SysAuditAllForced", _mn["coverage_forced_yes_all"], "main.coverage_forced_yes_all")
_share("SysAuditMissed", _mn["missed_class_reported_all"], "main.missed_class_reported_all", nd=1)
_share("SysAuditFsMissed", _mn["missed_class_reported_few_shot"], "main.missed_class_reported_few_shot", nd=1)
# consistency: stated-draw denominator = few-shot minus D and E
if (_mn["AorAprime_among_stated_draw_few_shot"]["n"]
        != _mn["n_few_shot"] - _mn["category_counts_few_shot"]["D"] - _mn["category_counts_few_shot"]["E"]):
    raise ValueError("SR: stated-draw denominator is not few-shot minus D and E")
put("SysAuditUnseenStated", len(srj["unseen_class_scoring_stated"]), f"{_srj} unseen_class_scoring_stated (list length)")

# Sensitivity analyses
for stem, key in [("SysAuditSearch", "sensitivity_search_only"), ("SysAuditNoAnom", "posthoc_excluding_anomaly_PU"),
                  ("SysAuditInh", "posthoc_inherited_recoded_A")]:
    d = srj[key]
    _share(f"{stem}Fs", d["AorAprime_few_shot"], f"{key}.AorAprime_few_shot")
    _share(f"{stem}FsStated", d["AorAprime_among_stated_draw_few_shot"], f"{key}.AorAprime_among_stated_draw_few_shot")
put("SysAuditNoAnomExcl", _int(_mn["n_reduced_label"] - srj["posthoc_excluding_anomaly_PU"]["n_reduced_label"]),
    f"{_srj} main.n_reduced_label - posthoc_excluding_anomaly_PU.n_reduced_label")
put("SysAuditInhRecoded", _int(srj["posthoc_inherited_recoded_A"]["category_counts_all"]["A"] - _mn["category_counts_all"]["A"]),
    f"{_srj} posthoc_inherited_recoded_A.category_counts_all.A - main.category_counts_all.A")
_v, _s = _sv("inaccessible_worst_case_percent")
put("SysAuditInaccWorst", _half_up(_v, 0) + "\\%", _s)
_v, _s = _sv("inaccessible_best_case_percent")
put("SysAuditInaccBest", _half_up(_v, 0) + "\\%", _s)

# Main metric and any use of a metric
_metnm = {"auc": "AUC", "accuracy": "Acc", "macro-f1": "MF", "balanced-accuracy": "BA",
          "f1-other": "FOther", "log-loss": "LogLoss", "other": "Other"}
for key, nm in _metnm.items():
    _share(f"SysAuditFsMain{nm}", srj["main_metric_few_shot"][key], f"main_metric_few_shot.{key}")
    _share(f"SysAuditMain{nm}", srj["main_metric"][key], f"main_metric.{key}")
_share("SysAuditFsUseMF", srj["metrics_any_use_few_shot"]["macro-f1"], "metrics_any_use_few_shot.macro-f1")
_share("SysAuditUseMF", srj["metrics_any_use"]["macro-f1"], "metrics_any_use.macro-f1")

# Values found only in the report, the protocol or the coding files (audit_numbers.json)
for nm, key in [("DisPairs", "disagreement_pairs"), ("DisPapers", "disagreement_papers"),
                ("Postcheck", "postcheck_corrections"), ("CodeRead", "code_read_papers"),
                ("HumanN", "human_check_sample_size"), ("OldTotal", "old_audit_total_papers"),
                ("OldReduced", "old_audit_reduced_label_papers"), ("OldFsK", "old_audit_few_shot_k"),
                ("OldFsN", "old_audit_few_shot_n"), ("NTerms", "n_label_scarcity_terms")]:
    _v, _s = _sv(key)
    put(f"SysAudit{nm}", _v, _s)
for nm, key in [("HumanPct", "human_check_percent"), ("OldFsPct", "old_audit_few_shot_percent")]:
    _v, _s = _sv(key)
    put(f"SysAudit{nm}", f"{_v}\\%", _s)
# Blind third coder on the random 32-paper sample (supplementary/audit/coding/coder3_agreement.json)
_c3p = "supplementary/audit/coding/coder3_agreement.json"
_c3 = _json.loads(read(SR / "coding" / "coder3_agreement.json"), parse_float=Decimal)
if _c3["n"] != int(srn["human_check_sample_size"]["value"]):
    raise ValueError("SR: third-coder sample size differs from the human-check sample size")
put("SysAuditCoderThreeN", _c3["n"], _c3p + " n")
_c3vars = {"category": "Cat", "coverage_forced": "Cov", "few_shot": "Fs", "main_metric": "Met"}
for key, nm in _c3vars.items():
    put(f"SysAuditCoderThreeKappa{nm}", _half_up(_c3[key]["cohen_kappa"], 2), f"{_c3p} {key}.cohen_kappa")
    put(f"SysAuditCoderThreeAgree{nm}", _half_up(_c3[key]["raw_agreement"], 2), f"{_c3p} {key}.raw_agreement")
import csv as _csv
_c3d = list(_csv.DictReader(read(SR / "coding" / "coder3_disagreements.csv").splitlines()))
put("SysAuditCoderThreeDisPairs", len(_c3d), "supplementary/audit/coding/coder3_disagreements.csv rows")
put("SysAuditCoderThreeDisPapers", len({r["paper_id"] for r in _c3d}),
    "supplementary/audit/coding/coder3_disagreements.csv distinct paper_id")
_c3m = list(_csv.DictReader(read(SR / "coding" / "coder3_check.csv").splitlines()))
put("SysAuditCoderThreeMissedYes", sum(r["missed_class_reported"] == "yes" for r in _c3m),
    "supplementary/audit/coding/coder3_check.csv missed_class_reported == yes")
_c3k = [_c3[k]["cohen_kappa"] for k in _c3vars]
put("SysAuditCoderThreeKappaMin", _half_up(min(_c3k), 2), _c3p + " min cohen_kappa")
put("SysAuditCoderThreeKappaMax", _half_up(max(_c3k), 2), _c3p + " max cohen_kappa")
# Supplementary full-text search (supplementary/audit/fulltext/analysis_supp.json)
_ftp = "supplementary/audit/fulltext/analysis_supp.json"
_ft = _json.loads(read(SR / "fulltext" / "analysis_supp.json"))
_pr = _ft["prisma"]
for nm, val, key in [("Census", _pr["stage1_included_C"], "prisma.stage1_included_C"),
                     ("Sampled", _pr["R_sampled"], "prisma.R_sampled"),
                     ("EligibleR", _pr["stage1_included_R"], "prisma.stage1_included_R"),
                     ("Records", _pr["identification"]["records_after_dedup"], "prisma.identification.records_after_dedup"),
                     ("Assessed", _pr["fulltext_assessed"], "prisma.fulltext_assessed"),
                     ("Included", _pr["included"], "prisma.included"),
                     ("NoReduced", _pr["stage2_excluded_by_reason"]["no_reduced_label_experiment"],
                      "prisma.stage2_excluded_by_reason.no_reduced_label_experiment"),
                     ("Coverage", _ft["all"]["coverage_forced"].get("yes", 0), "all.coverage_forced.yes"),
                     ("CatE", _ft["all"]["category"].get("E", 0), "all.category.E"),
                     ("AorAp", _ft["all"]["category"].get("A", 0) + _ft["all"]["category"].get("A'", 0),
                      "all.category.A + A'"),
                     ("Missed", _ft["all"]["missed_class_reported"].get("yes", 0), "all.missed_class_reported.yes")]:
    put(f"SuppFt{nm}", val, f"{_ftp} {key}")
if _pr["coded"] != _ft["all"]["n"]:
    raise ValueError("SuppFt: coded count differs from n")
_d0, _s = _sv("date_window_start")
_d1, _s1 = _sv("date_window_end")
_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]
put("SysAuditYearFrom", _d0[:4], _s)
put("SysAuditYearTo", _d1[:4], _s1)
put("SysAuditDateTo", f"{int(_d1[8:10])} {_months[int(_d1[5:7]) - 1]} {_d1[:4]}", _s1)
if srn["snowball_seed_papers"]["value"] != ["VIME (NeurIPS 2020)", "SCARF (ICLR 2022)", "STUNT (ICLR 2023)",
                                            "TabLLM (AISTATS 2023)"]:
    raise ValueError("SR: snowball seed list changed in the protocol")
if srn["report_n_few_shot"]["value"] != _mn["n_few_shot"]:
    raise ValueError("SR: REPORT and analysis_output.json disagree on the few-shot count")

# ------------------------------------------------------------- DEV ----------
PC = REP / "P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md"
m = re.search(r"DEV has (\d+) binary and (\d+) multiclass", read(PC))
put("DevBinary", m.group(1), PC.name)
put("DevMulti", m.group(2), PC.name)
put("DevN", int(m.group(1)) + int(m.group(2)), PC.name)
P7A = read(P7)
m = re.search(r"\*\*Sorted by class \((\d+)\):\*\*", P7A)
put("DevSorted", m.group(1), P7.name)
put("DevDupDatasets", 6, P7.name + " §6.1, 'S2 excludes the six datasets with conflicting duplicates'")
put("DevIDOneNN", "0.89", P7.name + " §6.1, '5-fold 1-NN accuracy 0.89 on that column alone'")
put("DevSpentEvals", 120, "internal discovery synthesis note §2.4 item 6 (about 120 evaluations)")
put("DevSpentDatasets", 10, "internal discovery synthesis note §2.4 item 6 (10 DEV TUNE datasets)")
put("GrinsztajnSorted", "22/22", "internal discovery synthesis note §2.6 (file 25 §2)")
put("TabRepoSorted", "164/164", "internal discovery synthesis note §2.6 (file 25 §3)")


# ------------------------------------------------- Protocol corrections ----
# reports/P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md, section "Results (2026-10-27, exploratory, DEV only)".
PCR = section(read(PC), "## Results", r"\n### Prior art")
_N = r"[+−-]?\d+\.\d+"


def _sg(x: str) -> float:
    return float(x.replace("−", "-").replace("+", ""))


def _pair(pat: str) -> tuple[float, float]:
    m = re.search(pat, PCR)
    if m is None:
        raise ValueError(f"corrections pattern not found: {pat}")
    return _sg(m.group(1)), _sg(m.group(2))


def _put_range(stem: str, pat: str, nd: int = 2, sign: bool = False,
               names: tuple[str, str] = ("Lo", "Hi")) -> None:
    lo, hi = _pair(pat)
    put(stem + names[0], f(lo, nd, sign=sign), PC.name + " Results")
    put(stem + names[1], f(hi, nd, sign=sign), PC.name + " Results")


m = re.search(r"k = 5: (\d\.\d+) / (\d\.\d+) / (\d\.\d+) / (\d\.\d+)", PCR)
k5 = [float(v) for v in m.groups()]
put("CfourGregKFiveLo", f(min(k5), 2), PC.name + " Results (GREG, k = 5)")
put("CfourGregKFiveHi", f(max(k5), 2), PC.name + " Results (GREG, k = 5)")
_put_range("CfourGregOneC", rf"from 1C \(ratio ({_N})–({_N})\)")
_put_range("CfourGregTenC", rf"At 10C the ratio is ({_N})–({_N})")
_put_range("CfourDesOneC", rf"\| 1C \| ({_N})–({_N})")
_put_range("CfourDesThreeC", rf"\| 3C \(r = 2\) \| ({_N})–({_N})")
_put_range("CfourDesTenC", rf"\| 10C \| ({_N})–({_N})")
_put_range("CfourCovOnlyBefore", rf"bias: ({_N})…({_N}) →", 3, sign=True)
_put_range("CfourCovOnlyAfter", rf"→ ({_N})…({_N}) in binary", 3, sign=True)
_put_range("CfourCovOnlyFMfiveThree", rf"FM5 ({_N}) → ({_N}) at 3C", 3, sign=True)
bias = md_rows(section(PCR, "**The bias being corrected**", r"\n\nIn accuracy"))
one_c = [_sg(r[1]) for r in bias[1:] if r and r[0].startswith("FM")]
put("CfourBiasOneLo", f(min(one_c), 3, sign=True), PC.name + " Results (bias table, 1C)")
put("CfourBiasOneHi", f(max(one_c), 3, sign=True), PC.name + " Results (bias table, 1C)")
m = re.search(r"New runs\.\*\* `reports/p1_corrections_runs.py`: ([\d,]+) FM2 fits", PCR)
put("CfourNewFits", m.group(1).replace(",", "{,}"), PC.name + " Results")
_put_range("CfiveEMPriorErr", rf"error to the true complement prior is ({_N})–({_N})")
_put_range("CfiveCtxPriorErr", rf"against ({_N})–({_N}) for the plain context shares")
# losses: the source lists the smaller loss first, so name them by loss size, not Lo/Hi
_put_range("CfiveEMMacro", rf"({_N}) to ({_N}) for every FM × budget", 2, sign=True, names=("LossMin", "LossMax"))
_put_range("CfiveEMAcc", rf"Effect on accuracy:\*\* ({_N}) to ({_N})", 2, sign=True, names=("LossMin", "LossMax"))
_put_range("CfiveOracleICLMacro", rf"macro-F1 ({_N}) to ({_N});", 3, sign=True)
_put_range("CfiveOraclePFN", rf"oracle ({_N}) to ({_N}) in both metrics", 3, sign=True)
_put_range("CfiveUniformPFN", rf"for macro-F1: ({_N}) to ({_N})", 3, sign=True)
_put_range("CfiveOracleGapOne", rf"leaves it at ({_N}) to ({_N})", 2, sign=True)
_put_range("CfiveUniformICLMacro", rf"it \*hurts\* macro-F1 at 3–10C \(({_N}) to ({_N})\)", 3, sign=True)


# ---------------------------------------------- Study 7: seen-classes-only metric ----
SEEN = section(s7, "## Per-budget contrasts, seen_macro_f1", r"\n## ")
_seen = {}
for r in md_rows(SEEN)[1:]:
    m = re.fullmatch(r"strat − random (fm[1235])", r[0].strip())
    if m:
        _seen[m.group(1)] = [float(c.split("(")[0].replace("−", "-")) for c in r[1:5]]
for j, w in [(0, "One"), (1, "Three")]:
    vals = [v[j] for v in _seen.values()]
    put(f"SsevSeenGap{w}Min", f(min(vals), 3, sign=True), S7.name + " Per-budget contrasts, seen_macro_f1")
    put(f"SsevSeenGap{w}Max", f(max(vals), 3, sign=True), S7.name + " Per-budget contrasts, seen_macro_f1")

# ------------------------------------------- Reproducibility: seeds, fits, compute ----
# Read-only: provenance JSON files and the per-fit "seconds" column of the stored runs.
# One stored row = one fit of an FM on one labelled set, scored on the pool.
import json  # noqa: E402

import pandas as pd  # noqa: E402

RES = SIM / "results"


def _prov(sub: str) -> dict:
    files = sorted((RES / sub).glob("provenance_*.json"))
    return json.loads(files[0].read_text(encoding="utf-8"))


pv7, pv9 = _prov("study7"), _prov("study9/t1prime")
put("ReproBaseSeed", pv7["base_seed"], "results/study7/provenance_*.json (base_seed)")
if pv9["base_seed"] != pv7["base_seed"]:
    raise ValueError("Study 7 and Study 9 use different base seeds")
put("ReproNsSeven", pv7["seed_namespace"], "results/study7/provenance_*.json (seed_namespace)")
put("ReproNsNine", pv9["seed_namespace"], "results/study9/t1prime/provenance_*.json (seed_namespace)")
put("ReproNsNinePerm", pv9["perm_namespace"], "results/study9/t1prime/provenance_*.json (perm_namespace)")


def _fits(globs: list[str]) -> tuple[int, float]:
    n, sec = 0, 0.0
    for g in globs:
        for path in sorted(RES.glob(g)):
            d = pd.read_parquet(path, columns=["seconds"])
            n += len(d)
            sec += float(d["seconds"].sum())
    if n == 0:
        raise ValueError(f"no runs found for {globs}")
    return n, sec / 3600


_comp = {"Sfour": ["study4_stage_a/*.parquet", "study4_stage_b/*.parquet"],
         "Ssev": ["study7/a/*.parquet", "study7/b/*.parquet", "study7/c/*.parquet", "study7/p/*.parquet"],
         "Snine": ["study9/t1prime/*.parquet"],
         "Corr": ["p1_corrections/design/*.parquet", "p1_corrections/random/*.parquet"]}
_tot_n, _tot_h = 0, 0.0
for stem, globs in _comp.items():
    n, h = _fits(globs)
    _tot_n += n
    _tot_h += h
    put(f"Repro{stem}Fits", f"{n:,}".replace(",", "{,}"), "results/" + ", ".join(globs) + " (rows = fits)")
    put(f"Repro{stem}Hours", f"{h:.0f}", "results/" + ", ".join(globs) + " (sum of 'seconds' / 3600)")
# Study 12: stored rows include redraw attempts and reused contexts, so its fits and FM hours are
# taken from the Compute section of its generated report (parsed above).
_tot_n += int(_cmp.group(1).replace(",", ""))
_tot_h += float(_cmp.group(2))
put("ReproTotalFits", f"{_tot_n:,}".replace(",", "{,}"), "sum of the Repro*Fits sources and STUDY12_REPORT.md Compute")
put("ReproTotalHours", f"{_tot_h:.0f}", "sum of the Repro*Hours sources and STUDY12_REPORT.md Compute")

# ---------------------------------- E17 calibration, continuous moderator (read-only) ----
# Spearman rho over all T1' tables between the pool's distance from uniform (tv) and the
# stratified - random gap in calibration-in-the-large error (citl_l1); answers the post hoc
# median split. Source: exploratory/e17_metrics/e17_moderators.csv (written by e17_metrics.py).
import csv  # noqa: E402

_MOD = HERE / "exploratory" / "e17_metrics" / "e17_moderators.csv"
with _MOD.open(encoding="utf-8") as fh:
    _rows = [r for r in csv.DictReader(fh) if r["set"] == "t1prime" and r["metric"] == "citl_l1"]
for _m, _w in [("3", "Three"), ("10", "Ten")]:
    _r = [r for r in _rows if r["m"] == _m][0]
    _src = f"exploratory/e17_metrics/e17_moderators.csv (t1prime, citl_l1, m={_m})"
    put(f"CalRhoTvM{_w}", f(float(_r["rho_tv"]), 2), _src)
    put(f"CalRhoTvM{_w}Lo", f(float(_r["rho_tv_lo"]), 2), _src)
    put(f"CalRhoTvM{_w}Hi", f(float(_r["rho_tv_hi"]), 2), _src)
    put(f"CalRhoTvM{_w}P", pfmt(float(_r["p_tv"])), _src)
    put("CalRhoTvN", _r["n"], _src) if _w == "Three" else None
_all = [float(r["rho_tv"]) for r in _rows]
put("CalRhoTvMaxAll", f(max(_all), 2), "e17_moderators.csv (t1prime, citl_l1, max over m)")
put("CalRhoTvMinAll", f(min(_all), 2), "e17_moderators.csv (t1prime, citl_l1, min over m)")

# T1' between-table heterogeneity (Study 12 data; REML random effects per FM).
# Source: analysis/out/t1p_meta.csv (written by analysis/t1p_heterogeneity.py).
_HET = HERE / "analysis" / "out" / "t1p_meta.csv"
_hsrc = "analysis/out/t1p_meta.csv"
with _HET.open(encoding="utf-8") as fh:
    _h = list(csv.DictReader(fh))
_FMW = {"fm1": "FMone", "fm2": "FMtwo", "fm3": "FMthree", "fm5": "FMfive"}
put("THetK", _h[0]["k"], _hsrc)
for _met, _stem in [("macro_f1", "F"), ("accuracy", "Acc"), ("balanced_accuracy", "BA")]:
    _rs = [r for r in _h if r["metric"] == _met]
    _ra = [r for r in _rs if r["mult"] == "all"]
    put(f"THet{_stem}IMin", f(min(float(r["I2"]) for r in _rs), 2), _hsrc + f" ({_met}, min over FM x budget)")
    put(f"THet{_stem}IMax", f(max(float(r["I2"]) for r in _rs), 2), _hsrc + f" ({_met}, max over FM x budget)")
    put(f"THet{_stem}TauMin", f(min(float(r["tau"]) for r in _ra), 3), _hsrc + f" ({_met}, mean over budgets)")
    put(f"THet{_stem}TauMax", f(max(float(r["tau"]) for r in _ra), 3), _hsrc + f" ({_met}, mean over budgets)")
    put(f"THet{_stem}ShareMin", pct(min(float(r["share_pos"]) for r in _ra)), _hsrc + f" ({_met}, all, min over FM)")
    put(f"THet{_stem}ShareMax", pct(max(float(r["share_pos"]) for r in _ra)), _hsrc + f" ({_met}, all, max over FM)")
    for r in _ra:
        _w = _FMW[r["fm"]]
        put(f"THet{_stem}{_w}AllPILo", f(float(r["pi_lo"]), 3), _hsrc + f" ({_met}, {r['fm']}, all)")
        put(f"THet{_stem}{_w}AllPIHi", f(float(r["pi_hi"]), 3), _hsrc + f" ({_met}, {r['fm']}, all)")
        put(f"THet{_stem}{_w}AllShare", pct(float(r["share_pos"])), _hsrc + f" ({_met}, {r['fm']}, all)")
    if _met == "macro_f1":
        for r in [r for r in _rs if r["mult"] == "1"]:
            _w = _FMW[r["fm"]]
            put(f"THetF{_w}MOnePILo", f(float(r["pi_lo"]), 3), _hsrc + f" (macro_f1, {r['fm']}, m=1)")
            put(f"THetF{_w}MOnePIHi", f(float(r["pi_hi"]), 3), _hsrc + f" (macro_f1, {r['fm']}, m=1)")

# Under-miss of the stored Study 9 random draws at one label per class on T1'.
# Source: exploratory/verify/out/v_checks.txt (written by exploratory/verify/v_checks.py), "Claim k".
_vc = (HERE / "exploratory" / "verify" / "out" / "v_checks.txt").read_text(encoding="utf-8")
_vsrc = "exploratory/verify/out/v_checks.txt (Claim k)"
_vb = re.search(r"^binary: draws=(\d+) observed miss rate=([\d.]+) exact\(pool\)=([\d.]+).*?z=(-?[\d.]+)", _vc, re.M)
_vp = re.search(r"binary exact Poisson-binomial .*?two-sided ~([\d.]+)", _vc)
_vr = re.search(r"re-derived from sel_seed exactly: (\d+) of (\d+)", _vc)
put("VerMissBinDraws", _vb.group(1), _vsrc)
put("VerMissBinObs", f(float(_vb.group(2)), 2), _vsrc)
put("VerMissBinExp", f(float(_vb.group(3)), 2), _vsrc)
put("VerMissBinZ", f(float(_vb.group(4)), 1), _vsrc)
put("VerMissBinP", _vp.group(1), _vsrc)
assert _vr.group(1) == _vr.group(2)
put("VerMissRederived", _vr.group(1), _vsrc)

# Correction 4 on T1' (exploratory): ranges over the four FMs of the per-FM macros
# written by exploratory/c4_t1p/run_c4_t1p.py (GREG with k = 5 random runs, macro-F1).
_c4 = (HERE / "exploratory" / "c4_t1p" / "out" / "c4_t1p_numbers.tex").read_text(encoding="utf-8")
_c4src = "exploratory/c4_t1p/out/c4_t1p_numbers.tex (range over 4 FMs)"


def _c4vals(stem: str) -> list[float]:
    out = []
    for fm in ("PFNtwo", "ICL", "Kumo", "PFNthreefive"):
        m_ = re.search(r"\\newcommand\{\\ExCfourTp" + fm + stem + r"\}\{([^}]*)\}", _c4)
        out.append(float(m_.group(1).replace("$", "").replace("+", "")))
    return out


for _stem, _nm, _nd in [("FGregKFiveRatio", "Ratio", 2), ("FGregKFiveCover", "Cover", 2),
                        ("FGregKFiveMOneCover", "MOneCover", 2), ("FGregKFiveMOneRatio", "MOneRatio", 2),
                        ("FGregKFiveBias", "Bias", 4)]:
    _v = _c4vals(_stem)
    put(f"CfourTpGreg{_nm}Min", f(min(_v), _nd, sign=_nm == "Bias"), _c4src + f" {_stem}")
    put(f"CfourTpGreg{_nm}Max", f(max(_v), _nd, sign=_nm == "Bias"), _c4src + f" {_stem}")
_rd = [abs(v) for st in ("MOne", "MThree", "MFive", "MOneZero") for v in _c4vals("FCovOnlyRedrawAseven" + st)]
put("CfourTpRedrawAbsBiasMax", f(max(_rd), 3), _c4src + " FCovOnlyRedrawAseven, max |bias| over FMs and m")

# ---------------------------------------------------------------- write -----
lines = ["% AUTO-GENERATED by paper/p1/make_numbers.py -- do not edit by hand.",
         "% Each macro lists its source file in the comment.", ""]
for k in sorted(OUT):
    lines.append(f"\\newcommand{{\\{k}}}{{{OUT[k]}}} % {SOURCES[k]}")
(HERE / "numbers_p1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {len(OUT)} macros to {HERE / 'numbers_p1.tex'}")
