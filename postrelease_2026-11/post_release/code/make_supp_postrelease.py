"""Write supplement/postrelease_supp.tex from results/RESULTS.json (frozen analysis output). Formatting only."""
import json, os, sys
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(B, 'results', 'RESULTS.json')))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(B), 'supplement', 'postrelease_supp.tex')
def n(x, d=3):
    s = f"{x:+.{d}f}"; return s.replace('-', '$-$').replace('+', '$+$')
def ci(c): return f"[{n(c[0])}, {n(c[1])}]"
def p(x):
    if x < 1e-3:
        m, e = f"{x:.1e}".split('e'); return f"${m}{{\\times}}10^{{{int(e)}}}$"
    return f"{x:.3f}"
FM = [('fm1', 'TabPFN v2'), ('fm2', 'TabICLv2'), ('fm3', 'Kumo-Tabular'), ('fm5', 'TabPFN v3.5')]
T = r['tests']; c1 = r['H-cov1']; pf = r['per_fm']
L = []
A = L.append
A(r"""\FloatBarrier
\section{Post-release tables: collection, plan and full results}
\label{app:postrelease}
\paragraph{Collection.} The inclusion rule was written and hashed before any candidate was judged. A candidate is a
record of Zenodo, figshare, Mendeley Data or Dataverse whose first public version appeared on or after 2026-09-24, the
day after the release of Kumo-Tabular, the last of the four FMs; the date is the earliest version found through each
repository's version interface. A mechanical filter (file size, an outcome-like column with 2 to 10 classes and at
least 11 rows in the smallest) produced an ordered list of 593 records. We judged them strictly in that order against
fixed criteria (open licence, real data, a class outcome named by the authors, the encoding rule of the recent tables)
and stopped at the 40th eligible table, at position 546; every one of the 506 rejections carries a code and a reason.
No model was fitted during collection, and the only label statistic used was the class count. Each table was checked
by content hash, row overlap and a size signature against every earlier set of this paper and against the hash list
of the replication tables, with no match. Figshare gave 18 tables, Zenodo 16, Mendeley Data 4 and Dataverse 2.

\paragraph{Plan.} The analysis plan, the coefficients of both trait models and the pre-data prediction for every table
were written and hashed before any FM was run on these tables. The design is that of the recent-table study:
stratified, random and redraw labelling at $m\in\{1,3,5,10\}$, the same four FMs on the same CPU class, one thread per
fit. The plan has two families with Holm's correction within each. The primary family holds the ranking test of the
pre-data model (one-sided Spearman) and the equivalence of stratified and redraw labelling at one label per class
(TOST, margin $\pm0.02$, mean of the four FMs). The secondary family holds the balance part for both TabPFN models
and its order across FMs, balanced accuracy and accuracy on imbalanced tables ($C\pi_{\min}\le0.6$) for each FM, the
cost-of-misses ranking of the coverage part, the ranking test of the mechanism model and the rate at which the winner
of a clear comparison changes. A pilot on earlier tables gave a power of at least 0.8 for both primary tests at 35
tables. The plan also fixed the wording: a ranking test that passes with a calibration that fails supports the
claim that traits predict where the gap appears, not how large it is.

\paragraph{One deviation.} For one table, birdpresence (three discrete features), one labelled set holds identical
rows, and both TabPFN models refuse to fit it. The plan drops a table that fails three times (complete case), so
\PRnTables{} tables enter the analysis. The queue script stopped the whole run at that failure instead of skipping
the unit, and we changed it to skip and continue; the analysis code, frame, seeds and models were unchanged, and the
analysis ran once.

\paragraph{Two hosts.} The fits ran on two Apple-silicon CPUs of different generations, each unit (table and FM) on
one host only. The two generations do not give bit-identical results: in the later
composition experiment (Section~\ref{app:balancemech}), one unit that ran on both by accident differed by up to
0.007 in macro-F1, and every verdict was identical with either copy. Differences of this size are small against the
spread between tables, and every test here is paired within a table and FM, so both arms of each contrast ran on the
same host.

\paragraph{Release date of Kumo-Tabular.} The inclusion rule dated the release of Kumo-Tabular to 2026-09-23 and admitted
tables first published on or after 2026-09-24. A later check found an earlier creation date for its model repository
(2026-09-01) and a public release commit dated 2026-09-28; the release post is dated 29 September. All four models
were therefore trained before the first eligible table appeared, but three tables (citizenstatus, hfref,
menohealth; first published 25--27 September) appeared before Kumo-Tabular was public. Excluding them (exploratory,
36 tables) leaves the Spearman correlation of the mechanism model at 0.86, that of the pre-data model at 0.73
(0.72 with them), the mean remainder of the mechanism model at $-0.002$ ($-0.003$), and the mean macro-F1 gap of
each FM at $+0.041$, $+0.014$, $+0.027$ and $+0.057$ ($+0.040$, $+0.013$, $+0.025$, $+0.056$).
""")
A(r"""\begin{table}[ht]
\centering\scriptsize
\caption{Planned tests on the post-release tables (""" + str(r['n_tables']) + r""" tables). $p$: unadjusted; Holm: adjusted
within the family. Spearman tests are one-sided; Wilcoxon tests are two-sided, with the direction fixed in the plan; the equivalence
test is a paired TOST with margin $\pm0.02$. Order test: mean balance part of the two TabPFN models minus that of the
other two. Change of winner: share of clear comparisons whose winner changes between protocols, against the share
between two halves of the random repetitions (noise).}
\label{tab:postrelease-tests}
\begin{tabular}{@{}lllll@{}}
\toprule
Family & Test & Estimate & $p$ & Holm $p$ \\
\midrule""")
hp = r['H-pred']
rows = [
 ('Primary', 'Pre-data model ranks the gap', f"$\\rho={hp['rho']:.2f}$", 'H-pred'),
 ('Primary', 'Stratified $\\equiv$ redraw at $m=1$', f"{n(c1['bal1_mean'])} {ci(c1['bal1_ci'])}", 'H-cov1'),
 ('Secondary', 'Balance part, TabPFN v2', n(pf['fm1']['bal_mf']), 'H-bal.fm1'),
 ('Secondary', 'Balance part, TabPFN v3.5', n(pf['fm5']['bal_mf']), 'H-bal.fm5'),
 ('Secondary', 'Balance part larger for both TabPFN models', '', 'H-bal.order'),
] + [('Secondary', f'Balanced accuracy, {nm}', n(pf[f]['gap_ba']), f'H-BA.{f}') for f, nm in FM] \
  + [('Secondary', f'Accuracy, imbalanced, {nm}', n(pf[f]['gap_acc_imb']), f'H-acc.{f}') for f, nm in FM] + [
 ('Secondary', 'Cost of misses ranks the coverage part', f"$\\rho={r['H-miss']['rho_predcov_cov']:.2f}$", 'H-miss.cost'),
 ('Secondary', 'Mechanism model ranks the gap', f"$\\rho={r['M2']['rho']:.2f}$", 'H-pred.M2'),
 ('Secondary', 'Winner of clear comparisons changes', f"{r['H-rank']['rate']:.2f} vs {r['H-rank']['noise_rate']:.2f}", 'H-rank'),
]
for fam, name, est, k in rows:
    A(f"{fam} & {name} & {est} & {p(T[k]['p'])} & {p(T[k]['p_adj'])} \\\\")
A(r"""\bottomrule
\end{tabular}
\end{table}
""")
A(r"""\begin{table}[ht]
\centering\scriptsize
\caption{Per-FM results on the post-release tables: stratified minus random, mean over tables of the mean over
$m\in\{1,3,5,10\}$ (balance part over $m\in\{3,5,10\}$), with bootstrap 95\% intervals over tables. Macro-F1 gap and the
coverage part at $m=1$ (Cov.) are descriptive. BA: balanced accuracy. Acc., imb.: accuracy on the """ + str(r['n_imbalanced']) + r""" tables with
$C\pi_{\min}\le0.6$.}
\label{tab:postrelease-fm}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}llllllll@{}}
\toprule
FM & Macro-F1 gap & $p$ & Balance & BA & Acc., imb. & Acc., all & Cov., $m=1$ \\
\midrule""")
for f, nm in FM:
    v = pf[f]
    A(f"{nm} & {n(v['gap_mf'])} {ci(v['gap_mf_ci'])} & {p(T['D-gap.'+f]['p'])} & {n(v['bal_mf'])} & {n(v['gap_ba'])} & {n(v['gap_acc_imb'])} & {n(v['gap_acc_all'])} & {n(c1['per_fm'][f]['cov1'])} \\\\")
m2 = r['M2']; pr = r['pre92']; hm = r['H-miss']
A(r"""\bottomrule
\end{tabular}
\end{table}

\paragraph{Calibration.} The pre-data model predicted a mean gap of """ + n(hp['pred_mean']) + r""" against an observed
""" + n(hp['obs_mean']) + r""" (calibration slope """ + f"{hp['slope']:.2f} {ci(hp['slope_ci']).replace('$+$','')}" + r""", mean remainder """ + n(hp['mean_resid']) + " " + ci(hp['mean_resid_ci']) + r"""), so it ranks
the tables but overstates the size of the gap. The mechanism model, whose coefficients were also fixed in advance, has
slope """ + f"{m2['slope']:.2f} {ci(m2['slope_ci']).replace('$+$','')}" + r""" and mean remainder """ + n(m2['mean_resid']) + " " + ci(m2['mean_resid_ci']) + r""". It differs
from the pre-data model in two terms, the cost of misses and the signal of the table; the share of draws that miss a
class alone correlates negatively with the coverage part ($\rho=""" + f"{hm['rho_missshare_cov']:.2f}".replace('-', '-') + r"""$), whereas the cost of
misses ranks it ($\rho=""" + f"{hm['rho_predcov_cov']:.2f}" + r"""$). Refitted on all 92 earlier tables, the pre-data
model ranks the post-release tables at $\rho=""" + f"{pr['rho']:.2f}" + r"""$ with a mean remainder of """ + n(pr['mean_resid']) + r""" (exploratory),
so adding the recent tables to its fit removes most of the overstatement. The share of random draws that miss a
class, counted in the stored draws, matched its exact hypergeometric value (mean difference
""" + n(hm['mean_diff']) + " " + ci(hm['ci']) + r""").
""")
open(out, 'w').write('\n'.join(L) + '\n')
print('wrote', out)
