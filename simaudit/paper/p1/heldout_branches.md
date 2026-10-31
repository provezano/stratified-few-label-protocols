# HELD-OUT outcome branches for P1 (prepared 2026-10-30, before any HELD-OUT result exists)

> Plan last changed at R025, before any HELD-OUT outcome was computed or viewed. Line numbers refer to an earlier draft of the paper.

This file is written before any HELD-OUT result exists, and nobody has looked at partial results. It holds the text that `main.tex` will need for each outcome that the pre-registrations allow, so that the final fill-in only picks a branch and defines the macros.

- Every number is a placeholder macro `\HO...`. No value is filled in. The macro list is in Sec. 5.
- Line numbers refer to `main.tex` at commit R026 (after the blind-review revision of 2026-10-30).
- Sources of the rules:
  - `prereg/STUDY13_HELDOUT_CONFIRMATION.md` Sec. 4.1–4.4, Amendment 1 B.8 and C.3, Amendment 3 C.1–C.5;
  - `prereg/STUDY9_HOW_MANY_LABELS.md` Sec. 6 and Sec. 9;
  - `prereg/JOINT_OPENING_2026-10-30.md` Sec. 3 (cross-study Holm family of 19).
- Style: the plain-style rules apply as before. Pre-registered FMs come first. The three secondary FMs are never called "confirmed".

## 0. Decision rules used below (copied from the preregs, not changed)

**Study 13, per contrast (Sec. 4.1).** Holm over the 6 primary contrasts.
- (i) per FM, and (ii-a): *replicated* if Holm p < 0.05 and mean Δ > 0; *reversed* if Holm p < 0.05 and mean Δ < 0; *not replicated* otherwise.
- (ii-c): *tie established* if the Holm TOST p < 0.05; *difference* if not equivalent and the two-sided Wilcoxon p < 0.05 (unadjusted, with its sign); *unresolved* otherwise.
- *Not evaluable:* more than 10% of the 32 tables lost for that contrast (Sec. 1). This is not the same as "not replicated".

**Study 13, claims (Sec. 4.2–4.3).**
- C1 replicates iff (i) is replicated for at least 3 of the 4 FMs and no (i) is reversed. C1 does not replicate if fewer than 3 are replicated. C1 is reversed if any (i) is reversed. (By Sec. 4.3, a reversal is also a non-replication; we name it separately because the text differs.)
- Ranking claim: *supported* if (ii-a) is replicated and the (ii-c) tie is established; *half supported* if (ii-a) is replicated and the tie is not established; *not supported* if (ii-a) is not replicated.
- Modifiers:
  - *attenuated:* HO mean / DEV mean < 0.5 for a replicated contrast (Sec. 4.2);
  - *fragile:* the verdict changes under SENS-INT (Sec. 1);
  - *label-fragile:* the verdict changes under SENS-LABEL (Amendment 1 A.6).
- If C1 does not replicate, the abstract and conclusion must say so, with the HO estimates and CIs (Sec. 4.3). If the ranking claim is not supported, the contributions drop "changes rankings" for this pair, and the flip rate is still reported (Sec. 4.3).

**Wording (Study 13 Sec. 4.4; opening note Sec. 3).**
- *"confirmed on HELD-OUT"* is used only if the verdict also holds with the cross-study Holm p-values: 19 tests, namely Study 13 ×6, Study 9 ×9 and Study 10 ×4.
- *"replicated on HELD-OUT (study-wise), not under the cross-study correction"* is used if it holds only with the within-study Holm.
- Both adjusted p-values are reported in every case.
- For C1 the rule is applied to the count: C1 is confirmed if at least 3 FMs are replicated under the cross-study Holm, and none is reversed under it.
- The ranking claim is confirmed or replicated at the level that both (ii-a) and (ii-c) reach.

**Study 13 secondary FMs (Amendment 3 C.1–C.5).**
- S-10: Holm over {fm6, fm7, fm8}. "The optimism extends to fmX" iff fmX is Holm-significant with Δ > 0.
- S-11: Holm over 3, with no predicted direction.
- Neither enters any confirmatory family. An FM with no analysable table stays in the family with p = 1 and is "not evaluable".
- For all three, HELD-OUT is a replication on new data, not a blind test.

**Study 9 (Sec. 9).**
- The HELD-OUT run uses the same code and seeds rule, with Holm within its 9-test HELD-OUT family and across studies.
- *Replication* means the same decision direction as on EXP (T1′).
- Non-replication of the C and IR moderators (H1b, H3b) is expected from range restriction. It is reported, not explained away.
- What P1 reports from EXP:
  - H2a stratified arm: supported;
  - magnitude rule: it outputs "about 2.6", which lies outside the CI, so P1 reports the estimate (a declared deviation);
  - H3b: supported;
  - H1b, H2a for S-KMn and S-LUn, H2b ×2 and H4 ×2: not supported.

**Study 10.** P1 states no Study 10 result. The only mention is the counting note at L3229. Study 10 therefore enters P1 only through the size and members of the cross-study Holm family. P1 needs no Study 10 branch.

## 1. Inventory of placeholders in main.tex

There are 3 `\TODO` boxes. There are also 13 other places whose text depends on HELD-OUT.

| # | Line(s) | Where | Current text (short) | Kind |
|---|---|---|---|---|
| P1 | 111 | Abstract | "and \HeldOutN{} sealed tables drawn like DEV (pending)" | pending wording |
| P2 | 129 | Abstract | `\TODO{HELD-OUT confirmation sentence ...}` | **\TODO** |
| P3 | ≈181 | Contributions, item 2 | "(confirmation on held-out tables from the same pool is pending)" | pending wording |
| P4 | ≈183–186 | Contributions, item 2 (ranking) | "The winner of a pairwise comparison changes in \StwelveFlipF{} of the comparisons (pre-declared secondary); for the pre-registered pair, the pre-registered verdict under stratified draws on PMLB/UCI is a difference, not a tie." | add HO ranking verdict |
| P5 | ≈585–596 | Sec. 5, Data | "HELD-OUT contains \HeldOutN{} tables that remain sealed. They are held-out tables from the same pool as DEV, not an external test. They are ... the only test on unseen tables. The HELD-OUT study repeats the protocol comparison of the DEV study (not the mechanism); we expect low power ... Before the opening, we fixed by the pre-registered decision rules how each claim of the abstract will be worded ..." | sealed wording, future tense |
| P6 | 600 | Sec. 5, Data | `\TODO{HELD-OUT: report the confirmatory replication ...}` | **\TODO** |
| P7 | new | Sec. res-bias and res-ranks ("Under stratified draws, a model advantage is smaller") | (no text yet) | new result sentences |
| P8 | ≈1004 | Limitations | "Only HELD-OUT (pending; same pool as DEV) tests unseen tables." | pending wording |
| P9 | ≈1028–1030 | Conclusion | "Confirmation on held-out tables is pending." plus `\TODO{Update after HELD-OUT ...}` | **\TODO** |
| P10 | 1645 | Claims table, row "Gap S−R" | "HELD-OUT pending" | table cell |
| P11 | 3392 ff. | tab:hyp | no Study 13 block; Study 9 row is PMLB/UCI only | new table rows |
| P12 | 3366 ff. | App, results of Study 9 family | PMLB/UCI only | add HO sentence |
| P13 | 2755 | App fmextra | "HELD-OUT replicates their gaps on new data" (future) | add S-10/S-11 sentence |
| P14 | 3270 | App prereg, Study 9 bullet | no HELD-OUT run record | add run record |
| P15 | 3310 | App prereg, Study 13 bullet | "HELD-OUT is pending." | run record |
| P16 | 3437 | fig:heldout (fig8) | placeholder figure and caption | figure + caption |

The Models paragraph ("for them HELD-OUT is a replication on new data") stays the same in every branch.

The placeholder boxes are P2, P6 and P9; P16 and P10 are the other two items.

**Page budget.** The main text ends exactly on p.12. P2, P7 and P9 add text to the main body. Each branch below has a short form (main text) and a long form (appendix, new `app:heldout` next to fig:heldout). Put the detail in the appendix. Any line added on a page must be saved on that same page (lesson from the seven-FM round).

## 2. Study 13 branches

**Naming (since blind review 2, 2026-10-30).** The main text no longer uses study numbers, E-codes or the codes L0/H1/H2. In main-text sentences below, write "the DEV study" for Study 7, "the learning-curve study" for Study 9, "the PMLB/UCI study" for Study 12 (the set T1′ is now called PMLB/UCI in the main text since blind review 3), "the HELD-OUT study" for Study 13, and "the protocol comparison" for Part (a); the numbers stay in the appendix (Table `tab:studies`). Abstract finding (1) now ends with the redraw sentence (TabICLv2 and Kumo-Tabular gaps mostly from coverage), and finding (2) reads (since blind review 3) "Because the gap differs between models, the protocol changes comparisons: on PMLB/UCI, a different model wins \StwelveFlipF{} ... (pre-declared secondary). For the pre-registered pair, chosen from earlier DEV work, TabICLv2 beats TabPFN v3.5 under random draws ..., but the pre-registered tie under stratified draws is not established." Put the HELD-OUT C1 sentence after finding (1), and the ranking verdict at the end of the pre-registered part of finding (2).

**Since blind review 4 (2026-10-30).**
- Abstract finding (2) now reads: "(2) Pre-declared secondary: because the gap differs between models, the winner of a pairwise comparison changes between the protocols in \StwelveFlipF{} of the comparisons on PMLB/UCI, against \StwelveFlipFNoise{} between two halves of the random draws. For the pre-registered pair, chosen from earlier DEV work, TabICLv2 beats TabPFN v3.5 under random draws at three to ten labels per class (...); under stratified draws, the pre-registered verdict on PMLB/UCI is a difference, with TabPFN v3.5 slightly ahead (\StwelveIIcMean), and on DEV no tie is established." Add the HELD-OUT ranking verdict at the end of this finding, with the HO (ii-c) verdict word as pre-registered (tie / difference / unresolved), never only "not established" when the verdict is "difference".
- The exploratory redraw sentence moved from finding (1) to finding (3). Finding (3) states the accuracy gap by pool ("raises accuracy on near-balanced pools but lowers it on imbalanced ones"); HELD-OUT does not test it.
- The flip rate counts six learners (four FMs and the two TabPFN models with balance_probabilities); the four-FM rate is \StwelveFlipFFM{} against \StwelveFlipFFMNoise.
- The main text has a results table (`tab:main`): per pre-registered FM, the pre-registered contrast on DEV and PMLB/UCI with CIs. If HELD-OUT is reported in the main text, add one column "HELD-OUT (P)" with \HOIFM{one,two,three,five}Mean and CIs to `tab:main` instead of a min-to-max range in the prose, and save the width elsewhere (drop the brackets of the DEV column first).
- The mechanism rule now reports the literal pre-registered verdicts (TabICLv2 and Kumo-Tabular "beyond a prior shift", the TabPFN models "unresolved"; macros `\ExRvLitRule*`) and two deviations; HELD-OUT does not re-test the mechanism, so no branch changes it.

**Difference in differences (since blind review 3).** The DiD of TabPFN v3.5 and TabICLv2 is no longer a result: it equals the difference of their gaps in finding (1) and appears only as an identity check in app:redraw (`\ExRvDidDev`, `\ExRvDidTp`). Do **not** report `\HODidMean`/`\HODidLo`/`\HODidHi` in the main text; these macros are redundant. If computed, at most one identity-check clause in app:redraw.

**Replication wording (since blind review 3).** PMLB/UCI is new draws on tables the learning-curve study had already seen, not an independent replication: in main-text sentences about PMLB/UCI write "holds again with new draws", never "replicates". HELD-OUT is the only test on unseen tables, so "replicates" stays correct for HELD-OUT (and for the pre-registered verdict words).

Macro conventions:
- `\HON` = tables analysed. Per contrast X:
  - `\HOXMean`, `\HOXLo`, `\HOXHi`: mean and bootstrap CI;
  - `\HOXRaw`: unadjusted p;
  - `\HOXHolm`: within-study Holm p;
  - `\HOXXHolm`: cross-study Holm p;
  - `\HOXVerdict`: the verdict word.
- `\HOINRep` = number of the four FMs replicated, within-study. `\HOINRepX` = the same number under the cross-study Holm.
- `\HOIMinMean` and `\HOIMaxMean` = the range of the (i) means over the four FMs.

### 2.1 C1 (optimism in macro-F1), a sentence for the abstract (P2)

**C1-a. Replicates, and holds under the cross-study Holm ("confirmed").**
> On \HON{} sealed HELD-OUT tables, opened once under the pre-registration, the overstatement is confirmed for \HOINRepX{} of the four pre-registered FMs (\HOIMinMean{} to \HOIMaxMean, Holm-adjusted across the studies that share the opening).

**C1-b. Replicates only study-wise.**
> On \HON{} sealed HELD-OUT tables, opened once under the pre-registration, the overstatement replicates for \HOINRep{} of the four pre-registered FMs (\HOIMinMean{} to \HOIMaxMean; Holm within the study), but not under the correction across the studies that share the opening.

**C1-c. Does not replicate (fewer than 3 FMs, none reversed).**
> On \HON{} sealed HELD-OUT tables, the overstatement replicates for only \HOINRep{} of the four pre-registered FMs (\HOIMinMean{} to \HOIMaxMean), so by our pre-registered rule it is not confirmed beyond DEV and PMLB/UCI.

**C1-d. Reversed (at least one FM with Holm p < 0.05 and Δ < 0).**
> On \HON{} sealed HELD-OUT tables, the stratified protocol lowers macro-F1 for \HOINRev{} of the four pre-registered FMs (\HOIMinMean{} to \HOIMaxMean), so the overstatement shown on DEV and PMLB/UCI does not hold on these tables.

**C1-e. Not evaluable (more than 10% of tables lost for 2 or more FMs, so that 3 replications cannot be reached).**
> On HELD-OUT, too many tables failed to evaluate the overstatement for \HOINNotEval{} of the four FMs, so the pre-registered test is not evaluable.

Modifiers. Add the clause after the C1 sentence; it may replace the CI in the main text.
- *Attenuated* (some replicated FM with ratio < 0.5): "...; for \HOINAtten{} of them, the gap is less than half its DEV size."
- *Fragile* (the C1 verdict changes under SENS-INT): "...; without the \HOSensIntNDrop{} tables flagged by the integrity screen, the verdict is \HOSensIntCOneVerdict."
- *Label-fragile* (the C1 verdict changes under SENS-LABEL): "...; without the \HOSensLabelNDrop{} tables whose labels were merged, the verdict is \HOSensLabelCOneVerdict."

### 2.2 Ranking claim, a sentence for the abstract and contributions (P2, P4)

Current P4 text: "by the pre-registered rules, the claim that the protocol changes this ranking is not supported on DEV and \StwelveRankingVerdict{} on PMLB/UCI (Section~\ref{sec:design})." Replace "and \StwelveRankingVerdict{} on PMLB/UCI" with "\StwelveRankingVerdict{} on PMLB/UCI and \HORankingVerdict{} on HELD-OUT", or append one of the following when the wording level must be stated. The verdict words are defined once in the Statistics paragraph (Sec. 5): supported, half supported, not supported.

**R-a. Supported, confirmed (both (ii-a) and the (ii-c) tie hold under the cross-study Holm).**
> On HELD-OUT, the claim is supported and confirmed under the cross-study correction: TabICLv2 is ahead under random draws (\HOIIaMean, Holm $p=$ \HOIIaXHolm), and the two models tie under stratified draws (\HOIIcMean{} [\HOIIcLo, \HOIIcHi], TOST Holm $p=$ \HOIIcXHolm).

**R-b. Supported study-wise only.**
> On HELD-OUT, it is supported study-wise (\HOIIaMean, Holm $p=$ \HOIIaHolm; tie \HOIIcMean{} [\HOIIcLo, \HOIIcHi], TOST Holm $p=$ \HOIIcHolm), but not under the cross-study correction (\HOIIaXHolm, \HOIIcXHolm).

**R-c. Half supported ((ii-a) replicated; tie not established, (ii-c) unresolved).**
> On HELD-OUT, it is half supported: TabICLv2 is ahead under random draws (\HOIIaMean, Holm $p=$ \HOIIaHolm), but a tie under stratified draws is not established (\HOIIcMean{} [\HOIIcLo, \HOIIcHi]).

**R-c′. Half supported, and (ii-c) shows a difference.** (This is still "half supported" by Sec. 4.2. The sign of the difference is stated.)
> On HELD-OUT, it is half supported: TabICLv2 is ahead under random draws (\HOIIaMean, Holm $p=$ \HOIIaHolm), and under stratified draws \HOIIcLeader{} is ahead (\HOIIcMean{} [\HOIIcLo, \HOIIcHi], Wilcoxon $p=$ \HOIIcWilcoxonP, unadjusted).

**R-d. Not supported ((ii-a) not replicated or reversed).**
> On HELD-OUT, it is not supported: under random draws, the gap between TabICLv2 and TabPFN v3.5 is \HOIIaMean{} (Holm $p=$ \HOIIaHolm; \HOIIaVerdict).

If R-d applies, Sec. 4.3 also requires (and the Data paragraph already promises it):
- In contribution item 2, keep the flip-rate sentence and the measured DEV/PMLB/UCI facts. Append: "On HELD-OUT, TabICLv2 is not ahead of TabPFN v3.5 under random draws."
- Abstract finding (2): after "TabICLv2 beats TabPFN v3.5 under random draws at three to ten labels per class (...)" add "on DEV and PMLB/UCI, but not on HELD-OUT".
- Section title "The protocol changes comparisons between models" and Recommendation 4 ("An advantage can shrink, grow or reverse") stay, because they rest on the PMLB/UCI flip rate, not on this pair.
- Keep the flip-rate sentence (S-9).

Combining the abstract sentences for P2. Since 2026-10-30 the abstract lists five numbered findings. Put the HELD-OUT result as one sentence after finding (1), and add the ranking result to finding (2); target 2 lines in all:
- C1-a or C1-b, then "The ranking claim is \HORankingVerdict{} there." Use the "confirmed"/"study-wise" wording of R-a/R-b only if space allows.
- C1-c, C1-d or C1-e: put the non-replication first, then the ranking clause.

### 2.3 Sec. 5 Data (P5, P6)

P5, replace "that remain sealed." with "that were sealed until a single joint opening on 2026-10-30 for Studies~9, 10 and~13, after every pre-registration involved had been frozen (Appendix~\ref{app:prereg})." Keep "They are held-out tables from the same pool as DEV, not an external test." Replace the future-tense sentences ("The HELD-OUT study repeats ... we expect low power ... Before the opening, we fixed ... we will drop the ranking claim ...") by:
- R-a, R-b or R-c: "The HELD-OUT study repeats the protocol comparison of the DEV study; we expected low power for the tie in the ranking claim (Section~\ref{sec:res-ranks}). Each claim of the abstract is worded by the pre-registered decision rules, as fixed before the opening."
- R-d: "As its pre-registration requires, we drop the ranking claim, because TabICLv2 is not ahead under random draws there."
In every branch keep, in the past tense: "(not the mechanism)" after "repeats the protocol comparison of the DEV study", and "the only test on unseen tables". No difference-in-differences sentence (removed in blind review 3).

P6, replace the `\TODO` with (same in every branch):
> The HELD-OUT study (Study~13 in Table~\ref{tab:studies}) repeats the protocol comparison of the DEV study on HELD-OUT (Section~\ref{sec:res-bias}; Appendix~\ref{app:heldout}).

### 2.4 Results (P7), main text: one sentence in sec:res-bias and one in sec:res-ranks

In sec:res-bias, after the PMLB/UCI sentence on the (i) gaps:

- **C1-a:** "On the \HON{} HELD-OUT tables, the gap is \HOIMinMean{} to \HOIMaxMean{} and is significant for \HOINRepX{} of the four FMs after the cross-study Holm correction (Figure~\ref{fig:heldout}; Appendix~\ref{app:heldout})."
- **C1-b:** "On the \HON{} HELD-OUT tables, the gap is \HOIMinMean{} to \HOIMaxMean{} and is significant for \HOINRep{} of the four FMs with Holm's correction within the study, but for only \HOINRepX{} with the cross-study correction (Appendix~\ref{app:heldout})."
- **C1-c:** "On the \HON{} HELD-OUT tables, the gap is \HOIMinMean{} to \HOIMaxMean{} and is significant for only \HOINRep{} of the four FMs, fewer than the three that our rule requires (Appendix~\ref{app:heldout})."
- **C1-d:** "On the \HON{} HELD-OUT tables, the gap is \HOIMinMean{} to \HOIMaxMean, and it is significantly negative for \HOINRev{} of the four FMs (Appendix~\ref{app:heldout})."
- **C1-e:** "On HELD-OUT, the gap could not be evaluated for \HOINNotEval{} of the four FMs because more than 10\% of the tables failed (Appendix~\ref{app:heldout})."

In sec:res-ranks, after the PMLB/UCI verdict, use the R-a to R-d sentence of Sec. 2.2 in its short form: "On HELD-OUT, the ranking claim is \HORankingVerdict{} (Appendix~\ref{app:heldout})." Do not add a difference-in-differences clause (removed in blind review 3; see the note at the top of Sec. 2).

### 2.5 Limitations (P8)

Replace "Only HELD-OUT (pending; same pool as DEV) tests unseen tables." with:
- **C1-a or C1-b:** "HELD-OUT (\HON{} tables) tests unseen tables, but it was drawn from the same frame as DEV (small numeric OpenML tables), so it shows that DEV was not a sampling accident, not that the result holds beyond that frame."
- **C1-c, C1-d or C1-e:** "On the unseen HELD-OUT tables (\HON, drawn from the same frame as DEV), the overstatement did not replicate (Section~\ref{sec:res-bias}), so the DEV and PMLB/UCI results may not hold on new tables of that frame."

The "same frame" clause comes from Study 13 Sec. 10, so it applies in every branch.

### 2.6 Conclusion (P9)

Replace "Confirmation on held-out tables is pending." and the `\TODO` with (since blind review 4 the conclusion has no \emph{} lead-ins: drop "\emph{Beyond these tables}," and start with "On \HON{} sealed HELD-OUT tables, ..."):
- **C1-a:** "\emph{Beyond these tables}, the overstatement is confirmed on \HON{} sealed HELD-OUT tables for \HOINRepX{} of the four pre-registered FMs."
- **C1-b:** "\emph{Beyond these tables}, it replicates on \HON{} sealed HELD-OUT tables for \HOINRep{} of the four pre-registered FMs, but not under the correction across the studies that share that set."
- **C1-c:** "\emph{Beyond these tables}, it does not replicate on \HON{} sealed HELD-OUT tables (\HOINRep{} of four FMs; \HOIMinMean{} to \HOIMaxMean)."
- **C1-d:** "\emph{Beyond these tables}, it is reversed on \HON{} sealed HELD-OUT tables for \HOINRev{} of four FMs (\HOIMinMean{} to \HOIMaxMean)."
- **C1-e:** "\emph{Beyond these tables}, the HELD-OUT test is not evaluable (too many failed tables)."

Under R-d, the conclusion clause "the winner of a comparison changes in \StwelveFlipF{} of the comparisons on PMLB/UCI" stays (it rests on the flip rate, not on this pair).

Under C1-c, C1-d or C1-e, abstract finding (1) already names DEV and PMLB/UCI; add the HELD-OUT non-replication right after it, as Sec. 4.3 requires, and keep "the three exploratory models show the same" scoped to DEV and PMLB/UCI.

### 2.7 Claims table (P10)

Row "Gap S−R": replace "HELD-OUT pending" with:
- C1-a: "HELD-OUT confirmed (\HOINRepX/4)"
- C1-b: "HELD-OUT study-wise (\HOINRep/4)"
- C1-c: "not replicated on HELD-OUT (\HOINRep/4)"
- C1-d: "reversed on HELD-OUT (\HOINRev/4)"
- C1-e: "HELD-OUT not evaluable"

Append "attenuated" or "fragile" when a modifier applies. If the row overflows the float again (it did by 8.5pt on 2026-10-29), shorten "PMLB/UCI mostly binary" first.

### 2.8 tab:hyp, a new Study 13 block (P11)

Same columns as the Study 12 block. Insert it after the Study 12 secondary block, as `\multicolumn{7}{l}{\emph{Study 13 (pre-registered, HELD-OUT, \HON{} tables; within-study Holm, cross-study Holm in brackets)}}`:
```
(i) v2 & $\Delta$, mean over $m\in\{1,3,5,10\}$ & Wilcoxon & S13 & \HOIFMoneRaw & \HOIFMoneHolm{} [\HOIFMoneXHolm] & \HOIFMoneVerdict, \HOIFMoneMean \\
(i) TabICLv2 & same & Wilcoxon & S13 & \HOIFMtwoRaw & \HOIFMtwoHolm{} [\HOIFMtwoXHolm] & \HOIFMtwoVerdict, \HOIFMtwoMean \\
(i) Kumo & same & Wilcoxon & S13 & \HOIFMthreeRaw & \HOIFMthreeHolm{} [\HOIFMthreeXHolm] & \HOIFMthreeVerdict, \HOIFMthreeMean \\
(i) v3.5 & same & Wilcoxon & S13 & \HOIFMfiveRaw & \HOIFMfiveHolm{} [\HOIFMfiveXHolm] & \HOIFMfiveVerdict, \HOIFMfiveMean \\
(ii-a) & TabICLv2 $-$ v3.5 under R, $m\in\{3,5,10\}$ & Wilcoxon & S13 & \HOIIaRaw & \HOIIaHolm{} [\HOIIaXHolm] & \HOIIaVerdict, \HOIIaMean \\
(ii-c) & TabICLv2 $-$ v3.5 under S (tie) & TOST $\pm0.01$ & S13 & \HOIIcRaw & \HOIIcHolm{} [\HOIIcXHolm] & \HOIIcVerdict, \HOIIcMean{} [\HOIIcLo, \HOIIcHi] \\
```
Below these rows add one row for the Study 9 HELD-OUT replication (see 3.3), and in the caption add: "Brackets: Holm over the \HOXFamN{} confirmatory tests of the studies that share HELD-OUT."

### 2.9 Secondary FMs, S-10 and S-11 (P13, and P2/P7 if space allows)

These are never called "confirmed". Use "pre-registered secondary" and "replication on new data". Per FM X ∈ {fm6 TabDPT, fm7 Mitra, fm8 LimiX}, the macros are `\HOSTenFMsixMean`, `\HOSTenFMsixHolm` and so on.

S-10, at P13 after "...it does not test a blind prediction.":
- **All three extend** (all Holm p < 0.05, Δ > 0): "On HELD-OUT, the gap is \HOSTenFMsixMean{} for TabDPT, \HOSTenFMsevenMean{} for Mitra and \HOSTenFMeightMean{} for LimiX, significant for all three with Holm's correction over the three (largest $p=$ \HOSTenHolmMax), so the overstatement extends to them on new data."
- **Some extend:** "On HELD-OUT, the gap is significant for \HOSTenExtendList{} (Holm over the three secondary FMs) but not for \HOSTenNotList{} (\HOSTenNotMean, Holm $p=$ \HOSTenNotHolm)." Use the names; do not use "partly confirmed".
- **None extends:** "On HELD-OUT, the gap is \HOSTenMinMean{} to \HOSTenMaxMean{} and is not significant for any of the three (Holm over three; smallest $p=$ \HOSTenHolmMin)."
- **Against the predicted sign** (Holm p < 0.05 and Δ < 0 for some FM): "For \HOSTenRevList, stratified sampling lowers macro-F1 on HELD-OUT (\HOSTenRevMean, Holm $p=$ \HOSTenRevHolm), against the predicted sign."
- **Not evaluable** (an FM with n = 0, or more than 10% of tables lost): "\HOSTenNotEvalList{} could not be evaluated on HELD-OUT (failed jobs); it stays in the Holm family with $p=1$."
- Every S-10 branch adds the unadjusted accuracy gap: "In accuracy (unadjusted), the gaps are \HOSTenAccFMsixMean, \HOSTenAccFMsevenMean{} and \HOSTenAccFMeightMean."

S-11 (no predicted direction), one sentence after S-10:
- **None significant:** "Their gap minus the mean gap of the classical learners is \HOSElevenMinMean{} to \HOSElevenMaxMean, not significant for any (Holm over three)."
- **Some significant:** "Their gap minus the mean gap of the classical learners is significantly \HOSElevenSignWord{} for \HOSElevenSigList{} (\HOSElevenSigMean, Holm $p=$ \HOSElevenSigHolm) and not significant for the others."

Agreement with DEV and PMLB/UCI, as one clause when the HO sign differs from what we already report:
- "...which agrees with DEV and PMLB/UCI"
- or "...unlike on DEV and PMLB/UCI, where \HOSTenDisagreeList{} had a positive gap".

Main text, only if the abstract or the results have room: "The three secondary FMs show the gap on HELD-OUT for \HOSTenNExtend{} of three (pre-registered secondary; replication on new data)."

Seven-FM claims currently in main.tex (abstract finding (1), "the three exploratory models show the same"; conclusion L1036, "overstate the macro-F1 of all seven FMs"):
- If S-10 holds for all three, add "and on HELD-OUT".
- Otherwise keep "on DEV and PMLB/UCI" explicit, and add "on HELD-OUT for \HOSTenNExtend{} of the three secondary FMs".

### 2.10 Pre-declared secondary analyses S-1 to S-9 and S-12 to S-14 (appendix app:heldout only)

One template per family. Pick the clause by the outcome. Each line names the DEV/PMLB/UCI claim in P1 that it checks.
- **S-1 classical learners (E1):** "learner-agnostic" is claimed iff at least 5 of 6 are Holm-significant with Δ > 0.
  - Met: "The bias appears in \HOSOneNSig{} of the six classical learners (Holm over six), as on DEV."
  - Not met: "...in only \HOSOneNSig{} of six, so we do not claim on HELD-OUT that it is learner-agnostic."
- **S-2 AUC over all classes, m = 1 (E2):** "\HOSTwoNSig{} of four FMs significant (Holm over four), as predicted / against the prediction for \HOSTwoNotList."
- **S-3 accuracy gap against TV (E6):**
  - Predicted ρ < 0 holds: "Spearman $\rho=$ \HOSThreeRho{} (one-sided $p=$ \HOSThreeP), as on DEV."
  - Otherwise: "...; the association seen on DEV does not replicate."
  - The binary B* sign agreement is \HOSThreeBstarAgree, descriptive.
- **S-4 fixed split (arm F):**
  - "The gap on a fixed test set is \HOSFourMinMean{} to \HOSFourMaxMean, significant for \HOSFourNSig{} of four FMs (Holm over four);
  - the difference between the arms is \HOSFourArmDiffMin{} to \HOSFourArmDiffMax{} (inside / outside ±0.01)."
- **S-5 binary against multiclass:**
  - "The gap at one label per class is \HOSFiveBin{} on binary and \HOSFiveMulti{} on multiclass tables (Mann–Whitney $p=$ \HOSFiveMW);
  - larger for multiclass as predicted / not."
  - If a group has fewer than 6 tables: "descriptive only, \HOSFiveNMulti{} multiclass tables".
- **S-6 random-effects:** "prediction interval above 0 for \HOSSixPIAboveList, straddling 0 for \HOSSixPIStraddleList{} (macro-F1); expectation from DEV: above 0 for TabPFN v2, Kumo and v3.5, straddling 0 for TabICLv2."
- **S-7 E13 (FM gap minus classical):**
  - "D > 0 for TabPFN v3.5: \HOSSevenFMfiveMean{} (Holm $p=$ \HOSSevenFMfiveHolm), as predicted / not;
  - D < 0 for TabICLv2: \HOSSevenFMtwoMean{} (Holm $p=$ \HOSSevenFMtwoHolm), as predicted / not."
- **S-8 redraw:**
  - "Stratified minus redraw is significant for \HOSEightSigList{} (Holm over four);
  - predicted: TabPFN v2 and v3.5 only."
  - TabICLv2 − v3.5 under redraw: \HOSEightRank.
  - Infeasible redraw cells: \HOSEightNInfeasible.
- **S-9 flip rate:**
  - "\HOSNineF{} against \HOSNineFNoise{} for noise (difference \HOSNineFDiff, [\HOSNineFDiffLo, \HOSNineFDiffHi]);
  - above noise as predicted / not."
- **S-12 kappa and MCC:** "\HOSTwelveNSig{} of eight FM-by-metric tests significant with Δ > 0 (Holm over eight)."
- **S-13 all-class minus seen-class AUC, m = 1:** "D > 0 for \HOSThirteenNSig{} of four FMs (Holm over four)."
- **S-14 CITL on the more imbalanced half:** "\HOSFourteenNSig{} of four FMs significant with Δ > 0 (Holm over four; about 16 tables)."

Integrity report, in every branch: "\HOSensIntNFlag{} tables flagged by the integrity screen; \HOLabelNMerged{} with merged labels; \HOLabelNUnverified{} unverified." Add "SENS-LABEL+ is not reported because more than half of the tables are unverified" only if \HOLabelNUnverified > 16 (Amendment 2).

## 3. Study 9 HELD-OUT branches (P11, P12, P14; abstract only if space allows)

Macros:
- `\HOSnineN`;
- `\HOSnineRawHTwoaStrat`, `\HOSnineHolmHTwoaStrat` (within the HELD-OUT family of 9), `\HOSnineXHolmHTwoaStrat`;
- `\HOSnineRhoOneStrat` [`\HOSnineRhoOneStratLo`, `\HOSnineRhoOneStratHi`];
- `\HOSnineHolmHThreeb`, `\HOSnineHThreebTau`;
- `\HOSnineNOtherSupported`.

### 3.1 H2a, stratified arm (the Study 9 claim that P1 reports)
- **Replicates, confirmed** (Holm p < 0.05 also under the cross-study Holm): "On the \HOSnineN{} HELD-OUT tables, one stratified label per class again beats one random label in more than half of the tables (sign test, cross-study Holm $p=$ \HOSnineXHolmHTwoaStrat), so this result is confirmed on HELD-OUT."
- **Replicates study-wise:** "...again beats one random label in more than half of the tables (Holm $p=$ \HOSnineHolmHTwoaStrat{} within Study~9), but not under the cross-study correction ($p=$ \HOSnineXHolmHTwoaStrat)."
- **Does not replicate:** "On HELD-OUT, the sign test does not show that one stratified label per class beats one random label in more than half of the tables (Holm $p=$ \HOSnineHolmHTwoaStrat), so the PMLB/UCI result does not replicate."

Magnitude rule. Same thresholds: the rule outputs "about 2.6" iff the CI meets [2.0, 3.2]. Since 2026-10-30 the paper reports the estimate, not the rule's output, when 2.6 lies outside the CI (a declared deviation, Sec. res-rho and App. prereg).
- Rule outputs "about 2.6" and 2.6 is inside the CI: "the median of $\rho(1)$ is \HOSnineRhoOneStrat{} [\HOSnineRhoOneStratLo, \HOSnineRhoOneStratHi], and the pre-registered magnitude rule gives ``about 2.6'', which lies inside this interval."
- Rule outputs "about 2.6" but 2.6 is outside the CI: "the median of $\rho(1)$ is \HOSnineRhoOneStrat{} [\HOSnineRhoOneStratLo, \HOSnineRhoOneStratHi]; the rule would again give ``about 2.6'', outside the interval, so we report the estimate, as on PMLB/UCI."
- Rule does not output it: "the median of $\rho(1)$ is \HOSnineRhoOneStrat{} [\HOSnineRhoOneStratLo, \HOSnineRhoOneStratHi], so by the rule we report this estimate."

Abstract finding (5) (L125, "for TabICLv2 on PMLB/UCI, one stratified label per class is worth \SnineRhoOneStrat{} ..."):
- If the sign test replicates, add "and \HOSnineRhoOneStrat{} on HELD-OUT".
- Otherwise add "; on HELD-OUT the sign test does not replicate (\HOSnineRhoOneStrat; Appendix~\ref{app:heldout})".

FM-robust (Study 9 Sec. 6), as one clause in the appendix: "TabPFN v2 shows the same direction (unadjusted $p=$ \HOSnineFMoneStratP)" or "TabPFN v2 does not (unadjusted $p=$ \HOSnineFMoneStratP)".

### 3.2 The other eight Study 9 tests (P12, appendix)
- **H3b:**
  - Replicates: "H3b holds again (Kendall $\tau_b=$ \HOSnineHThreebTau, Holm $p=$ \HOSnineHolmHThreeb)."
  - Does not replicate: "H3b does not replicate (Kendall $\tau_b=$ \HOSnineHThreebTau, Holm $p=$ \HOSnineHolmHThreeb); HELD-OUT has a narrower range of imbalance than PMLB/UCI, which the pre-registration named as the likely reason, but we do not count it as an explanation."
- **H1b, H2a for the two selectors, H2b ×2, H4 ×2** (not supported on EXP):
  - All still not supported: "The other seven tests are again not supported, as on PMLB/UCI (replication of the same decisions)."
  - Some become supported: "Of the seven tests not supported on PMLB/UCI, \HOSnineNOtherSupported{} are supported on HELD-OUT (\HOSnineOtherSupportedList), so their PMLB/UCI decisions do not replicate."
- Every Study 9 HELD-OUT p-value enters the cross-study Holm. The appendix reports both adjusted p-values for each test.

### 3.3 tab:hyp row (P11)
```
H2a S (HO) & $P(\rho(1)>1)>1/2$, stratified & sign test & S9 HO & \HOSnineRawHTwoaStrat & \HOSnineHolmHTwoaStrat{} [\HOSnineXHolmHTwoaStrat] & \HOSnineHTwoaVerdict, median $\rho(1)=$ \HOSnineRhoOneStrat; magnitude rule: \HOSnineMagWord \\
```

### 3.4 Study 9 bullet, App prereg (P14)

Append:
> HELD-OUT was run once (Section~9 of the pre-registration), with all four FMs, the same code and the label normalisation of Amendment~2, at the joint opening of 2026-10-30.

## 4. App prereg Study 13 bullet (P15) and figure (P16)

P15: replace "HELD-OUT is pending." with:
> HELD-OUT was opened once on 2026-10-30, jointly with Studies~9 and~10, after all three pre-registrations were frozen; the opening note lists the cross-study Holm family of \HOXFamN{} tests (\texttt{\HOOpeningRev}). \HOSensIntNFlag{} tables were flagged by the integrity screen, \HOLabelNMerged{} had merged labels and \HOLabelNUnverified{} were unverified.

If any runner was resumed after a crash, add: "\HONResumed{} jobs were resumed from checkpoints with unchanged code." Add "\HONFailedTables{} tables failed for at least one FM" if that number is > 0.

`\HOOpeningRev` is the revisions.yaml id of the opening note. It is not created yet; it will be the next free id (R018 or later).

P16, the fig:heldout caption (same in every branch, since the figure shows the data):
> HELD-OUT replication on \HON{} tables (Study~13, pre-registered). Stratified minus random per FM and budget, in macro-F1 and in accuracy; layout as in Figure~\ref{fig:gap}. Filled marks are the four primary FMs (confirmatory, macro-F1); open marks are TabDPT v1.3, Mitra-v2 and LimiX-16M, pre-registered secondary learners whose outputs on DEV (and PMLB/UCI for LimiX) had been seen, so for them this is a replication on new data.

The figure is generated by `figures/make_figures.py` `fig_heldout_placeholder`, which is to become `fig_heldout`. Its file name stays `fig8_heldout_placeholder.pdf` unless the include path is changed with it.

## 5. Placeholder macros (none defined yet)

All macros are to be generated by the Study 13 / Study 9 analysis outputs into one `heldout_numbers.tex`. None is typed by hand.

- **Study 13, counts and integrity:**
  - `\HON`, `\HOXFamN`, `\HOOpeningRev`;
  - `\HOSensIntNFlag`, `\HOSensIntNDrop`, `\HOSensIntCOneVerdict`;
  - `\HOSensLabelNDrop`, `\HOSensLabelCOneVerdict`;
  - `\HOLabelNMerged`, `\HOLabelNUnverified`;
  - `\HONResumed`, `\HONFailedTables`.
- **Study 13 C1:**
  - `\HOINRep`, `\HOINRepX`, `\HOINRev`, `\HOINNotEval`, `\HOINAtten`;
  - `\HOIMinMean`, `\HOIMaxMean`;
  - per FM ∈ {FMone, FMtwo, FMthree, FMfive}: `\HOI<FM>{Raw,Holm,XHolm,Verdict,Mean}`.
- **Study 13 ranking:**
  - `\HOIIa{Raw,Holm,XHolm,Verdict,Mean}`;
  - `\HOIIc{Raw,Holm,XHolm,Verdict,Mean,Lo,Hi,WilcoxonP,Leader}`;
  - `\HORankingVerdict`.
- **S-10 / S-11:**
  - `\HOSTenFM{six,seven,eight}{Mean,Holm}`, `\HOSTenAccFM{six,seven,eight}Mean`;
  - `\HOSTen{HolmMax,HolmMin,MinMean,MaxMean,NExtend,ExtendList,NotList,NotMean,NotHolm,RevList,RevMean,RevHolm,NotEvalList,DisagreeList}`;
  - `\HOSEleven{MinMean,MaxMean,SignWord,SigList,SigMean,SigHolm}`.
- **S-1 to S-14:**
  - `\HOSOneNSig`, `\HOSTwoNSig`, `\HOSTwoNotList`;
  - `\HOSThree{Rho,P,BstarAgree}`;
  - `\HOSFour{MinMean,MaxMean,NSig,ArmDiffMin,ArmDiffMax}`;
  - `\HOSFive{Bin,Multi,MW,NMulti}`;
  - `\HOSSix{PIAboveList,PIStraddleList}`;
  - `\HOSSevenFM{five,two}{Mean,Holm}`;
  - `\HOSEight{SigList,Rank,NInfeasible}`;
  - `\HOSNine{F,FNoise,FDiff,FDiffLo,FDiffHi}`;
  - `\HOSTwelveNSig`, `\HOSThirteenNSig`, `\HOSFourteenNSig`.
- **Study 9:**
  - `\HOSnineN`;
  - `\HOSnine{RawHTwoaStrat,HolmHTwoaStrat,XHolmHTwoaStrat,HTwoaVerdict}`;
  - `\HOSnineRhoOneStrat{,Lo,Hi}`, `\HOSnineFMoneStratP`;
  - `\HOSnine{HolmHThreeb,HThreebTau,NOtherSupported,OtherSupportedList,MagWord}`.

Text-valued macros (verdicts, lists, `\HOIIcLeader`, `\HOSElevenSignWord`) must come from the analysis code's verdict strings. Do not write them by hand from the numbers.

## 6. Fill-in order on the day
1. Generate `heldout_numbers.tex` and `\input` it in the preamble.
2. Read the verdicts and pick, in this order:
   - the C1 branch (2.1);
   - the ranking branch (2.2);
   - the wording level (cross-study or study-wise, per claim);
   - the S-10 branch (2.9);
   - the Study 9 branches (3.1–3.2).
3. Apply P1–P16.
4. Build the paper and remove the placeholder boxes P2, P6 and P9.
