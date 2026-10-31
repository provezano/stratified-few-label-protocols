# Coding manual (verbatim copy of 00_PROTOCOL.md section 7, plus coder procedure)

## 7. Coding manual (variables per included paper)

Code the paper's **main reduced-label protocol**: the one behind its headline few-label / low-label results. If several protocols are used, code the main one and note the others in `notes`. For active learning, code how the **initial** labelled pool is drawn. If the whole labelled set is chosen by an algorithm with no random initial set, use C.

Categories (same definitions as the prior audit, §0):
- **A**: m labels per class, or a total of k balanced across classes. Assumes the labels.
- **A′**: another draw that uses the labels to guarantee coverage: proportional stratification, a constraint of ≥ 1 per class, a seed of 1 per class, or discarding seeds that miss a class.
- **B**: uniform draw of a total budget (no use of labels).
- **C**: selected budget (active or structured selection, e.g. by recency, clustering, uncertainty).
- **D**: a fraction of the training split, without saying whether it is stratified.
- **E**: not specified, or no experiment.

Variables (one row per paper in the coding CSV):

| column | values | rule |
|---|---|---|
| `paper_id` | as given | |
| `category` | A, A′ (write `A'`), B, C, D, E | main protocol |
| `category_secondary` | same or empty | other protocol used in the paper, if any |
| `coverage_forced` | yes / no / unclear | **yes** if the stated draw guarantees ≥ 1 labelled row per class by construction or by rejection (per-class draw with m ≥ 1, explicit ≥ 1 constraint, 1-per-class seed, seed rejection). **no** if a draw without any such guarantee is stated (uniform total, stated non-stratified fraction, selection with no class constraint). **unclear** if the draw is not described, or if labels are used but the guarantee is not stated (e.g. proportional stratification with tiny k and unstated rounding). |
| `few_shot` | yes / no | **yes** if the paper frames the setting as few-shot / k-shot / cold start, or any main reduced-label experiment uses ≤ 100 labelled rows in total or ≤ 10 per class. Otherwise no. |
| `missed_class_reported` | yes / no | yes only if the paper reports how often (or in how many runs/datasets) a class had no labelled row. A mention that it can happen, without a number, is `no` (note it). |
| `metrics` | semicolon list from: accuracy, AUC, macro-F1, F1-other (binary/weighted/micro/unspecified), balanced-accuracy, log-loss, other | metrics reported for the reduced-label classification results |
| `main_metric` | one of the above | the metric of the headline table/figure |
| `unseen_class_scoring` | free text or `not stated` | how classes absent from the labelled set are scored, if stated |
| `budget_unit` | per-class / total / fraction / mixed | unit in which the labelled budget is set |
| `smallest_budget` | free text | e.g. "1 per class", "4 total", "1%" |
| `n_seeds` | integer or `not stated` | repetitions of the label draw for the main protocol |
| `code_checked` | yes / no | yes only if the coder read the official code (read-only) to decide the draw |
| `quote` | ≤ 15 words, verbatim | the sentence that best shows the draw |
| `quote_location` | section / page / table | where the quote is |
| `notes` | free text | ambiguities, other protocols |

Decision aids:
- "randomly sample k examples" with nothing else → B (coverage `no`), unless the code (if checked) shows otherwise.
- "k-shot" defined as k per class → A. "k-shot" defined as k total and drawn class-balanced → A. "k total, stratified" (proportional) → A′.
- "We use 10% of the training data as labelled" with no draw described → D. "stratified 10%" → A′. "random 10% (not stratified)" stated explicitly → B.
- Learning curves over training sizes with the draw unstated → D if sizes are fractions, E if sizes are counts with nothing about the draw. Counts with "random subsample" → B.
- Train/test split stratification does **not** count; only the draw of the labelled set does.
- If the protocol is inherited by citation ("following STUNT"), code the inherited protocol and note it.

Quotations: only text actually read in the source; ≤ 15 words; with location. If the coder cannot find a sentence about the draw, write `none found` and code E (or D for fractions).


## Coder procedure

- Work only from the paper text files you are given (`downloads/text/<paper_id>.txt`, extracted from the PDF with `=== PAGE n ===` markers) and, if needed, the paper's arXiv/proceedings link. You may read the official code (read-only, e.g. raw GitHub files) only when the text leaves the draw ambiguous; then set `code_checked = yes`. Never execute downloaded code.
- Useful search terms inside a text file: `labeled`, `labelled`, `shot`, `per class`, `each class`, `budget`, `stratif`, `balanced`, `random`, `sample`, `seed`, `fraction`, `%`, `initial pool`, `missing class`, `macro`, `F1`, `AUC`, `accuracy`.
- Do NOT open: any other coder's files in `coding/`, `<prior audit file>`, or anything under `<project root>/`. Do not use web search to find other people's descriptions of the paper.
- Quotes must be verbatim from the text you read (≤ 15 words). If PDF extraction garbled a word, fix only obvious hyphenation/spacing.
- Write one CSV row per paper with exactly these columns, in this order:
  `paper_id,category,category_secondary,coverage_forced,few_shot,missed_class_reported,metrics,main_metric,unseen_class_scoring,budget_unit,smallest_budget,n_seeds,code_checked,quote,quote_location,notes`
- Use the exact value spellings of the table above (`A'` for A-prime; `yes`/`no`/`unclear`; metric names `accuracy`, `AUC`, `macro-F1`, `F1-other`, `balanced-accuracy`, `log-loss`, `other`).
