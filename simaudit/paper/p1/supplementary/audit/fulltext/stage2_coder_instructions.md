# Stage-2 full-text screening + coding instructions (supplementary search)

You screen and, if included, code tabular papers from their full text. Work only from the text files given (`downloads/text/<rec_id>.txt`, extracted from the PDF, with `=== PAGE n ===` markers). Do not use web search for other people's descriptions of the paper.

## A. Eligibility (stage 2)

Include if:
1. at least one experiment trains, adapts or conditions (in-context) a model on tabular data for **classification**; and
2. that experiment uses a **reduced labelled set**: a few-shot setting (<= ~100 labelled rows, or framed as few-shot / k-shot / cold start), or a labelled subsample of the training labels chosen to study label scarcity or small-data performance (semi-supervised fraction, active-learning initial pool, label budget, **learning curves over training/labelled-set size**, subsampling a dataset to a small n). For in-context-learning models the context set is the labelled training set; subsampling context/training rows to small n counts.

Exclude (reason, first that applies): `regression_only`; `not_tabular`; `survey_or_position`; `zero_shot_only` (no labelled rows at all); `no_reduced_label_experiment` (only full training sets; training size varied only for runtime/compute/latency scaling; naturally small datasets used whole and not framed as few-shot; subsampling only of large datasets down to e.g. 10k rows for compute reasons is NOT a reduced-label experiment unless the paper studies performance as a function of that size); `not_accessible` (text missing/garbled).

Search the text for: `learning curve`, `training size`, `sample size`, `subsample`, `few-shot`, `shot`, `small data`, `low-data`, `number of training`, `n_train`, `labeled`, `labelled`, `per class`, `stratif`, `budget`, `fraction`, `%`. Check the appendix too.

## B. Coding manual (verbatim rules of the main review)

Code the paper's **main reduced-label protocol**: here, the experiment that qualifies the paper under A (e.g. the small-n learning curve or few-shot table). Others go in `notes`. For active learning, code how the **initial** labelled pool is drawn. If the whole labelled set is chosen by an algorithm with no random initial set, use C.

Categories:
- **A**: m labels per class, or a total of k balanced across classes. Assumes the labels.
- **A'**: another draw that uses the labels to guarantee coverage: proportional stratification, a constraint of >= 1 per class, a seed of 1 per class, or discarding seeds that miss a class.
- **B**: uniform draw of a total budget (no use of labels).
- **C**: selected budget (active or structured selection, e.g. by recency, clustering, uncertainty).
- **D**: a fraction of the training split, without saying whether it is stratified.
- **E**: not specified, or no experiment.

| column | values | rule |
|---|---|---|
| `category` | A, A', B, C, D, E | main protocol |
| `category_secondary` | same or empty | other protocol used in the paper, if any |
| `coverage_forced` | yes / no / unclear | **yes** if the stated draw guarantees >= 1 labelled row per class by construction or by rejection (per-class draw with m >= 1, explicit >= 1 constraint, 1-per-class seed, seed rejection). **no** if a draw without any such guarantee is stated (uniform total, stated non-stratified fraction, selection with no class constraint). **unclear** if the draw is not described, or if labels are used but the guarantee is not stated (e.g. proportional stratification with tiny k and unstated rounding). |
| `few_shot` | yes / no | **yes** if the paper frames the setting as few-shot / k-shot / cold start, or any main reduced-label experiment uses <= 100 labelled rows in total or <= 10 per class. Otherwise no. |
| `missed_class_reported` | yes / no | yes only if the paper reports how often (or in how many runs/datasets) a class had no labelled row. A mention that it can happen, without a number, is `no` (note it). |
| `metrics` | semicolon list from: accuracy, AUC, macro-F1, F1-other, balanced-accuracy, log-loss, other | metrics reported for the reduced-label classification results |
| `main_metric` | one of the above | metric of the headline table/figure of the reduced-label experiment |
| `unseen_class_scoring` | free text or `not stated` | how classes absent from the labelled set are scored, if stated |
| `budget_unit` | per-class / total / fraction / mixed | |
| `smallest_budget` | free text | e.g. "1 per class", "4 total", "1%" |
| `n_seeds` | integer or `not stated` | repetitions of the label draw for the main protocol |
| `code_checked` | yes / no | yes only if you read the official code (read-only) to decide the draw |
| `quote` | <= 15 words, verbatim | the sentence that best shows the draw |
| `quote_location` | section / page / table | |
| `notes` | free text | ambiguities, other protocols |

Decision aids:
- "randomly sample k examples" with nothing else -> B (coverage `no`), unless the code (if checked) shows otherwise.
- "k-shot" defined as k per class -> A. "k-shot" defined as k total and drawn class-balanced -> A. "k total, stratified" (proportional) -> A'.
- "We use 10% of the training data as labelled" with no draw described -> D. "stratified 10%" -> A'. "random 10% (not stratified)" stated explicitly -> B.
- Learning curves over training sizes with the draw unstated -> D if sizes are fractions, E if sizes are counts with nothing about the draw. Counts with "random subsample" -> B.
- Train/test split stratification does **not** count; only the draw of the labelled set does.
- If the protocol is inherited by citation ("following STUNT"), code the inherited protocol and note it.
- Quotations: only text actually read; <= 15 words; with location (page from the `=== PAGE n ===` marker plus section/table). If no sentence about the draw is found, write `none found` and code E (or D for fractions).
- Code: you may read the official code (read-only, raw GitHub files via WebFetch) ONLY when the text leaves the draw ambiguous; then `code_checked = yes`. Never execute code.

## C. Output

Write two CSVs (proper quoting):
1. screening: `rec_id,decision,reason,evidence,evidence_location,frame,note` (decision include/exclude; reason empty for include; evidence = <= 15 words verbatim showing the reduced-label experiment or its absence; frame F1/F2).
2. coding (included papers only): `rec_id,category,category_secondary,coverage_forced,few_shot,missed_class_reported,metrics,main_metric,unseen_class_scoring,budget_unit,smallest_budget,n_seeds,code_checked,quote,quote_location,notes`
Use exact spellings (`A'`, `yes`/`no`/`unclear`, metric names as listed).
