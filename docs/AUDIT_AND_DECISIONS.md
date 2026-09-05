# Data Pipeline Audit + Open Decisions

Audit of Member 4's data pipeline against `docs/Implementation.md`, the base
paper (Xylouris et al., IEEE TCE 2025), Saurabh's paper, and Saurabh's supplied
notebooks. Written before M1 starts implementing.

> **Status: superseded in part.** The open questions in §3 have since been
> decided and moved to **`docs/PROJECT_DECISIONS.md`**, which is the file to
> follow. This one is kept as the record of what the audit found and fixed.
> §3 below is annotated with each outcome.

---

## 1. What was audited

| Item | Where it lives |
|---|---|
| NCSRD preprocessing | `ncsrd/prep.py` on `feature/data-pipeline` |
| NCSRD model adapter | `ncsrd/adapter.py` on `feature/data-pipeline` |
| Data4Cyber preprocessing | `data4cyber/prepare_data4cyber.py` on `feature/data-pipeline` |
| Reference implementation | `docs/Saurabh's Project Files/03…`, `04…`, `06…` |

**Verdict: M4's work was a sound base and is now finished.** The NCSRD
pipeline is a faithful, well-documented script form of Saurabh's notebooks 03
and 04. Thirteen issues were fixed across two passes -- one real leakage bug in
each dataset, two structural gaps that blocked M1 outright, and assorted
hygiene. Everything M4 got right was preserved.

---

## 2. Findings

### 2.1 Correct — keep as is

* `prep.py` matches notebook 03 step for step: same attack windows, same 14
  dropped identifier/address columns, same `cell_*_{ul,dl}_retx → *_retx_max`
  consolidation, same median imputation, same `*_apn` drop, same
  `StandardScaler`, same zero-variance pruning. 49 features + `attack_label`.
* `adapter.py` matches notebook 04's split: stratified 80/20, `random_state=42`,
  SMOTE on the **training split only**. Val and test keep the real 6.28% attack
  rate. Test set is never resampled.
* `save_split()` / `load_split()` persist the exact row indices — this is what
  makes the plan's "ONE split" rule (§17) enforceable across five people.
* `sequence_index.csv` preserves `_time` + `imeisv` outside the feature matrix.
  Necessary for temporal/group splits, LSTM windows, and every later
  window-based analysis. Good call.
* M4 documented the leakage risks honestly rather than hiding them.
* Optional dependencies (`xgboost`, `imblearn`, `tensorflow`) are imported
  lazily with actionable error messages.
* Data4Cyber fits imputation, variance filter and scaling on **train only**, and
  splits by scenario rather than by random row. Both are the right instinct.
* Data4Cyber is *not* forced into the NCSRD 49-feature schema — exactly what
  Implementation.md §4.2 asks for.

### 2.2 Fixed

| # | Problem | Severity | Fix |
|---|---|---|---|
| F1 | **Clock leakage (Data4Cyber).** `Substation-AC-Meter.realtime` survived into the 141-feature set. It is an absolute epoch clock, monotone within a scenario, and all 8 scenarios occupy **disjoint** ranges — so it is a perfect scenario id, and under a scenario-based split, a direct shortcut to the label. | **High** | `CLOCK_SUFFIXES = (".realtime", ".timestamp")` excluded in `feature_columns()`. 141 → 140 features. |
| F2 | **No row-level Data4Cyber output.** Only window tensors were written. The base paper's XGBoost is a *per-sample* classifier, so M1's Data4Cyber deliverable was impossible; and 232 training windows is far too little to train anything on. | **High** (blocks M1 Phase 2) | `<split>_rows.npz` now written alongside `<split>_windows.npz`, from the same split and the same transform, carrying `scenario` + `timestamp` so rows can be regrouped into windows at evaluation time. |
| F3 | **Scaling leakage (NCSRD).** `prep.py` fits `StandardScaler` on all 424,221 rows before any split. Faithful to notebook 03, but it means test statistics are baked into the training features, so *no* leakage-free run was possible. | Medium | `adapter.split(refit_scaler=True)` re-standardizes each split on training rows only. Because standardization is affine per column, applying it to the already-z-scored CSV is numerically identical to scaling the raw data on train only — verified to 1e-15. No need to re-run `prep.py`. Default stays `False` so notebook parity is untouched. |
| F4 | **No way to do window-based evaluation.** `DataBundle` returned `y_test` but no row ids, so test predictions could not be joined back to `_time`/`imeisv`. Every later phase (500-sample threshold windows, 1000-sample SHAP windows, per-window FPR variance) needs that join. | Medium | `DataBundle` now carries `train_index`, `val_index`, `test_index`, `test_time`, `test_ue`. |
| F5 | **Machine-specific paths.** Defaults were CWD-relative (`data/processed`) and `manifest.json` recorded `C:\Users\yashk\Downloads\…`. | Medium | `src/common/config.py` derives every path from the repo root. Manifest records only the folder name. |
| F6 | **Empty root `requirements.txt`, plus a second one in `ncsrd/`.** Implementation.md §3.3 says freeze *one* environment; as committed, nobody could set up from the repo. | Medium | One populated root `requirements.txt`. |
| F7 | **No `.gitignore`.** The 8 scenario folders of raw Data4Cyber (~400 MB of `.pcapng`) were committed into the repo. | Medium | `.gitignore` added covering `data/`, `results/`, captures, models, venvs. **This does not un-commit what is already on `feature/data-pipeline` — see A1 below.** |
| F8 | **Structure clutter.** 8 × `placeholder.txt`, two READMEs, three empty top-level docs, and code at `ncsrd/` + `data4cyber/` instead of the `src/` layout in Implementation.md §3.1/§16. | Low | Consolidated onto `main` in the planned layout; `.gitkeep` instead of `placeholder.txt`. |

### 2.3 Fixed in the second pass (after the D1-D9 decisions)

| # | Problem | Severity | Fix |
|---|---|---|---|
| F9 | **The pipeline could only produce Saurabh's 49 features.** The base paper's Table II is a different feature set -- verified: all **24** distinct per-cell metrics consolidated (M4 consolidated only `*_retx`) plus **14** UE/bearer columns, keeping `tac`, `cell_id` and the ~10 constant columns the 49-set drops. The two share only **6** column names, so M1 had no way to build a true base-paper configuration. | **High** (blocked D1) | `ncsrd_prep.py` now builds `base38` and `saurabh49` from identical rows, each verified column-for-column in `tests/test_pipeline.py`. |
| F10 | **Data4Cyber's only split held out S6 entirely**, making the primary experiment a novel-attack test rather than the independent within-dataset evaluation D3 asks for. Attack rate was 0.51 / 0.70 / 0.75 across splits, and `S1_industroyer_pv_alt` was silently unused. | **High** (blocked D3) | Added `block` mode: contiguous 120 s blocks stratified by scenario and label. Every family in every split, rates 0.61 / 0.58 / 0.55, no block or window crossing a boundary. The scenario holdout survives as a labelled secondary experiment and now includes S1_alt. |
| F11 | **No enforced split policy.** Any member could call `split()` with their own numbers and silently break the comparison. | Medium | `config.REFERENCE_SPLIT` / `config.PROJECT_STANDARD_SPLIT`, surfaced as `ad.reference_split()` / `ad.project_standard_split()`. |
| F12 | **No tests.** Nothing would have caught a broken feature set or a leaky split. | Medium | `tests/test_pipeline.py` -- 22 checks; the NCSRD half needs no real data. |
| F13 | Windows were grouped by scenario, so under a within-scenario split a window could span a train/test boundary. | Medium | `make_windows()` groups by whatever unit was assigned (block or scenario). |

### 2.4 Known limitations -- not fixed, by choice

* **`adapter._split_windows()` re-splits at the window level**, independently of
  the row-level split, so the LSTM's train/test membership is not the same
  partition as XGBoost's. Fine for a standalone LSTM, wrong for a strict
  four-model comparison. Deep models are optional (D8), so this is left alone.
* **`ran_ue_id` survives into `saurabh49`.** It is a session identifier, not a
  behaviour metric, and notebook 03 never adds it to the drop list. Reproduced
  faithfully rather than "fixed", because changing it would move Saurabh's
  reference numbers. It is absent from `base38`. Worth a sentence in the report.
* **Class-count mismatch.** `prep.py` and M4's README report 397,597 / 26,624;
  Saurabh's paper reports 397,576 / 26,645. Same total (424,221), ~21 rows
  differ — almost certainly an inclusive/exclusive window boundary. Harmless,
  but say which you used.
* **NCSRD median imputation is still global**, before the split. The same class
  of leak as F3, but far weaker -- a median over 424k rows barely moves. Fixing
  it would break notebook parity. Mention it in the report.

---

## 3. Open decisions — ALL NOW DECIDED

Kept for the reasoning. The binding versions live in
`docs/PROJECT_DECISIONS.md`.

### D1. What "reproduce the base paper" means — **DECIDED**

> **Outcome:** the exact 38 features were recovered from Table II and
> verified against the real raw schema, so the "not recoverable" note
> below is obsolete. Two separate configurations: `base_paper` on
> `base38`, `base_saurabh_static` on `saurabh49`. Both are in the pipeline.

**This is the most important one.** M4's pipeline reproduces **Saurabh's**
preprocessing, which is *not* the base paper's. Confirmed differences:

| | Base paper (Xylouris et al. §IV.D) | Saurabh / M4 pipeline |
|---|---|---|
| Features | **38** | **49** |
| Imbalance handling for XGBoost | **random undersampling**, custom strategy, + tuned `scale_pos_weight` | **SMOTE** |
| Split | train / **validation** / test, + cross-validation | 80/20 train/test, no validation |
| Stopping | **early stopping** on validation | fixed `n_estimators=300` |
| Categoricals | "label encoding of categorical variables" | `*_apn` columns dropped |
| Threshold | optimized (on validation) | Saurabh picks τ=0.7197 on the **test** set |
| Reported | acc 0.996, P 0.96, R 0.98, F1 0.97 | F1 0.9573 ("baseline static") |

Note the base paper used SMOTE and Focal Loss for **CNN/LSTM/MLP only** — not
for XGBoost. Copying `for_xgboost(balance="smote")` and calling it a base-paper
reproduction would be wrong.

Also: **the exact 38 features are not recoverable.** Table II in the PDF is a
raster image; `pdftotext` extracts nothing from it. We cannot know which 38.

**Recommendation.** Run *two* configurations from the one shared pipeline and
report both:

* **`base_paper`** — the paper's *model recipe* on the shared 49 features:
  three-way train/val/test, random undersampling + `scale_pos_weight`, logloss,
  early stopping on validation, threshold chosen on **validation**. Document
  "49 features instead of 38 (Table II is an image; the exact list is not
  published)" as an explicit, stated deviation.
* **`base_saurabh_static`** — SMOTE, 80/20, `n_estimators=300`, τ from the
  precision-recall curve. This is Saurabh's ablation row A0 (F1 0.9573) and is
  the anchor that makes the 3×1 comparison apples-to-apples.

Both are M1's, both go in `base_ncsrd_results.csv` as separate rows.

### D2. Which split is the headline number — **DECIDED**

> **Outcome:** `project_standard_split()` (temporal + train-only scaling)
> is the headline; `reference_split()` is the reproduction number. One
> frozen index file, shared by all three methods.

`random` (notebook-faithful, optimistic — 5-second samples of 7 UEs put
near-duplicates on both sides) or `temporal` (honest)?

**Recommendation:** `random` + `refit_scaler=False` is the **reproduction**
number, reported against the papers. `temporal` + `refit_scaler=True` is the
**headline** number, and all three methods are compared on it. Freeze it once
with `ad.save_split("data/ncsrd/processed/split_indices.npz")` and have everyone
`load_split()` that file. Do not let three people call `split()` themselves.

### D3. The Data4Cyber split — **DECIDED**

> **Outcome:** the stratified contiguous-block split is primary; the
> scenario holdout becomes a labelled secondary robustness experiment.
> Both are built by `data4cyber_prep.py`.

As committed, the default split holds out `S6_mqtt_supply_chain_compromise`
entirely — an attack family that never appears in training. That is
**zero-shot novel-attack detection**, a much stronger and different claim than
the "independent evaluation on a second dataset" that Implementation.md §20
settles on (Option A). Attack rate is also 50% train / 70% val / 75% test, and
`S1_industroyer_pv_alt` is silently unused.

**Recommendation:** make the primary 3×2 comparison a **stratified split within
scenarios** (or at minimum a scenario split where every attack family appears in
training, using `S1_industroyer_pv_alt` to balance), so it matches Option A.
Keep the current cross-scenario split as a clearly-labelled *secondary*
robustness result — it is a genuinely interesting extra, just not the headline.

### D4. Row-level vs window-level — **DECIDED**

> **Outcome:** row level is primary for all detection metrics; windows
> only for metrics that genuinely need a time axis.

The dataset is small: 12,562 rows total, 408 windows total. Per-window metrics
computed over 118 test windows will be very noisy.

**Recommendation:** report **row-level** metrics as primary for all three
methods on Data4Cyber (that is what `<split>_rows.npz` is for), and window-level
only for the stability metrics that require windows. State the sample sizes in
the results table so nobody over-reads a 118-window number.

### D5. Where the threshold is chosen — **DECIDED**

> **Outcome:** on validation, always, except inside the explicitly
> labelled Saurabh reproduction configuration.

Saurabh's notebook picks the F1-optimal τ on the **test** set, which is
optimistically biased. Our proposed method (FPR ≤ α) must pick τ somewhere too.

**Recommendation:** every method picks its threshold on **validation**, and
reports on test. Reproduce Saurabh's test-set choice only inside the
`base_saurabh_static` reproduction row, labelled as such.

### D6. Metric definitions — **DECIDED**

> **Outcome:** attack class (label 1) is primary; accuracy, weighted F1
> and ROC-AUC are secondary. Results row schema in PROJECT_DECISIONS.md D6.

Base paper quotes weighted F1 *and* attack-class F1; Saurabh quotes attack-class
F1. These differ a lot on a 6.28%-positive problem.

**Recommendation:** the **attack class (label = 1)** is the reported class for
precision / recall / F1 / FPR everywhere; report accuracy and weighted F1 as
secondary columns. `FPR = FP / (FP + TN)`. M5 should own one `evaluate.py` and
nobody else should compute metrics by hand.

---

## 4. Actions

| | Action | Status |
|---|---|---|
| **A1** | ~400 MB of raw Data4Cyber (including `.pcapng`) is committed on `feature/data-pipeline`. `.gitignore` stops recurrence but not history. | **OPEN, needs a human.** Requires a history rewrite plus a force-push, which is out of scope here. Do not merge that branch as is. See section 5. |
| **A2** | Move the pipeline code into `src/common/data/` and drop the `placeholder.txt` files. | **DONE** on `main` (uncommitted). M4 should adopt this layout rather than re-merging the old one. |
| **A3** | Publish the dataset documentation required by Implementation.md 4.3. | **DONE** -- "Dataset reference" in `src/common/data/README.md`, covering both datasets. |
| **A4** | `run_experiment.py` is empty. | **Left to M5**, correctly. The file now documents the intended CLI, and the results row schema is fixed in PROJECT_DECISIONS.md D6 so M1's output will merge cleanly when the evaluator lands. |

## 5. The one thing that needs manual Git work

`feature/data-pipeline` carries the raw dataset in its history:

```
data4cyber/**/*.pcapng           ~400 MB
data4cyber/**/dataset*.csv        ~38 MB
data4cyber/prepared_data/*.npz
```

Deleting those files in a new commit does **not** shrink the repository -- the
blobs stay reachable from `c6c3fa6`. Cleaning up means rewriting that branch's
history and force-pushing, which every member then has to re-clone around.
Options, best first:

1. **Rewrite before merging** -- `git filter-repo --path data4cyber/ --invert-paths`
   on the branch, force-push, everyone re-clones. Cheapest now, while only one
   branch is affected and nobody has built on it.
2. **Abandon the branch** -- `main` already carries the corrected pipeline code,
   so deleting the branch achieves the same thing with no rewrite.
3. **Accept the size** -- a ~450 MB clone is survivable for a student project,
   but annoying, and it grows if anyone re-adds data.

Agree the choice with the group *before* anyone pushes, and keep `.gitignore` in
place so it cannot happen twice.
