# Project Decisions

The rules all five members follow. If your code disagrees with this file, your
code is wrong. If this file is wrong, change it here first and tell the group —
do not work around it locally.

The only other documents you need:
* `README.md` — setup, dataset placement, how to run things.
* `docs/Implementation.md` — roadmap and division of work. A plan, not a spec;
  where it disagrees with this file on method, this file wins.
* the reference papers in `docs/`.

---

## 1. The research progression

```
BASE PAPER                Xylouris et al., IEEE TCE 2025
  XGBoost, static threshold
        |  limitations: independent features, static threshold, passive SHAP
        v
SAURABH'S EXTENSION       correlation behavioural graph, per-window F1-optimal
  graph_frob_div          threshold, SHAP drift via Kendall tau
        |  limitations: FPR spikes, static correlation baseline, slow SHAP
        v
OUR PROPOSED WORK         EWMA + CUSUM baseline adaptation, FPR-constrained
  (faculty extension)     threshold, lightweight SHAP, distillation for edge
```

Each stage must be reproducible **on its own**. Never let a later stage's ideas
leak backwards into an earlier one.

## 2. The 3 × 2 matrix

| Method | NCSRD-DS-5GDDoS | Data4Cyber |
|---|---|---|
| **Base paper** (M1) | reproduce | adapt |
| **Saurabh** (M2) | reproduce | adapt |
| **Proposed** (M3) | implement | adapt |

Plus an ablation of the proposed method (M5 + M3).

---

## D1 — Base paper reproduction

### The exact 38 features

Table II of the base paper. Verified against the real raw schema: the first 24
are exactly the distinct per-cell metrics (with none left over) and the last 14
are UE/bearer-level columns — 24 + 14 = 38. The paper states that "the relevant
cell-specific features were consolidated into unified columns based on the UE's
associated cell", which is why `cell_1_cqi` / `cell_3_cqi` become one `cqi`.

```
consolidated per-cell (24)
  ul_retx  dl_retx  ul_tx  dl_tx  ul_bitrate  dl_bitrate  ul_mcs  dl_mcs
  ul_path_loss  cell_id  epre  turbo_decoder_avg  initial_ta  ul_err
  ul_n_layer  ul_phr  ul_rank  dl_err  cqi  p_ue  pusch_snr  ri
  turbo_decoder_max  turbo_decoder_min

UE / bearer level (14)
  tac  bearer_0_session_id  bearer_1_session_id
  bearer_0_dl_total_bytes  bearer_0_ul_total_bytes
  bearer_0_qos_flow_id  bearer_0_sst
  bearer_1_dl_total_bytes  bearer_1_ul_total_bytes
  bearer_1_qos_flow_id  bearer_1_sst
  t3512  ue_aggregate_max_bitrate_dl  ue_aggregate_max_bitrate_ul
```

Encoded in `src/common/data/ncsrd_prep.BASE38_FEATURES`. Two naming notes:

* Table II writes `bearer_{0,1}_session_id`; the raw column is
  `bearer_{0,1}_pdu_session_id` (`TABLE2_ALIASES`).
* **Residual ambiguity.** `ul_bitrate` and `dl_bitrate` exist *both* as
  top-level UE columns and as per-cell metrics. We use the consolidated
  per-cell versions, because that is the only reading under which all 24 cell
  metrics map onto Table II with none left over. Flip
  `BASE38_BITRATE_SOURCE = "ue"` to test the alternative; the count is 38 either
  way. This is the one place where the paper is genuinely under-specified.

### Two configurations that must never be mixed

| | **True base-paper config** | **Saurabh-style static baseline** |
|---|---|---|
| Feature set | `base38` (38) | `saurabh49` (49) |
| Cell handling | all 24 metrics consolidated | only `*_retx` consolidated |
| Constant columns | **kept** (the paper's 38 include ~10) | dropped (13 of them) |
| `tac`, `cell_id` | kept | dropped as identifiers |
| `ran_ue_id` | not present | **present** (notebook 03 never drops it) |
| Imbalance | random undersampling + `scale_pos_weight` | SMOTE on train only |
| Purpose | reproduce the base paper | anchor Saurabh's ablation row A0 |

They share only **6** column names. They are different lineages, not variants.
Load them by name:

```python
from ncsrd_adapter import NetworkDataAdapter
ad = NetworkDataAdapter.for_feature_set("base38")      # or "saurabh49"
```

### Base-paper XGBoost recipe (from §IV.D)

Follow as closely as the paper allows: median imputation, label encoding of
categoricals (in practice every selected column is already numeric),
`StandardScaler`; **custom random undersampling** that shrinks the majority
class while keeping all minority samples; tuned `scale_pos_weight`; `logloss`
as the eval metric; early stopping monitored on a validation split;
cross-validation for model selection. Reported metrics: accuracy, precision,
recall, F1, PR curve, ROC-AUC.

Note the paper used **SMOTE and Focal Loss for CNN/LSTM/MLP only** — not for
XGBoost. Do not apply SMOTE in the true base-paper configuration.

Reference targets (attack class): accuracy 0.996, precision 0.96, recall 0.98,
F1 0.97, weighted F1 1.00.

**There is no public Xylouris implementation.** The base system is
reconstructed from the paper text, Table II, the supplied notebooks and the
public dataset. Say so in the report, and document every deviation.

---

## D2 — NCSRD evaluation: two clearly separate purposes

### A. Reference reproduction
Compare against published numbers as faithfully as possible.
`config.REFERENCE_SPLIT` — random stratified 80/20, seed 42, scaler as
`ncsrd_prep.py` fit it (globally).

```python
ad.reference_split()
```

Optimistic by construction: rows are 5-second samples of the same 7 UEs, so a
random split puts near-duplicates on both sides, and the scaler saw the test
rows. **Never quote these as the headline.**

### B. Project standard — the headline comparison
`config.PROJECT_STANDARD_SPLIT` — temporal 70/10/20, train-only scaling.

```python
ad.project_standard_split()
```

Chronological, so no future information leaks backwards, and all preprocessing
statistics come from training rows only.

### One frozen split for all three methods

Build once, commit the index file, everyone loads it:

```python
# once
ad = NetworkDataAdapter.for_feature_set("saurabh49").project_standard_split()
ad.save_split(config.SPLIT_INDEX_FILE)

# everyone else, every method
ad = NetworkDataAdapter.for_feature_set(...).load_split(config.SPLIT_INDEX_FILE)
```

`base38` and `saurabh49` are built from identical rows in identical order, so
one index file applies to both. Nobody calls `split()` with their own numbers.

---

## D3 — Data4Cyber splits

### Primary: `block` (the 3 × 2 comparison)
Each scenario's timeline is cut into contiguous 120-second blocks; whole blocks
go to train/validation/test, stratified by scenario and block label.

* every attack family appears in every split
* adjacent near-identical seconds never straddle a boundary
* no window is ever built across a boundary
* attack rate ≈ 0.61 / 0.58 / 0.55 — comparable across splits

Output: `data/data4cyber/processed/block/`

### Secondary: `scenario` (robustness only)
Whole scenarios held out, so the test set contains attack families never seen
in training (S6, MQTT supply-chain). This is a **novel-attack robustness**
experiment. Report it separately, clearly labelled. Do not put it in the main
3 × 2 table.

Output: `data/data4cyber/processed/scenario/`

### Always excluded
`*.realtime` and `Profile.timestamp` are absolute wall-clocks. All 8 scenarios
occupy disjoint clock ranges, so those columns are a perfect scenario id and a
direct shortcut to the label. Enforced by `CLOCK_SUFFIXES`, checked by
`tests/test_pipeline.py`.

---

## D4 — Metric level

**Row / sample level is primary** for: precision, recall, F1, accuracy, FPR,
ROC-AUC. Use `<split>_rows.npz` on Data4Cyber and the `DataBundle` on NCSRD.

**Windows only for metrics that genuinely need a time axis:** FPR variance,
maximum FPR, threshold variance, drift and adaptation behaviour, detection
latency, drift latency, drift-event counts.

Always state the sample size. Data4Cyber has ~3.5k test rows but only ~85 test
windows — a window-level number there is noisy and must not be over-read.

---

## D5 — Threshold selection

Project-standard experiments: **train on train, choose the threshold on
validation, report on test.** Never select a threshold using test labels.

The one exception is reproducing Saurabh's published static threshold
(τ ≈ 0.7197), which he picks on test. Keep that behaviour *only* inside the
reproduction configuration and label it `reference_reproduction` in the results.

---

## D6 — Metric definitions

Primary metrics refer to the **attack class (label = 1)**:

```
Precision(1) = TP / (TP + FP)
Recall(1)    = TP / (TP + FN)
F1(1)        = 2PR / (P + R)
FPR          = FP / (FP + TN)
```

Secondary: accuracy, weighted F1, ROC-AUC.

These differ a lot on a 6.28 %-positive problem — the base paper quotes both
attack-class F1 (0.97) and weighted F1 (1.00). Always say which you mean.

One shared evaluator (M5) will own these definitions. Until it exists, emit the
row schema below so results merge cleanly later.

### Results row schema

Every run appends one row to `results/raw/<method>_<dataset>_results.csv`:

```
dataset, method, config, feature_set, split_policy, seed,
precision, recall, f1, fpr, accuracy, weighted_f1, roc_auc,
tp, fp, fn, tn, threshold, threshold_selected_on,
n_train, n_test, inference_latency_ms, model_size_kb, notes
```

`config` distinguishes e.g. `base_paper` / `base_saurabh_static`;
`split_policy` is `reference` or `project_standard`.

---

## D7 — What Data4Cyber does and does not show

Data4Cyber is a **smart-grid / ICS testbed** (Modbus and MQTT power-meter
telemetry, Industroyer / ARP-spoofing / supply-chain attacks). It is a
different cyber-physical domain, not more 5G data.

* **Correct claim:** the detection *methodology* remains useful on a second
  cyber-physical dataset after dataset-specific adaptation.
* **Wrong claim:** "our 5G detector generalizes to 5G", or that anything
  transfers unchanged.

The two datasets keep separate feature schemas and separate preprocessing
adapters by design. Only the method transfers. Document every adaptation.

---

## D8 — CNN / LSTM / MLP

**Optional. Not now.** The core project is the XGBoost-based
Base → Saurabh → Proposed progression. The adapter already emits the right
tensor shapes if anyone has spare time at the end.

---

## D9 — Repository and data policy

Datasets are **not** committed. `data/` is git-ignored except for `.gitkeep`
markers. Download links and unpack locations are in the root `README.md`;
`src/common/config.py` resolves every path from the repository root, so no code
ever contains `C:/Users/<name>/...`.

The raw Data4Cyber captures that were once committed have been removed from
history. Keep it that way: put datasets under `data/`, never anywhere else in
the tree, and never commit them. Only each Data4Cyber scenario's `dataset.csv`
is used by any pipeline — the `.pcapng` captures are not needed at all.

---

## Datasets

Neither dataset is committed. Download links and unpack locations are in
`README.md`. Facts everyone should share:

### NCSRD-DS-5GDDoS — https://zenodo.org/records/13900057

| | |
|---|---|
| File used | `amari_ue_data_classic_tabular.csv`, an Amarisoft 5G testbed capture |
| Raw size | 424,660 rows × 80 columns; 424,221 usable after timestamp cleaning |
| Sampling | one row per UE per ~5 s |
| UEs | 7 distinct `imeisv` values |
| Cells | `cell_1_*` and `cell_3_*`; a UE attaches to one at a time, so ~57 % of per-cell columns are NaN |
| Label | `attack_label`, derived by time-window matching — the raw data has no label column |
| Class balance | ~397,597 benign / ~26,624 attack — **6.28 % positive** |
| Attack classes | SYN flood, ICMP flood, UDP fragmentation, DNS flood, GTP-U flood |
| Missing values | ~9.8 M NaNs, median-imputed; `bearer_1_apn` is 18 % missing and dropped |
| Categoricals | `*_apn`, `*_ip`, `*_ipv6` only — every modelled feature is numeric |
| Windows | 500 samples (threshold / correlation), 1000 samples (SHAP drift) |

Attack windows come from the dataset documentation and are treated as ground
truth, both endpoints inclusive. Saurabh's paper reports 397,576 / 26,645 —
about 21 rows differ from ours, almost certainly a boundary convention.
Harmless, but state which you used.

### Data4Cyber — https://zenodo.org/records/19965384

| | |
|---|---|
| Files used | `dataset.csv` in each of 8 scenario folders (`S0_…`–`S6_…`) |
| Raw size | 14,354 rows × 156 columns (~1,800 rows per scenario) |
| Sampling | one row per second |
| Label | `attack_active` (`True`/`False`); `attack_phase` gives finer classes |
| Class balance | ~55 % attack overall — far more balanced than NCSRD |
| Attack classes | Industroyer (PV, BSS), ARP spoofing / MitM, MQTT supply-chain compromise |
| Features | 145 numeric candidates → 140 after the train-fit variance filter |
| Excluded | `Attacker.*` (trivial giveaway), `*.realtime` and `Profile.timestamp` (clock leakage, D3) |
| Windows | 60 s, 30 s stride, built inside one block/scenario |

Per-scenario attack rate: S0 0.00, S1 0.51, S1_alt 0.51, S2 0.51, S3 1.00,
S4 0.70, S5 0.74, S6 0.75. Attack is strongly time-localized (benign first,
attack later) — which is why a naive chronological split would be degenerate
and the primary split assigns stratified contiguous blocks instead.

---

## Reference targets

Numbers to reproduce, **not** to hard-code. Attack class unless stated.

| Source | Configuration | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| Base paper | XGBoost, 38 features | 0.97 | 0.96 | 0.98 |
| Saurabh | Baseline static | 0.9573 | 0.9448 | 0.9701 |
| Saurabh | + Graph | 0.9653 | 0.9599 | 0.9707 |
| Saurabh | + SHAP drift | 0.9625 | 0.9520 | 0.9731 |
| Saurabh | All three | 0.9648 | 0.9587 | 0.9711 |
| Saurabh | Baseline adaptive | 0.9730 | 0.9615 | 0.9848 |

The base paper also reports accuracy 0.996 and weighted F1 1.00. Saurabh
reports the graph cutting false positives from 302 to 216, and adaptive
thresholding producing FPR outliers up to 0.34 in borderline windows — the
latter is the specific weakness our proposed work targets.

Expect the project-standard split (D2) to score **below** these; that gap is a
finding, not a failure.

---

## Known limitations to state in the report

Deliberately not "fixed", because fixing them would break reproduction:

* **NCSRD median imputation is global**, computed before the split. A mild leak;
  a median over 424k rows barely moves. Faithful to notebook 03.
* **`ran_ue_id` is in `saurabh49`.** It is a session identifier, not a behaviour
  metric, but notebook 03 never drops it. Reproduced as-is; absent from
  `base38`.
* **Reference-split scaling leakage.** `ncsrd_prep.py` fits the scaler on the
  whole file. `refit_scaler=True` (the project-standard default) removes it.
* **LSTM windows re-split independently** of the row-level split, so a deep
  model's partition is not identical to XGBoost's. Only matters if anyone does
  the optional D8 work.
* **Data4Cyber is small**: ~3.5k test rows but only ~85 test windows. Never
  over-read a window-level number there.

---

## Conventions

* **Branches.** `main` is the shared foundation: decisions, config, data
  pipelines, tests, docs. Four working branches, one per member who still has
  code to write — `feature/base` (M1 → `src/base/`), `feature/saurabh`
  (M2 → `src/saurabh/`), `feature/proposed` (M3 → `src/proposed/`),
  `feature/evaluation` (M5 → `run_experiment.py`, evaluator, results).
  Merge to `main` by PR. M4's pipeline work is already on `main`, so
  `feature/data-pipeline` and `develop` are no longer needed.
* **Seed.** `config.SEED = 42` everywhere. Record it in every results row.
* **Paths.** Import from `src/common/config.py`. Never hard-code.
* **Model interface.** Every method exposes `fit()`, `predict_proba()`,
  `predict()` so the runner can treat them interchangeably.
* **Changing shared code** (`src/common/`, split policy, metric definitions)
  requires telling the group first. Everything downstream depends on it.
* **After touching the pipeline**, run `python tests/test_pipeline.py`.
