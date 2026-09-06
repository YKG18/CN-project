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

> **CHANGED — everyone must re-freeze.** The project-standard split used to be
> a plain chronological 70/10/20 slice. That is now **stratified contiguous
> blocks**, because the old policy produced a validation band with **zero
> attack rows** (NCSRD's five attack windows are short and far apart), which
> made threshold selection impossible. If you froze a split before this change,
> delete it and rerun `python src/common/data/freeze_split.py`.

### A. Reference reproduction

Compare against published numbers as faithfully as possible.

| Config | Use |
|---|---|
| `config.REFERENCE_SPLIT` | random stratified 80/20, seed 42, no validation, scaler as `ncsrd_prep.py` fit it — Saurabh's notebook exactly |
| `config.BASE_REFERENCE_SPLIT` | the same, plus 10 % validation, for methods that need one (early stopping, threshold selection) |

```python
ad.reference_split()                       # == config.REFERENCE_SPLIT
ad.split(**config.BASE_REFERENCE_SPLIT)    # when you need a validation set
```

Optimistic by construction: rows are 5-second samples of the same 7 UEs, so a
random split puts near-duplicates on both sides, and the scaler saw the test
rows. **Never quote these as the headline.**

### B. Project standard — the headline comparison

`config.PROJECT_STANDARD_SPLIT` — **stratified contiguous 20-minute blocks**,
70/10/20, train-only scaling.

```python
ad.project_standard_split()                # or, preferably, load the frozen file
```

How it works: the capture is cut into `config.BLOCK_MINUTES` blocks; each block
is put in a stratum by its attack content (all benign / mixed / all attack);
whole blocks are then dealt to train, validation and test. Rows never move
individually, so near-duplicate neighbours stay on the same side, and every
split gets both classes.

Measured on the real data (seed 42):

| split | rows | attack | rate | blocks |
|---|---:|---:|---:|---:|
| train | 296,846 | 18,713 | 6.30 % | 229 |
| validation | 41,313 | 2,551 | 6.17 % | 33 |
| test | 86,062 | 5,360 | 6.23 % | 66 |

against a 6.28 % population rate. Train contains all five attack types.

**Why 20 minutes.** A block must exceed the longest evaluation window any
method uses (`SHAP_WINDOW` = 1000 samples). At 20 minutes the median block
holds ~1,400 rows, so both the 500-sample threshold window and the
1000-sample SHAP window fit inside one block. At 10 minutes blocks hold ~700
rows and the SHAP window does not fit; at 30 minutes too few attack blocks
remain to spread across three splits. Changing `BLOCK_MINUTES` means
re-checking both properties.

### One frozen split for all three methods

```bash
python src/common/data/freeze_split.py          # once, writes config.SPLIT_INDEX_FILE
python src/common/data/freeze_split.py --check  # verify yours matches
```

```python
ad = NetworkDataAdapter.for_feature_set("saurabh49").load_split(
         config.SPLIT_INDEX_FILE)
```

`base38` and `saurabh49` are built from identical rows in identical order, so
one index file applies to both — a test asserts this. The file lives under the
git-ignored `data/`; it is not committed, but it is fully determined by
`PROJECT_STANDARD_SPLIT` and `SEED`, so every machine regenerates the same
indices. **Nobody calls `split()` with their own numbers.**

### Build every window INSIDE one block

This is the part that will silently corrupt results if ignored. Test rows are
now a set of blocks scattered through the capture, not one continuous stream.
A 500- or 1000-sample window that runs off the end of a block splices together
moments hours apart.

`DataBundle` gives you what you need:

```python
b = ad.for_xgboost(balance="none")
# b.test_block  block id per test row   (rows come back ordered by block, time)
# b.test_time   int64 NANOseconds since epoch, UTC
# b.test_index  original row ids

for blk in np.unique(b.test_block):
    rows = np.flatnonzero(b.test_block == blk)      # already chronological
    for start in range(0, len(rows) - W + 1, W):    # W = 500 or 1000
        window = rows[start:start + W]
```

Skip blocks shorter than your window, exactly as `data4cyber_prep.make_windows`
does. This applies to Saurabh's Module B (500-sample adaptive threshold) and
Module C (1000-sample SHAP drift), and to the proposed EWMA/CUSUM baseline
updates.

### Fit every baseline on training rows only

Under the project-standard policy, anything estimated from data is estimated
from **training rows**, not the whole file:

* the benign **correlation baseline** for Saurabh's Module A and our EWMA
  baseline — use training benign rows, not all benign rows;
* the **SHAP reference ranking** that drift is measured against;
* scaling — already handled by `refit_scaler=True`.

Saurabh's notebook computes the correlation baseline over the entire dataset.
That is fine inside the `reference` reproduction; it is leakage inside
`project_standard`.

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

Both datasets now use the same idea — whole contiguous blocks, stratified,
never split mid-block — so the windowing rule is identical on either side:
**build windows inside one block** (D2). `data4cyber_prep.make_windows` already
enforces it; on NCSRD use `bundle.test_block`.

### The attack class is the MAJORITY here

NCSRD is 6.28 % attack; Data4Cyber train is ~61 % attack. Any imbalance
handling written for NCSRD points the wrong way unless it decides the majority
from the data. The base paper's rule -- "reduce the majority class while
retaining all minority class samples" -- therefore thins *attacks* on
Data4Cyber. `base_xgboost.undersample_majority()` already does this; SMOTE-style
oversampling of "the minority" likewise means oversampling **benign** here.
Check the direction before reusing NCSRD code.

Ratios above the natural class ratio (~1.56) are no-ops, so a grid built for
NCSRD's 14.9:1 imbalance mostly repeats the same fit.

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

Under the project-standard split those windows must be built **inside one
block** — see D2. A window that crosses a block boundary joins moments hours
apart and any stability or latency number computed from it is meaningless.
Window counts are therefore smaller than the row counts suggest: the NCSRD test
split has 66 blocks of ~1,400 rows, so roughly 160 windows of 500 or 80 of
1000. Report the window count alongside any window-level metric.

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

## Non-stationary traffic (faculty direction 5)

`experiments/ncsrd/run_nonstationary.py` generates synthetic **benign** traffic
variations and measures the false-alarm rate over 1,003 sliding windows of 500
samples. Every row is benign, so FPR is the false-alarm rate directly.

Measured on the NCSRD project-standard split (static / frozen baseline):

| profile | FPR mean | FPR variance | FPR max |
|---|---:|---:|---:|
| stable (control) | 0.0010 | 0.000098 | 0.132 |
| gradual drift | 0.0036 | 0.000262 | 0.132 |
| bursty mMTC | 0.0135 | 0.001880 | 0.370 |
| periodic URLLC | 0.0467 | 0.021684 | 0.800 |

This **confirms the premise of the faculty brief**: a static benign correlation
baseline degrades badly once normal traffic changes shape — FPR variance rises
221x and worst-case FPR reaches 0.80 under periodic URLLC. It also shows our
EWMA implementation does not currently rescue it (see the known limitation
below); the streaming baseline is worse on every profile. Detection latency is
0.0012–0.011 ms/sample throughout, comfortably inside the brief's sub-100 ms
target.

---

## Known limitations to state in the report

* **The proposed online correlation baseline does not influence classification,
  and cannot with this architecture.** The faculty brief asks for the static
  benign correlation matrix to be *replaced* by an adapting EWMA baseline. EWMA
  and CUSUM are implemented and do run (47 updates / 24 change points on NCSRD),
  but `predict_proba` scores against `fitted_ewma_`, frozen at the end of
  training, so the adapted state never reaches the classifier.

  Two candidate fixes were implemented and measured, and **both are far worse**:

  | configuration | NCSRD F1 | FPR |
  |---|---:|---:|
  | train static / serve frozen *(current default)* | **0.9792** | 0.0009 |
  | train static / serve streaming | 0.2002 | 0.4608 |
  | train adaptive / serve frozen | 0.0827 | 0.0003 |
  | train adaptive / serve streaming | 0.0016 | 0.2119 |

  The cause is structural, not a bug and not leakage. The Frobenius divergence
  is only informative *against a fixed reference*; once the baseline chases the
  data, divergence collapses toward zero for benign and attack windows alike and
  the 50th feature stops carrying signal. Freezing the baseline is what makes
  the feature work.

  So the requirement is met in mechanism (EWMA/CUSUM exist, run, and are gated
  on predictions rather than labels) but **not in effect** — the adaptation does
  not change detections. Making it effective needs a different architecture, for
  example using the adapting baseline for a standalone drift alarm rather than
  as a classifier input. That is future work, not a fix.
  `predict_proba_streaming()` and `_build_adaptive_training_features()` are kept
  so both experiments stay reproducible. Report this honestly.
* **On the project-standard split, Proposed (P6) and Saurabh produce identical
  test predictions.** Verified directly: their `predict_proba` outputs match to
  0.0, and Saurabh's global threshold (0.5700) and the proposed constrained
  threshold (0.57) coincide, so the confusion matrices are identical. P6 keeps a
  Saurabh backbone and its EWMA / CUSUM / fast-SHAP modules change adaptation and
  latency, not the probability ranking on this data. Report the two rows as they
  are and argue the proposed contribution on the operational axes (FPR
  stability, drift adaptation, SHAP latency, distilled model size) rather than on
  F1 — do not present them as separate detection results.
* **Saurabh's per-window threshold is nearly static here**: only 2 distinct
  window thresholds across the NCSRD validation windows, so Module B behaves
  close to a global cut on this split. His paper reports thresholds spanning
  [0.04, 0.95] across 848 windows, because he re-optimises F1 on each window's
  own labels; we fit thresholds on validation and apply them cyclically to test
  windows so no test label is used (D5). The leakage-safe choice costs most of
  the adaptivity.
* **Our Saurabh configuration is not a literal reproduction.** His paper
  specifies `n_estimators=300` on SMOTE-balanced data; we use `n_estimators=1000`
  with early stopping and `scale_pos_weight`, on the project-standard split.
  Deliberate, but it means the published 0.9573 / 0.9730 are not directly
  comparable to our numbers.

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
* **`Profile.*` are simulator driving inputs**, not measured telemetry
  (irradiance, load and PV setpoints), and they track time of day — the
  strongest single-feature AUC on Data4Cyber is `Profile.total_irradiance` at
  0.76. They are kept because they are part of the released feature set and a
  real EMS knows them, but a sensitivity run without them is worth quoting:
  Base F1 moves 0.8949 → 0.8986, so the result does not depend on them
  (`experiments/data4cyber/run_base.py --no-profile`).
* **The NCSRD block split still shares attack episodes across splits.** With
  only five attack windows, blocks from the same episode land in different
  splits, so train and test can contain different minutes of the *same* attack.
  It is therefore an honest generalization estimate, **not** an unseen-attack
  test. The Data4Cyber `scenario` split (D3) is where unseen-attack robustness
  is measured.
* **The NCSRD test split does not contain every attack type.** Train sees all
  five; test sees a subset. That is a consequence of stratifying ~19 attack
  blocks across three splits, and it is why the attack *rate* rather than the
  attack *mix* is matched across splits.
* **`_time` is int64 nanoseconds since epoch.** pandas ≥ 2 parses these
  timestamps to `datetime64[us]`, so a bare `.astype("int64")` yields
  microseconds and silently turns 2024 into 1970. The adapter forces
  nanoseconds; a test guards it. Decode with
  `pd.to_datetime(t, unit="ns", utc=True)`.

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
* **Before any project-standard run**, `python src/common/data/freeze_split.py
  --check` — it tells you whether your frozen indices match the current policy.
* **Never call `split()` yourself for a project-standard run.** Load
  `config.SPLIT_INDEX_FILE`. The runners do this automatically.
