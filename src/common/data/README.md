# Shared data pipelines

Three files, one job: turn the two raw datasets into ML-ready arrays that every
method consumes identically.

| File | What it does |
|---|---|
| `ncsrd_prep.py` | Raw NCSRD CSV → two labelled, scaled feature sets + a row-aligned sequence index. Run once. |
| `ncsrd_adapter.py` | `NetworkDataAdapter` — splits and reshapes the processed NCSRD data per model family. Import this. |
| `data4cyber_prep.py` | Raw Data4Cyber scenarios → row-level and window-level arrays under two split modes. Run once. |

Policy (feature sets, splits, thresholds, metrics) lives in
[`docs/PROJECT_DECISIONS.md`](../../../docs/PROJECT_DECISIONS.md). This file is
the mechanics.

After changing anything here, run `python tests/test_pipeline.py`.

---

## 1. NCSRD

```bash
python src/common/data/ncsrd_prep.py          # builds BOTH feature sets
```

```
amari_ue_data_classic_tabular.csv   (424,660 x 80, raw)
            |
            |  ncsrd_prep.py
            v
   +------------------------+------------------------+
   |  ncsrd_base38.csv      |  ncsrd_saurabh49.csv   |
   |  38 features           |  49 features           |
   |  base paper Table II   |  Saurabh notebook 03   |
   +------------------------+------------------------+
            |  identical rows, identical order, one shared sequence_index.csv
            v
        ncsrd_adapter.py
```

### The two feature sets

They are **different lineages, not variants** — they share only 6 column names.
Mixing them invalidates the Base → Saurabh comparison.

| | `base38` | `saurabh49` |
|---|---|---|
| Cell metrics | all 24 consolidated to `cqi`, `dl_tx`, … | only `*_retx` → `ul_retx_max`, `dl_retx_max` |
| Constant columns | kept (~10; the paper's 38 include them) | dropped (13) |
| `tac`, `cell_id` | kept | dropped as identifiers |
| `ran_ue_id` | absent | **present** (notebook 03 never drops it) |
| `*_apn` | absent | dropped |

`base38` is verified column-for-column against Table II of the base paper;
`saurabh49` is verified against the column list saved in notebook 04's output.
Both assertions live in `tests/test_pipeline.py`.

### Steps

1. **Load & clean timestamps** — parse `_time` as UTC, drop rows with
   unparseable timestamps (`424,660 → 424,221`).
2. **Label attacks** — 5 known DDoS windows, both endpoints inclusive:

   | Attack | Window (UTC) |
   |---|---|
   | SYN Flood | 2024-08-18 07:00 → 08:00 |
   | ICMP Flood | 2024-08-19 07:00 → 09:41 |
   | UDP Fragmentation | 2024-08-19 17:00 → 18:00 |
   | DNS Flood | 2024-08-21 12:00 → 13:00 |
   | GTP-U Flood | 2024-08-21 17:00 → 18:00 |

   Result: ~397,597 benign / ~26,624 attack (6.28 %).
3. **Build the feature set** (the two paths above).
4. **Median-impute**, then **StandardScaler**.

Step 4 fits on the whole file — faithful to notebook 03, and the reason
`refit_scaler` exists (see §3).

### Outputs (`data/ncsrd/processed/`)

| File | Contents |
|---|---|
| `ncsrd_base38.csv` / `ncsrd_saurabh49.csv` | the datasets, scaled, no NaNs |
| `ncsrd_<set>_scaler.pkl` | the fitted `StandardScaler` |
| `ncsrd_<set>_meta.json` | column lists, dropped/kept columns, class counts, run config |
| `sequence_index.csv` | row-aligned `_time` + `imeisv`, shared by both sets |

> **Scaler gotcha.** For `saurabh49` the scaler was fit *before* zero-variance
> pruning, so it expects more columns than the CSV has. Transform with
> `meta["scaler_feature_names"]`, then select `meta["final_feature_names"]`.

---

## 2. Using the adapter

```python
from ncsrd_adapter import NetworkDataAdapter

ad = NetworkDataAdapter.for_feature_set("base38")   # or "saurabh49"
ad.project_standard_split()                          # or ad.reference_split()

b = ad.for_xgboost(balance="none")
model.fit(b.X_train, b.y_train)
probs = model.predict_proba(b.X_test)[:, 1]
```

Sanity-check a setup with `python src/common/data/ncsrd_adapter.py --check`.

### The `DataBundle`

| Attribute | Notes |
|---|---|
| `X_train`, `y_train` | resampled if you asked for it |
| `X_val`, `y_val` | `None` unless `val_size > 0` |
| `X_test`, `y_test` | **never** resampled |
| `train_index`, `val_index`, `test_index` | original row ids. `val`/`test` stay aligned with their labels; `train_index` is *pre*-resampling |
| `test_time`, `test_ue` | `_time` (int64 ns) and UE code per test row — group test predictions into windows with these |
| `input_shape`, `feature_names` | for Keras / SHAP |
| `class_weight`, `scale_pos_weight` | imbalance knobs, from the **post**-balancing labels (so `1.0` after SMOTE — don't double-correct) |
| `meta` | strategy, seed, balance, `refit_scaler`, timesteps |
| `.describe()` | printable summary; paste it into the report |

---

## 3. Splits

Use the two agreed policies rather than hand-rolling one:

| | `reference_split()` | `project_standard_split()` |
|---|---|---|
| Strategy | random stratified 80/20 | temporal 70/10/20 |
| Seed | 42 | 42 |
| Scaling | as `ncsrd_prep.py` fit it (global) | refit on training rows only |
| Purpose | notebook-04 parity, published-number comparison | the headline Base vs Saurabh vs Proposed table |

`strategy="group"` (`GroupShuffleSplit` on `imeisv`) is also available for
"does it work on an unseen device?". The capture has only 7 UEs, so expect
roughly 64/36 rather than 80/20 — a property of the dataset, not a bug.

> **Why the reference split is optimistic.** Rows are 5-second samples of the
> same 7 UEs, so consecutive rows are near-duplicates and a random split puts
> them on both sides. Two independent leaks: temporal (the model sees the
> future) and scaling (`ncsrd_prep.py` fit the scaler on all rows).
>
> **`refit_scaler=True` fixes the second one in one flag.** It re-standardizes
> every split using training-row statistics. Because standardization is affine
> per column, doing that to the already z-scored CSV is *numerically identical*
> to scaling raw data on train only (verified to 1e-15 in the test suite) — no
> need to re-run `ncsrd_prep.py`. It is stored in `save_split`/`load_split` and
> shown by `describe()`.

### One frozen split for everyone

```python
# one person, once
ad = NetworkDataAdapter.for_feature_set("saurabh49").project_standard_split()
ad.save_split(config.SPLIT_INDEX_FILE)      # commit this file

# everyone else, every method
ad = NetworkDataAdapter.for_feature_set(...).load_split(config.SPLIT_INDEX_FILE)
```

Both feature sets have identical rows in identical order, so one index file
covers both.

---

## 4. Class balance

`balance=` on any `for_*()` call. **Only the training split is ever touched.**

| `balance` | Effect | Needs |
|---|---|---|
| `"none"` | raw split | — |
| `"smote"` | synthetic minority samples (Saurabh's choice) | `imbalanced-learn` |
| `"undersample"` | drop random majority rows — closest to the base paper | — |
| `"oversample"` | duplicate random minority rows | — |
| `"class_weight"` | no resampling; use `bundle.class_weight` / `scale_pos_weight` | — |

The base paper uses **random undersampling** for XGBoost (SMOTE was for its
CNN/LSTM/MLP only). Saurabh uses **SMOTE**. Match the configuration you are
reproducing.

---

## 5. Data4Cyber

A **separate** adapter, deliberately not forced into the NCSRD schema: this is
a smart-grid / ICS testbed (Modbus + MQTT power-meter telemetry), not 5G.

```bash
python src/common/data/data4cyber_prep.py     # builds both split modes
```

| Mode | Split | Use |
|---|---|---|
| `block` | contiguous 120 s blocks, stratified by scenario + label | **primary** — the 3 × 2 comparison |
| `scenario` | whole scenarios held out (S6 unseen) | secondary — novel-attack robustness |

Outputs land in `data/data4cyber/processed/<mode>/`:

| File | Contents |
|---|---|
| `<split>_rows.npz` | `X (n, 140)`, `y`, `scenario`, `block`, `timestamp` — **per-sample, primary** |
| `<split>_windows.npz` | `X (n, 60, 140)`, `y` — per-window |
| `<split>_windows_metadata.json` | group + start/end time of each window |
| `manifest.json` | feature list, split definition, row/window counts and rates |

Every transform (median imputation, variance filter, standardization) is fit on
**training rows only**, and `*.realtime` / `Profile.timestamp` are always
excluded — all 8 scenarios occupy disjoint clock ranges, so those columns are a
perfect scenario id.

---

## 6. Dataset reference

### NCSRD-DS-5GDDoS — https://zenodo.org/records/13900057

| | |
|---|---|
| Source | `amari_ue_data_classic_tabular.csv`, an Amarisoft 5G testbed capture |
| Raw size | 424,660 rows × 80 columns; 424,221 usable after timestamp cleaning |
| Sampling | one row per UE per ~5 s |
| UEs | 7 distinct `imeisv` values |
| Cells | `cell_1_*` and `cell_3_*`; a UE attaches to one at a time, so ~57 % of per-cell columns are NaN |
| Label | `attack_label`, derived by time-window matching (no label column in the raw data) |
| Class balance | ~397,597 benign / ~26,624 attack — 6.28 % positive |
| Attack classes | SYN flood, ICMP flood, UDP fragmentation, DNS flood, GTP-U flood |
| Missing values | ~9.8 M NaNs, median-imputed; `bearer_1_apn` is 18 % missing and dropped |
| Categoricals | `*_apn`, `*_ip`, `*_ipv6` only — every modelled feature is numeric |
| Windowing | 500 samples (threshold / correlation), 1000 samples (SHAP drift) |

Assumptions: attack windows come from the dataset documentation and are treated
as ground truth; both endpoints inclusive. Saurabh's paper reports 397,576 /
26,645 — ~21 rows differ from ours, almost certainly a boundary convention.
Harmless, but state which you used.

### Data4Cyber — https://zenodo.org/records/19965384

| | |
|---|---|
| Source | 8 scenario folders, `dataset.csv` each; a smart-grid / ICS testbed |
| Raw size | 14,354 rows × 156 columns total (~1,800 rows per scenario) |
| Sampling | one row per second |
| Label | `attack_active` (`True`/`False`); `attack_phase` gives finer classes |
| Class balance | ~55 % attack overall — far more balanced than NCSRD |
| Attack classes | Industroyer (PV, BSS), ARP spoofing / MitM, MQTT supply-chain compromise |
| Features | 145 numeric candidates → 140 after the train-fit variance filter |
| Excluded | `Attacker.*` (trivial giveaway), `*.realtime` and `Profile.timestamp` (clock leakage) |
| Windowing | 60 s windows, 30 s stride, built inside one block/scenario |

Per-scenario attack rate: S0 0.00, S1 0.51, S1_alt 0.51, S2 0.51, S3 1.00,
S4 0.70, S5 0.74, S6 0.75. Attack is strongly time-localized (benign first,
attack later), which is why a naive chronological split would be degenerate and
the primary split assigns stratified contiguous blocks instead.

---

## 7. Troubleshooting

**`FileNotFoundError: ...ncsrd_saurabh49.csv not found`**
Run `python src/common/data/ncsrd_prep.py`, or pass an explicit path.

**`RuntimeError: strategy='temporal' needs sequence_index.csv`**
You ran with `--no-index`, or moved the CSV away from its sidecars. Regenerate.

**`ImportError: balance='smote' needs imbalanced-learn`**
`pip install imbalanced-learn`, or use `balance="undersample"` / `"class_weight"`.

**`MemoryError: ... > max_gib`**
LSTM windows are too big; the message names a stride that fits.

**`UserWarning: sequence_index.csv has N rows but the dataset has M`**
The files came from different runs. Regenerate both together.

**Everything scores > 0.99.** Expected on `reference_split()`. Re-run with
`project_standard_split()` before believing it.
