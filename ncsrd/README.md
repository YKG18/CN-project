# 5G UE-Level DDoS Detection — Preprocessing + Model Adapter

Two files, one job: turn the raw Amarisoft 5G capture into an ML-ready dataset,
then hand that dataset to **XGBoost, MLP, CNN and LSTM** in the exact shape each
one expects — using an **identical train/test split** so the four models are
actually comparable.

```
amari_ue_data_classic_tabular.csv   (424,660 x 80, raw)
            |
            |  prep.py          <- replicates notebook 03
            v
ue_attack_labeled_scaled.csv       (424,221 x 50 = 49 features + attack_label)
            |
            |  adapter.py       <- split logic from notebook 04
            v
   +--------+--------+--------+--------+
   | XGBoost|  MLP   |  CNN   |  LSTM  |
   |(n,49)  |(n,49)  |(n,49,1)|(n,T,49)|
   +--------+--------+--------+--------+
```

| File | What it is |
|---|---|
| `prep.py` | Script form of `03_attack_labeling_and_preprocessing.ipynb`. Labels, cleans, scales. Run once. |
| `adapter.py` | `NetworkDataAdapter` — splits and reshapes the processed data per model family. Import this. |
| `README.md` | You are here. |
| `requirements.txt` | Dependencies (core + optional). |

---

## 1. Install

```bash
pip install -r requirements.txt
```

`prep.py` and `adapter.py` only need **numpy, pandas, scikit-learn, joblib**.
Everything else (`xgboost`, `imbalanced-learn`, `tensorflow`/`torch`) is optional
and imported lazily — the adapter works fine without them and tells you exactly
what to `pip install` if you ask for a feature that needs one.

---

## 2. Run `prep.py` (once)

```bash
python prep.py -i path/to/amari_ue_data_classic_tabular.csv -o data/processed
```

| Flag | Default | Meaning |
|---|---|---|
| `-i`, `--input` | *required* | raw `amari_ue_data_classic_tabular.csv` |
| `-o`, `--outdir` | `data/processed` | where artifacts go |
| `--tol` | `0.0` | std threshold for dropping constant features (`0.0` = notebook-exact) |
| `--no-index` | off | skip `sequence_index.csv` — **don't**, the LSTM needs it |

Takes ~30 s (measured: 26 s) and roughly 2 GB of peak RAM. It prints every step so
you can check it against the notebook.

### What it does

1. **Load & clean timestamps** — parses `_time` as UTC, drops 439 rows with
   unparseable timestamps. `424,660 -> 424,221`.
2. **Label attacks** — 5 known DDoS windows from the dataset docs:

   | Attack | Window (UTC) | Rows |
   |---|---|---|
   | SYN Flood | 2024-08-18 07:00 → 08:00 | 4,206 |
   | ICMP Flood | 2024-08-19 07:00 → 09:41 | 11,244 |
   | UDP Fragmentation | 2024-08-19 17:00 → 18:00 | 4,194 |
   | DNS Flood | 2024-08-21 12:00 → 13:00 | 3,485 |
   | GTP-U Flood | 2024-08-21 17:00 → 18:00 | 3,495 |

   Result: **397,597 benign / 26,624 attack (6.28%)**.
3. **Drop 14 non-ML columns** — `imeisv`, `5g_tmsi`, `amf_ue_id`, `rnti`, `ran_id`,
   `ran_plmn`, `tac`, `tac_plmn`, `registered`, the 4 IP/IPv6 fields, and `_time`.
   Identifiers and addresses describe *which device*, not *how it behaved*; keeping
   them lets a model memorize UEs instead of learning attack signatures.
   `80 -> 67` columns.
4. **Consolidate per-cell retransmissions** — a UE attaches to one serving cell at
   a time, so `cell_1_*` / `cell_3_*` columns are ~57% NaN. Collapse
   `cell_{1,3}_ul_retx -> ul_retx_max` and `cell_{1,3}_dl_retx -> dl_retx_max`
   (row-wise max). `67 -> 65`.
5. **Median-impute** — 9.79M NaNs filled with column medians (robust to the
   heavy-tailed traffic distributions). 76,188 remain, all in `bearer_1_apn`.
6. **Drop `*_apn`** — categorical config, and `bearer_1_apn` is 18% missing.
   `65 -> 63`. Zero NaNs left.
7. **StandardScaler** on all 62 features (z-score).
8. **Drop 13 zero-variance features** — constants carry no signal:
   `bearer_{0,1}_qos_flow_id`, `bearer_{0,1}_sst`, `cell_{1,3}_cell_id`,
   `cell_{1,3}_ul_n_layer`, `cell_{1,3}_ul_rank`, `t3512`,
   `ue_aggregate_max_bitrate_{dl,ul}`. **`-> 49 features + attack_label`**.

### Outputs

| File | Contents |
|---|---|
| `ue_attack_labeled_scaled.csv` | **the dataset** — 424,221 x 50, scaled, zero NaNs (~400 MB) |
| `standard_scaler.pkl` | fitted `StandardScaler` — see the gotcha below |
| `sequence_index.csv` | row-aligned `_time` + `imeisv`. Not features; the adapter uses it for temporal/per-UE splits and LSTM windows |
| `prep_metadata.json` | every column list, both feature orders, class counts, run config |

> **Scaler gotcha.** The scaler was fit *before* zero-variance pruning, so it
> expects **62** columns, not 49. To scale new raw data for inference:
> transform with all 62 (`prep_metadata.json["scaler_feature_names"]`), then
> select the 49 in `prep_metadata.json["final_feature_names"]`.

---

## 3. Use the adapter

```python
from adapter import NetworkDataAdapter

ad = NetworkDataAdapter("data/processed/ue_attack_labeled_scaled.csv")

bundle = ad.for_xgboost()      # or .for_mlp() / .for_cnn() / .for_lstm()
model.fit(bundle.X_train, bundle.y_train)
```

That's the whole API. If you don't call `.split()` first, the adapter applies the
notebook-04 default automatically (random stratified 80/20, `random_state=42`).

Sanity-check your setup:

```bash
python adapter.py --check
```

builds all four bundles and prints their shapes.

### The `DataBundle` you get back

| Attribute | Type | Notes |
|---|---|---|
| `X_train`, `y_train` | ndarray | resampled if you asked for it |
| `X_val`, `y_val` | ndarray or `None` | `None` unless `val_size > 0` |
| `X_test`, `y_test` | ndarray | **never** resampled |
| `input_shape` | tuple | one sample, no batch axis → `keras.Input(shape=...)` |
| `class_weight` | `{0: w0, 1: w1}` | balanced weights from the training split |
| `scale_pos_weight` | float | `n_neg / n_pos` — the XGBoost imbalance knob |
| `feature_names` | list[str] | 49 names, in column order |
| `meta` | dict | split strategy, seed, balance mode, timesteps… |
| `.describe()` | str | printable summary — paste it in your report |

Features are `float32` (what Keras and PyTorch want); labels are `int8`.

---

## 4. The four model entry points

### XGBoost — `for_xgboost()` → `(n, 49)`

```python
import xgboost as xgb
from sklearn.metrics import classification_report

b = ad.for_xgboost(balance="smote")      # notebook-04 default

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)
model.fit(b.X_train, b.y_train)

probs = model.predict_proba(b.X_test)[:, 1]
print(classification_report(b.y_test, (probs > 0.5).astype(int)))
```

No `imbalanced-learn`? Skip SMOTE and let XGBoost handle the imbalance — cheaper
and usually just as good:

```python
b = ad.for_xgboost(balance="class_weight")
model = xgb.XGBClassifier(scale_pos_weight=b.scale_pos_weight, ...)   # ~14.93
```

For SHAP or `plot_importance`, keep the feature names attached:

```python
b = ad.for_xgboost(as_frame=True)        # DataFrames instead of ndarrays
shap.TreeExplainer(model).shap_values(b.X_test.sample(1000, random_state=42))
```

### MLP — `for_mlp()` → `(n, 49)`

```python
from tensorflow import keras          # or: import keras

b = ad.for_mlp(balance="smote")

model = keras.Sequential([
    keras.Input(shape=b.input_shape),          # (49,)
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc"),
                       keras.metrics.Recall(name="recall")])
model.fit(b.X_train, b.y_train, epochs=15, batch_size=512,
          validation_data=(b.X_val, b.y_val) if b.has_val else None)
```

Skipped SMOTE? Add `class_weight=b.class_weight` to `fit`.

### CNN — `for_cnn()` → `(n, 49, 1)`

A 1-D CNN over the **feature axis** (each row is one 5-second snapshot).
Channels-last, i.e. what Keras `Conv1D` expects.

```python
b = ad.for_cnn(balance="smote")

model = keras.Sequential([
    keras.Input(shape=b.input_shape),          # (49, 1)
    keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
```

**PyTorch** wants channels-first — transpose once:

```python
X = b.X_train.transpose(0, 2, 1)     # (n, 1, 49) for nn.Conv1d
```

Want a CNN over **time** instead of over features? Use the LSTM bundle — its
`(n, timesteps, 49)` output feeds a temporal `Conv1D` unchanged.

### LSTM — `for_lstm()` → `(n, timesteps, 49)`

The processed CSV has no time column, so the adapter rebuilds the time axis from
`sequence_index.csv`: rows are sorted by `(imeisv, _time)` and cut into sliding
windows **within each UE**, so no window ever splices two devices together.

```python
b = ad.for_lstm(timesteps=10, stride=1, label_mode="last")

model = keras.Sequential([
    keras.Input(shape=b.input_shape),          # (10, 49)
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc")])
model.fit(b.X_train, b.y_train, epochs=15, batch_size=256,
          class_weight=b.class_weight)         # <- default balance mode
```

| Parameter | Default | Meaning |
|---|---|---|
| `timesteps` | `10` | window length in samples (~5 s each, so ~50 s of history) |
| `stride` | `1` | step between window starts; `stride == timesteps` → disjoint |
| `label_mode` | `"last"` | `"last"` = label of the final sample; `"any"` = attack if any sample is |
| `sequence_mode` | `"windows"` | real time series. `"features"` = reshape each row to `(49, 1)` — no time axis, smoke tests only |
| `balance` | `"class_weight"` | see below |
| `max_gib` | `4.0` | refuse to allocate more than this; the error suggests a stride |

Memory scales with `n_windows x timesteps x 49 x 4 bytes`. Measured:

| `timesteps` | `stride` | windows | RAM |
|---|---|---|---|
| 10 | 1 | 424,166 | 0.77 GiB |
| 10 | 10 | 42,419 | 0.08 GiB |
| 20 | 1 | 424,106 | 1.55 GiB |
| 32 | 4 | 106,011 | 0.62 GiB |

Start with `timesteps=10, stride=10` while you're iterating; drop to `stride=1`
for the final run.

> **Why `class_weight` and not SMOTE here?** SMOTE interpolates between samples,
> which is meaningless across a time axis. `balance="oversample"` /
> `"undersample"` duplicate or drop **whole windows** and are safe.
> `balance="smote"` is accepted but flattens time first — it warns, and you
> shouldn't.

---

## 5. Splits

```python
ad.split(strategy="random", test_size=0.2, val_size=0.0,
         random_state=42, stratify=True)
```

Call it once; every `for_*()` afterwards uses the same partition. Returns `self`,
so it chains: `ad.split(...).for_mlp()`.

| `strategy` | How | Use it for |
|---|---|---|
| `"random"` **(default)** | stratified `train_test_split` — **exactly notebook 04** | reproducing the notebook |
| `"temporal"` | oldest 80% train, newest 20% test | an honest generalization estimate |
| `"group"` | `GroupShuffleSplit` on `imeisv` — no UE on both sides | "does it work on an unseen device?" |

With the default: **train 339,376 / test 84,845**, test labels
`[79520 benign, 5325 attack]` — identical to notebook 04's classification-report
supports.

`val_size` is a fraction of the **whole** dataset, carved out of train
(`test_size=0.2, val_size=0.1` → 70/10/20).

> **Read this before you report accuracy.** The default random split is what
> notebook 04 does, and it is optimistic. Rows are 5-second samples of the same
> 7 UEs, so consecutive rows are near-duplicates — a random split puts
> near-duplicates in *both* train and test, and the model can score well by
> near-memorization. Two independent reasons to be careful:
>
> - **Temporal leakage.** Random shuffling lets the model see the future.
> - **Scaling leakage.** `prep.py` fits `StandardScaler` on all 424,221 rows
>   before any split (faithful to notebook 03), so test-set statistics leak into
>   the training features.
>
> Report `strategy="random"` for notebook parity, and `strategy="temporal"`
> alongside it. The gap between them is the honest headline. For a fully
> leakage-free run, re-fit the scaler on the training rows only rather than
> using `prep.py`'s pre-scaled CSV.

### Same split for all four models

Do this so nobody accidentally compares apples to oranges:

```python
# one person, once
ad = NetworkDataAdapter().split(strategy="temporal", test_size=0.2, val_size=0.1)
ad.save_split("data/processed/split_indices.npz")   # commit this file

# everyone else
ad = NetworkDataAdapter().load_split("data/processed/split_indices.npz")
b = ad.for_cnn()
```

`split_indices.npz` stores the row indices plus the config, so the partition is
byte-identical across machines regardless of library versions.

---

## 6. Class balance options

`balance=` on any `for_*()` call. **Only the training split is ever touched** —
val and test stay at the real 6.28% attack rate, which is the whole point.

| `balance` | Effect | Needs | Good for |
|---|---|---|---|
| `"smote"` | synthetic minority samples (notebook-04 default) | `imbalanced-learn` | XGBoost, MLP, CNN |
| `"class_weight"` | no resampling; use `bundle.class_weight` / `bundle.scale_pos_weight` | — | LSTM (default), and anything memory-bound |
| `"oversample"` | duplicate random minority rows/windows | — | SMOTE-free alternative, sequence-safe |
| `"undersample"` | drop random majority rows/windows | — | fast iteration on a smaller set |
| `"none"` | raw split | — | measuring the imbalance itself |

`class_weight` and `scale_pos_weight` are always populated from the **post-balancing**
training labels, so after SMOTE they're `1.0` — don't double-correct.

---

## 7. Compare all four

```python
from adapter import NetworkDataAdapter

ad = NetworkDataAdapter().split(strategy="temporal", test_size=0.2, val_size=0.1)

for name in ["xgboost", "mlp", "cnn", "lstm"]:
    b = ad.get(name, balance="class_weight")
    print(b.describe())
    # ... train, evaluate, log to your results table
```

`ad.get(name, **kwargs)` dispatches to the right `for_*()`. Note that the LSTM's
test set counts **windows**, not rows, so report its metrics as such rather than
comparing raw support numbers with the other three.

### Threshold tuning

With 6.28% positives, `0.5` is rarely the best cut. Notebook 04 finds `0.708`;
do the same for every model:

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

precision, recall, thresholds = precision_recall_curve(b.y_test, probs)
f1 = 2 * precision * recall / (precision + recall + 1e-10)
best = thresholds[np.argmax(f1)]
print("best threshold", best, "best F1", f1.max())
```

Pick the threshold on **validation**, then report on test. Choosing it on test —
as notebook 04 does — reports an optimistically biased number.

---

## 8. Troubleshooting

**`FileNotFoundError: ...ue_attack_labeled_scaled.csv not found`**
Run `prep.py` first, or point the adapter at the file:
`NetworkDataAdapter("some/other/path.csv")`.

**`RuntimeError: strategy='temporal' needs sequence_index.csv`**
You ran `prep.py --no-index`, or moved the CSV away from its sidecars. Re-run
`prep.py` without `--no-index`, or pass `index_path=` explicitly. `for_xgboost`,
`for_mlp`, `for_cnn` and `strategy="random"` all work without it.

**`ImportError: balance='smote' needs imbalanced-learn`**
`pip install imbalanced-learn`, or use `balance="class_weight"` /
`balance="oversample"`.

**`MemoryError: ... > max_gib=4.0`**
LSTM windows are too big. The message tells you the stride that fits. Or raise
`max_gib` if you have the RAM.

**`UserWarning: sequence_index.csv has N rows but the dataset has M`**
The two files came from different `prep.py` runs. Regenerate both together.

**Group split isn't 80/20.** The capture has only **7 UEs**, so
`GroupShuffleSplit` can't hit an exact ratio — expect roughly 64/36. That's a
property of the dataset, not a bug. Treat per-UE results as directional.

**All four models score >0.99.** Expected on the default random split (see the
leakage note in §5). Re-run with `strategy="temporal"` before believing it.

---

## 9. Reproducing the notebooks exactly

```python
from adapter import NetworkDataAdapter

ad = NetworkDataAdapter("data/processed/ue_attack_labeled_scaled.csv")
b  = ad.for_xgboost(balance="smote")   # split: random, 0.2, seed 42; SMOTE on train
```

Verified against the reference notebooks:

| Checkpoint | Notebook | This code |
|---|---|---|
| rows after cleaning | 424,221 | 424,221 |
| final shape | (424221, 50) | (424221, 50) |
| class counts | 397,597 / 26,624 | 397,597 / 26,624 |
| zero-variance columns dropped | 13 | 13 (same names, same order) |
| train / test shape | (339376, 49) / (84845, 49) | (339376, 49) / (84845, 49) |
| test class support | 79,520 / 5,325 | 79,520 / 5,325 |
| after SMOTE | 318,077 / 318,077 | 318,077 / 318,077 |

Feature values match the notebook's printed output to six decimals.
