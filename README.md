# CN Project — Predictive Security for B5G/6G Networks

A **3 methods × 2 datasets** study of DDoS / anomaly detection.

| Method | NCSRD-DS-5GDDoS | Data4Cyber |
|---|---|---|
| **1. Base paper** — Xylouris et al., IEEE TCE 2025 | reproduce | adapt |
| **2. Saurabh's extension** — correlation graph, dynamic τ, SHAP drift | reproduce | adapt |
| **3. Our proposed work** — EWMA/CUSUM, FPR-constrained τ, fast SHAP, distillation | implement | adapt |

The research story is **Base → Saurabh → Proposed**, evaluated on both datasets,
plus an ablation of the proposed method.

> **Before writing any modelling code, read
> [docs/PROJECT_DECISIONS.md](docs/PROJECT_DECISIONS.md).** It is the single
> place defining feature sets, splits, thresholds and metric definitions. If
> your code disagrees with it, your code is wrong.
>
> [docs/Implementation.md](docs/Implementation.md) is the roadmap and division
> of work — a plan, not a binding spec.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Python 3.10–3.12.

## Data

Neither dataset is committed — they are large and git-ignored. Download and
unpack:

| Dataset | Source | Unpack to |
|---|---|---|
| NCSRD-DS-5GDDoS | https://zenodo.org/records/13900057 | `data/ncsrd/raw/` — needs `amari_ue_data_classic_tabular.csv` |
| Data4Cyber | https://zenodo.org/records/19965384 | `data/data4cyber/raw/` — the `S0_…`–`S6_…` folders |

Only each Data4Cyber scenario's `dataset.csv` is used; the `.pcapng` captures
are not needed by any pipeline.

Then build the processed datasets and check them:

```bash
python src/common/data/ncsrd_prep.py         # -> ncsrd_base38.csv + ncsrd_saurabh49.csv
python src/common/data/data4cyber_prep.py    # -> block/ (primary) + scenario/ (secondary)
python tests/test_pipeline.py                # 22 sanity checks
```

`ncsrd_prep.py` takes ~30 s and ~2 GB of peak RAM. Both scripts print every step.

## Structure

```
data/                     datasets — git-ignored, download them
  ncsrd/{raw,processed}/
  data4cyber/{raw,processed}/
src/
  common/
    config.py             frozen seed, split policies, paths. Import this.
    data/                 ncsrd_prep.py, ncsrd_adapter.py, data4cyber_prep.py
  base/                   M1 — base paper (XGBoost, static threshold)
  saurabh/                M2 — correlation graph, dynamic threshold, SHAP drift
  proposed/               M3 — EWMA, CUSUM, constrained τ, FastSHAP, distillation
experiments/{ncsrd,data4cyber}/   per-dataset run scripts
results/{raw,tables,plots}/
tests/test_pipeline.py    run after touching src/common/
run_experiment.py         M5 — unified runner (interface defined, not built yet)
docs/                     decisions, roadmap, reference papers
```

## Using the data

```python
import sys; sys.path.insert(0, "src/common/data")
from ncsrd_adapter import NetworkDataAdapter

ad = NetworkDataAdapter.for_feature_set("base38")    # or "saurabh49"
ad.project_standard_split()                           # or ad.reference_split()

b = ad.for_xgboost(balance="undersample")
model.fit(b.X_train, b.y_train)
probs = model.predict_proba(b.X_test)[:, 1]
```

`for_xgboost()` returns a `DataBundle` with `X_train/y_train`, `X_val/y_val`,
`X_test/y_test`, `feature_names`, `input_shape`, `class_weight`,
`scale_pos_weight`, and — for window-based evaluation — `train_index`,
`val_index`, `test_index`, `test_time`, `test_ue`. `.describe()` prints a
summary worth pasting into the report. The test split is never resampled.

For Data4Cyber, load `data/data4cyber/processed/block/<split>_rows.npz`
(`X`, `y`, `scenario`, `block`, `timestamp`). Row level is the primary metric
level; `<split>_windows.npz` exists for metrics that need a time axis.

**Two NCSRD feature sets, and they are different lineages — not variants.**
`base38` is the base paper's Table II; `saurabh49` is Saurabh's notebook 03.
They share only 6 column names. Never mix them in one comparison
(PROJECT_DECISIONS.md D1).

## Frozen settings

Do not change any of these alone — everything downstream depends on them.

| | | Where |
|---|---|---|
| Random seed | `42` | `config.SEED` |
| Reference split | random stratified 80/20, global scaling | `config.REFERENCE_SPLIT` |
| Project-standard split | temporal 70/10/20, train-only scaling | `config.PROJECT_STANDARD_SPLIT` |
| Frozen split indices | shared by all three methods | `config.SPLIT_INDEX_FILE` |
| NCSRD feature sets | `base38` / `saurabh49` | `ncsrd_prep.FEATURE_SETS` |
| Data4Cyber split | `block` primary, `scenario` secondary | `data4cyber_prep.SPLIT_MODES` |
| Threshold | chosen on validation, reported on test | PROJECT_DECISIONS.md D5 |
| Reported class | attack (label = 1) | PROJECT_DECISIONS.md D6 |
| Windows | 500 (threshold/correlation), 1000 (SHAP drift) | `config.THRESHOLD_WINDOW`, `config.SHAP_WINDOW` |

All paths resolve from the repository root via `src/common/config.py`. Never
hard-code an absolute path.

## Branches

`main` is the shared foundation. Each member works on their own branch and
merges back by PR.

| Branch | Owner | Scope |
|---|---|---|
| `feature/base` | M1 | `src/base/` |
| `feature/saurabh` | M2 | `src/saurabh/` |
| `feature/proposed` | M3 | `src/proposed/` |
| `feature/evaluation` | M5 | `run_experiment.py`, evaluator, tables, plots |

M4's pipeline work is already merged into `main`, so `feature/data-pipeline`
and `develop` are no longer needed.

Rules: no experimental commits straight to `main`; shared-interface changes get
discussed first; every experiment records its config and seed; nobody silently
changes preprocessing or the split for one method only.

## Troubleshooting

**`FileNotFoundError: ...ncsrd_saurabh49.csv`** — run `ncsrd_prep.py` first.

**`RuntimeError: strategy='temporal' needs sequence_index.csv`** — you ran with
`--no-index`, or moved the CSV away from its sidecars. Regenerate both together.

**`ImportError: balance='smote' needs imbalanced-learn`** — `pip install
imbalanced-learn`, or use `balance="undersample"` / `"class_weight"`.

**Everything scores > 0.99** — expected on `reference_split()`, which is
optimistic by construction. Re-run with `project_standard_split()` before
believing it.
