# CN Project — Predictive Security for B5G/6G Networks

A **3 methods × 2 datasets** study of DDoS / anomaly detection.

| Method | NCSRD-DS-5GDDoS | Data4Cyber |
|---|---|---|
| **1. Base paper** (Xylouris et al., IEEE TCE 2025) | reproduce | adapt |
| **2. Saurabh's extension** (correlation graph, dynamic τ, SHAP drift) | reproduce | adapt |
| **3. Our proposed work** (EWMA/CUSUM, FPR-constrained τ, fast SHAP, distillation) | implement | adapt |

**Start here:**

1. **[docs/PROJECT_DECISIONS.md](docs/PROJECT_DECISIONS.md)** — the rules everyone
   follows: feature sets, splits, thresholds, metric definitions. Read before
   writing any modelling code.
2. [docs/Implementation.md](docs/Implementation.md) — phase plan and division of work.
3. [docs/AUDIT_AND_DECISIONS.md](docs/AUDIT_AND_DECISIONS.md) — what the pipeline
   audit found and fixed.
4. [src/common/data/README.md](src/common/data/README.md) — how to use the pipelines.

## Layout

```
data/                 datasets — git-ignored, download them (see below)
  ncsrd/{raw,processed}/
  data4cyber/{raw,processed}/
src/
  common/
    config.py         frozen seed, split policies, paths. Import this.
    data/             ncsrd_prep.py, ncsrd_adapter.py, data4cyber_prep.py
  base/               M1 — base paper (XGBoost, static threshold)
  saurabh/            M2 — correlation graph, dynamic threshold, SHAP drift
  proposed/           M3 — EWMA, CUSUM, constrained threshold, FastSHAP, distillation
tests/test_pipeline.py  pipeline sanity checks — run after touching src/common/
experiments/          per-dataset run scripts
results/{raw,processed,tables,plots}/
run_experiment.py     M5 — unified runner
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Python 3.10–3.12.

## Data

Neither dataset is committed — they are large. Download and unpack:

| Dataset | Source | Unpack to |
|---|---|---|
| NCSRD-DS-5GDDoS | https://zenodo.org/records/13900057 | `data/ncsrd/raw/` — needs `amari_ue_data_classic_tabular.csv` |
| Data4Cyber | https://zenodo.org/records/19965384 | `data/data4cyber/raw/` — the `S0_…`–`S6_…` folders |

Only each scenario's `dataset.csv` is used from Data4Cyber; the `.pcapng`
captures are not needed by any pipeline.

Then build the processed datasets:

```bash
python src/common/data/ncsrd_prep.py        # -> ncsrd_base38.csv + ncsrd_saurabh49.csv
python src/common/data/data4cyber_prep.py   # -> block/ (primary) + scenario/ (secondary)
python tests/test_pipeline.py               # 22 sanity checks
```

## Using the data

```python
import sys; sys.path.insert(0, "src/common/data")
from ncsrd_adapter import NetworkDataAdapter

ad = NetworkDataAdapter.for_feature_set("base38")   # or "saurabh49"
ad.project_standard_split()                          # or ad.reference_split()

b = ad.for_xgboost(balance="undersample")
model.fit(b.X_train, b.y_train)
```

The two NCSRD feature sets are **different lineages, not variants** — `base38`
is the base paper's Table II, `saurabh49` is Saurabh's notebook 03. They share
only 6 column names. See PROJECT_DECISIONS.md D1.

## Branches

`main` holds everything shared: plan, decisions, config, data pipelines, tests,
docs. Work on your feature branch and merge back by PR.

| Branch | Owner | Scope |
|---|---|---|
| `feature/data-pipeline` | M4 | `src/common/`, dataset adapters, synthetic traffic |
| `feature/base` | M1 | `src/base/` |
| `feature/saurabh` | M2 | `src/saurabh/` |
| `feature/proposed` | M3 | `src/proposed/` |
| `feature/evaluation` | M5 | `run_experiment.py`, evaluator, tables, plots |

Rules: no experimental commits straight to `main`; shared-interface changes get
discussed first; every experiment records its config and seed; nobody silently
changes preprocessing or the split for one method only.

> **Note for M4:** `feature/data-pipeline` still carries ~400 MB of raw
> Data4Cyber in its history and should not be merged as is. See
> AUDIT_AND_DECISIONS.md §5.
