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
python src/common/data/freeze_split.py       # freeze the shared project-standard split
python tests/test_pipeline.py                # pipeline sanity checks
```

`ncsrd_prep.py` takes ~30 s and ~2 GB of peak RAM. Both scripts print every step.

## Running the experiments

One command runs the complete 3 × 2 matrix and prints the final comparison:

```bash
python run_experiment.py --all
```

It runs Base / Saurabh / Proposed on NCSRD and Data4Cyber, scores every cell
through the shared evaluator (`src/common/evaluator.py`, D6 definitions), and
prints one table with the detection metrics, confusion counts, threshold,
latency, model size and window-level FPR stability. Add `--save` to write the
D6 rows and detail JSON under `results/raw/`. Takes roughly 15–25 minutes.

`--all` always uses the **NCSRD project-standard split** and the **Data4Cyber
primary `block` split**. The Data4Cyber `scenario` holdout is a *secondary*
novel-attack robustness experiment (D3) and is deliberately never part of the
matrix.

Single cells, when you need one:

```bash
python run_experiment.py --dataset ncsrd      --method base     --config base_paper
python run_experiment.py --dataset ncsrd      --method saurabh  --config saurabh_full
python run_experiment.py --dataset ncsrd      --method proposed --config P6
python run_experiment.py --dataset data4cyber --method base
```

### Reproducing the published papers

These are **reference reproductions**, kept separate from the project-standard
comparison above, and are what to show if asked "did you reproduce the paper?"

```bash
# Base paper (Xylouris et al., Table III) — 38 features, random stratified split
python experiments/ncsrd/run_base.py --split reference

# Saurabh's pipeline + ablation (leave-one-out AND his additive A0–A4)
python experiments/ncsrd/run_saurabh.py --experiment all

# Proposed P0–P7 ablation
python experiments/ncsrd/run_proposed.py

# Non-stationary benign traffic (faculty direction 5)
python experiments/ncsrd/run_nonstationary.py

# Dual-divergence experiment (step C, experimental — not in the 3x2)
python experiments/ncsrd/run_dual_divergence.py
```

**Note on the Saurabh reproduction.** `run_saurabh.py` runs on the
project-standard frozen split with `class_weight`, not on Saurabh's own random
80/20 split with SMOTE, and its per-window thresholds are fitted on validation
and applied cyclically rather than re-optimised on each test window's labels.
Those are deliberate leakage-safe deviations (D5), so its numbers are **not**
expected to match his published 0.9573 / 0.9730 exactly. Only the Base paper
has a true `--split reference` reproduction path.

`experiments/ncsrd/run_base.py` is the authoritative base-paper reproduction:
it selects the undersampling ratio on validation, which `run_experiment.py`
does not (that uses the 1:1 default for speed), so the two give slightly
different reference numbers. Both are correct; quote the dedicated runner.

Deeper single-method runs, including the Data4Cyber secondary experiment:

```bash
python experiments/data4cyber/run_base.py --split both   # primary + secondary
python experiments/data4cyber/run_saurabh_data4cyber.py
```

## Structure

```
data/                     datasets — git-ignored, download them
  ncsrd/{raw,processed}/
  data4cyber/{raw,processed}/
src/
  common/
    config.py             frozen seed, split policies, paths. Import this.
    evaluator.py          M5 — the ONE metric implementation (D6 schema)
    data/                 ncsrd_prep.py, ncsrd_adapter.py,
                          data4cyber_prep.py, data4cyber_adapter.py,
                          freeze_split.py
  base/base_xgboost.py    M1 — base paper (XGBoost, static threshold)
  saurabh/saurabh_xgboost.py
                          M2 — correlation graph, dynamic threshold, SHAP drift
  proposed/               M3 — proposed_xgboost.py + ewma, cusum,
                          constrained_threshold, fast_shap, distillation,
                          nonstationary, adaptive_threshold,
                          dual_divergence
experiments/ncsrd/        run_base.py, run_saurabh.py, run_proposed.py,
                          run_nonstationary.py, run_dual_divergence.py
experiments/data4cyber/   run_base.py, run_saurabh_data4cyber.py
results/{raw,tables,plots}/       M5 owns final result collection
tests/                    test_pipeline.py, test_base.py, test_evaluator.py,
                          test_proposed.py, test_saurabh_ablation.py,
                          test_saurabh_integration.py
run_experiment.py         M5 — unified runner; `--all` runs the whole 3×2
docs/                     decisions, roadmap, reference papers
```

XGBoost is the only model family in the project. CNN/MLP/LSTM are **not**
implemented (PROJECT_DECISIONS.md D8); some adapters can emit their tensor
shapes, but no such model exists here.

## Using the data

```python
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "src/common/data")
from common import config
from ncsrd_adapter import NetworkDataAdapter

ad = NetworkDataAdapter.for_feature_set("base38")    # or "saurabh49"

# headline runs: load the frozen shared split, never roll your own
ad.load_split(config.SPLIT_INDEX_FILE)
# reproduction runs against published numbers:
# ad.reference_split()

b = ad.for_xgboost(balance="undersample")
model.fit(b.X_train, b.y_train)
probs = model.predict_proba(b.X_test)[:, 1]
```

For project-standard runs, build evaluation windows **inside one block**
(`b.test_block`) — test rows are blocks scattered through the capture, so a
window crossing a boundary joins moments hours apart. See PROJECT_DECISIONS.md
D2.

`for_xgboost()` returns a `DataBundle` with `X_train/y_train`, `X_val/y_val`,
`X_test/y_test`, `feature_names`, `input_shape`, `class_weight`,
`scale_pos_weight`, and — for window-based evaluation — `train_index`,
`val_index`, `test_index`, `test_time` (int64 nanoseconds), `test_ue` and
`test_block`. `.describe()` prints a summary worth pasting into the report.
The test split is never resampled.

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
| Project-standard split | stratified contiguous 20-min blocks, 70/10/20, train-only scaling | `config.PROJECT_STANDARD_SPLIT` |
| Block length | 20 minutes (fits a 1000-sample SHAP window) | `config.BLOCK_MINUTES` |
| Frozen split indices | shared by all three methods; build with `src/common/data/freeze_split.py` | `config.SPLIT_INDEX_FILE` |
| NCSRD feature sets | `base38` / `saurabh49` | `ncsrd_prep.FEATURE_SETS` |
| Data4Cyber split | `block` primary, `scenario` secondary | `data4cyber_prep.SPLIT_MODES` |
| Threshold | chosen on validation, reported on test | PROJECT_DECISIONS.md D5 |
| Reported class | attack (label = 1) | PROJECT_DECISIONS.md D6 |
| Windows | 500 (threshold/correlation), 1000 (SHAP drift) | `config.THRESHOLD_WINDOW`, `config.SHAP_WINDOW` |

All paths resolve from the repository root via `src/common/config.py`. Never
hard-code an absolute path.

## Ownership

All members' work is merged into `main`.

| Area | Owner |
|---|---|
| `src/common/data/`, both pipelines | M4 |
| `src/base/` | M1 |
| `src/saurabh/` | M2 |
| `src/proposed/` | M3 |
| `run_experiment.py`, `src/common/evaluator.py`, `results/` | M5 |

Rules: shared-interface changes get discussed first; every experiment records
its config and seed; nobody silently changes preprocessing or the split for one
method only; metrics come from the shared evaluator, never hand-rolled.

## Tests

```bash
python tests/test_pipeline.py            # shared pipelines, splits, leakage
python tests/test_base.py                # M1 base-paper methodology
python tests/test_evaluator.py           # M5 D6 metric definitions
python tests/test_proposed.py            # M3 components, thresholds, distillation
python tests/test_saurabh_ablation.py    # M2 module on/off matrix
python tests/test_saurabh_integration.py # M2 end-to-end chain
```

The two Saurabh files cover different things and both are kept:
`_ablation` sweeps the module on/off combinations and checks each module's
effect (feature count, threshold stats, drift windows); `_integration` runs one
full configuration end to end and checks the whole chain produces sane,
non-degenerate output.

## Troubleshooting

**`FileNotFoundError: ...ncsrd_saurabh49.csv`** — run `ncsrd_prep.py` first.

**`RuntimeError: strategy='temporal' needs sequence_index.csv`** — you ran with
`--no-index`, or moved the CSV away from its sidecars. Regenerate both together.

**`ImportError: balance='smote' needs imbalanced-learn`** — `pip install
imbalanced-learn`, or use `balance="undersample"` / `"class_weight"`.

**Everything scores > 0.99** — expected on `reference_split()`, which is
optimistic by construction. Re-run with `project_standard_split()` before
believing it.
