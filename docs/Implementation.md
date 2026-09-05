# Computer Networks Project — Complete Implementation Plan

> **Read `docs/PROJECT_DECISIONS.md` first.** This document is the phase
> plan and division of work. Where the two disagree on a *policy* — feature
> sets, splits, thresholds, metric definitions — PROJECT_DECISIONS.md wins;
> it records decisions taken after this plan was written. Sections updated
> to match are marked **[updated]**.

## 1. Project Structure

Our project should evaluate **3 approaches × 2 datasets**:

| Approach | NCSRD-DS-5GDDoS | Data4Cyber |
|---|---|---|
| **1. Base Paper** | Reproduce + evaluate | Adapt + evaluate |
| **2. Saurabh's Work** | Reproduce + evaluate | Adapt + evaluate |
| **3. Our Proposed Work** | Implement + evaluate | Adapt + evaluate |

The research progression is:

**Base → Saurabh → Proposed**

The evaluation progression is:

**NCSRD → Data4Cyber**

Alongside this, we conduct an **ablation study of our proposed method**.

### Why this structure?

- **NCSRD** is the dataset used by the base paper and Saurabh's work, so it provides the cleanest apples-to-apples comparison.
- **Data4Cyber** gives us a second environment in which to test whether the improvements remain useful beyond NCSRD.
- The **ablation study** shows which parts of our proposed method actually contribute to the improvement.

> Important: evaluating separately on two datasets demonstrates cross-dataset robustness. It is not true cross-dataset generalization unless we explicitly train on one dataset and test on another.

---

# 2. Overall System Architecture

We should eventually have one common experimental framework rather than three completely independent projects.

```text
                         DATASET
                    NCSRD / Data4Cyber
                           |
                           v
                 COMMON DATA PIPELINE
            cleaning / labels / scaling / split
                           |
                           v
                  COMMON WINDOWING
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
       BASE PAPER       SAURABH          PROPOSED
       XGBoost          Graph +          EWMA/CUSUM
       static τ         Adaptive τ       constrained τ
                       + SHAP drift      FastSHAP
                                         Distillation
          |                |                |
          +----------------+----------------+
                           |
                           v
                    COMMON EVALUATOR
                           |
          +----------------+----------------+
          |                |                |
       Accuracy         Stability        Deployment
       Precision        FPR variance     Latency
       Recall            FPR max         Memory
       F1                 τ variance     Model size
       AUC                                Power
                           |
                           v
                 COMPARISON + ABLATION
```

The important design principle is:

> **Only the method should change. Dataset handling, evaluation and reporting should be standardized wherever possible.**

---

# 3. PHASE 0 — Project Setup

## Goal

Create the common infrastructure before serious model development begins.

## Required actions

### 3.1 Create one GitHub repository

**Owner: Member 4**

Suggested structure:

**[updated]** — this is the structure now on `main`:

```text
project-root/
│
├── data/                        git-ignored; download per README
│   ├── ncsrd/{raw,processed}/
│   └── data4cyber/{raw,processed}/
│
├── src/
│   ├── common/
│   │   ├── config.py            frozen seed, split policies, paths
│   │   └── data/                ncsrd_prep, ncsrd_adapter, data4cyber_prep
│   ├── base/                    M1
│   ├── saurabh/                 M2
│   └── proposed/                M3
│
├── experiments/{ncsrd,data4cyber}/
├── results/{raw,processed,tables,plots}/
├── tests/test_pipeline.py       pipeline sanity checks — run after edits
├── notebooks/
├── configs/
│
├── requirements.txt   README.md   SETUP.md
├── EXPERIMENTS.md     RESULTS.md  run_experiment.py
└── docs/
    ├── PROJECT_DECISIONS.md     the rules everyone follows
    ├── Implementation.md        this file
    └── AUDIT_AND_DECISIONS.md   pipeline audit record
```

### 3.2 Git workflow

Use separate branches:

```text
main
develop
feature/base
feature/saurabh
feature/proposed
feature/data-pipeline
feature/evaluation
```

Rules:

1. Nobody directly pushes experimental changes to `main`.
2. Each member works mainly inside their assigned module.
3. Changes that affect shared interfaces must be discussed before merging.
4. Every experiment should record its configuration and random seed.
5. Never silently change preprocessing or train/test splits for only one method.

### 3.3 Freeze the environment

The group should agree on:

- Python version
- package versions
- random seed
- train/test split strategy
- preprocessing conventions
- windowing conventions
- metric definitions

**[updated] — DONE.** All of it lives in two files:

* `requirements.txt` (root, the only one) — Python 3.10-3.12 and pinned minimums.
* `src/common/config.py` — `SEED`, `REFERENCE_SPLIT`, `PROJECT_STANDARD_SPLIT`,
  `SPLIT_INDEX_FILE`, window sizes, and every path resolved from the repo root.

Import from `config`; never hard-code a seed, a split fraction or a path.
Metric definitions are in PROJECT_DECISIONS.md D6.

### 3.4 Define common interfaces

Before the project becomes large, agree on four interfaces:

```text
1. Dataset → standardized X, y and metadata

2. Model → fit(), predict_proba(), predict()

3. Windowing → standardized windows

4. Evaluation → standardized metrics/results
```

This is what prevents the three implementations from becoming incompatible.

**[updated]** — 1 and 3 exist. `NetworkDataAdapter` returns a `DataBundle`
carrying `X_train/y_train`, `X_val/y_val`, `X_test/y_test`, `feature_names`,
`input_shape`, `class_weight`, `scale_pos_weight`, and — for window-based
evaluation — `train_index` / `val_index` / `test_index` plus `test_time` and
`test_ue`. Data4Cyber emits the equivalent as `<split>_rows.npz` and
`<split>_windows.npz`. Interface 2 is each member's own class. Interface 4 is
M5's evaluator; until it lands, emit the results row schema in
PROJECT_DECISIONS.md D6.

---

# 4. PHASE 1 — Common Dataset Engineering

**Owner: Member 4**

Nobody should start serious cross-method comparison until this pipeline is stable.

## 4.1 NCSRD pipeline

https://zenodo.org/records/13900057

Build:

```text
Raw NCSRD
    ↓
Cleaning
    ↓
Attack labeling
    ↓
Feature selection
    ↓
Scaling
    ↓
Train/Test split
    ↓
Window generation
```

**[updated]** — the pipeline builds **two** feature sets from identical rows,
because the base paper and Saurabh do not use the same features
(PROJECT_DECISIONS.md, D1):

| set | features | used by |
|---|---|---|
| `base38` | the base paper's Table II, all 24 cell metrics consolidated | M1's true base-paper config |
| `saurabh49` | notebook 03's features, only `*_retx` consolidated | Saurabh-style static baseline, M2, M3 |

They share only 6 column names and must not be mixed.

Saurabh's own setup is 49 features, an 80/20 stratified split and SMOTE on the
training set only — reproduced exactly by `saurabh49` + `ad.reference_split()`.

```bash
python src/common/data/ncsrd_prep.py      # builds both sets + sequence_index
```

## 4.2 Data4Cyber pipeline

https://zenodo.org/records/19965384

Build a separate adapter:

```text
Raw Data4Cyber
    ↓
Schema analysis
    ↓
Label identification
    ↓
Feature selection
    ↓
Preprocessing
    ↓
Train/Test split
    ↓
Window/stream representation
```

Do **not** force Data4Cyber to have exactly the same feature structure as NCSRD.

The algorithms should have a common interface, but each dataset can have its own preprocessing adapter.

**[updated]** — two splits are produced (PROJECT_DECISIONS.md, D3):

* **`block` (primary)** — contiguous 120 s blocks assigned to train/val/test,
  stratified by scenario and label. Every attack family is in every split; no
  block or window straddles a boundary. This feeds the 3 × 2 comparison.
* **`scenario` (secondary)** — whole scenarios held out, so the test set has an
  unseen attack family. A robustness experiment, reported separately.

Absolute wall-clock columns (`*.realtime`, `Profile.timestamp`) are always
excluded: the 8 scenarios occupy disjoint clock ranges, making them a perfect
scenario id and a direct label shortcut.

```bash
python src/common/data/data4cyber_prep.py    # builds both split modes
```


## 4.3 Dataset documentation

Member 4 should document:

- dataset size
- features
- labels
- attack classes
- missing values
- class balance
- chosen preprocessing
- train/test split
- window size
- any dataset-specific assumptions

**[updated] — DONE:** see "Dataset reference" in
[`src/common/data/README.md`](../src/common/data/README.md), which covers both
datasets, and PROJECT_DECISIONS.md for the split and metric policy.

---

# 5. PHASE 2 — Reproduce the Base Paper

**Owner: Member 1**

The base paper evaluates CNN, LSTM, MLP and XGBoost. XGBoost is the key baseline because it gives the strongest reported performance and forms the foundation for Saurabh's work.

## 5.1 First target: Base XGBoost

```text
NCSRD
  ↓
Preprocessing
  ↓
XGBoost
  ↓
Static threshold
  ↓
Evaluation
```

Target reference region from the base paper (attack class):

```text
Accuracy  ≈ 0.996
F1        ≈ 0.97
Precision ≈ 0.96
Recall    ≈ 0.98
```

**[updated]** — M1 produces **two** configurations, kept strictly separate
(PROJECT_DECISIONS.md, D1):

| config | feature set | imbalance | purpose |
|---|---|---|---|
| `base_paper` | `base38` | random undersampling + `scale_pos_weight` | reproduce the base paper |
| `base_saurabh_static` | `saurabh49` | SMOTE on train | anchor Saurabh's ablation row A0 (F1 0.9573) |

and each is run under both split policies (D2): `reference` for comparison with
published numbers, `project_standard` for the headline 3 × 1 / 3 × 2 tables.

## 5.2 Required outputs

```text
base_ncsrd_results.csv
```

plus:

- confusion matrix
- ROC curve
- PR curve
- precision
- recall
- F1
- FPR
- AUC
- inference timing

## 5.3 Optional

Reproduce CNN/LSTM/MLP if time permits.

They are useful for demonstrating understanding of the base paper, but they are not the central path of the final 3×2 comparison.

## 5.4 Data4Cyber

After the NCSRD base is stable:

```text
Data4Cyber
  ↓
base preprocessing adapter
  ↓
base XGBoost
  ↓
evaluation
```

---

# 6. PHASE 3 — Reproduce Saurabh's Work

**Owner: Member 2**

The faculty-provided project package contains:

```text
01_data_understanding.ipynb
02_eda.ipynb
03_attack_labeling_and_preprocessing.ipynb
04_model_training_and_evaluation.ipynb
05_correlation_behavioral_graph.ipynb
06_dynamic_threshold_adaptation.ipynb
07_shap_active_drift_detection.ipynb
08_integration_ablation_study.ipynb
GUIDELINES TO RUN CODE.txt
```

The first goal is to run it **as supplied**, without immediately modifying it.

## 6.1 Module A — Correlation Behavioral Graph

```text
49 features
   ↓
Pearson correlation
   ↓
Benign baseline
   ↓
Window correlation
   ↓
Frobenius divergence
   ↓
graph_frob_div
```

The graph divergence is added as a feature to the XGBoost model.

## 6.2 Module B — Dynamic Threshold

```text
XGBoost probabilities
       ↓
500-sample window
       ↓
Precision–Recall curve
       ↓
F1-optimal threshold
```

## 6.3 Module C — SHAP Drift

```text
1000-sample window
       ↓
TreeSHAP
       ↓
Feature importance ranking
       ↓
Kendall tau comparison
       ↓
Drift flag
```

## 6.4 Reproduce original ablation

```text
A0  Baseline static
A1  +Graph
A2  +SHAP Drift
A3  All Three
A4  Baseline adaptive
```

Expected reference results from Saurabh's paper:

| Configuration | F1 | Precision | Recall |
|---|---:|---:|---:|
| Baseline static | 0.9573 | 0.9448 | 0.9701 |
| +Graph | 0.9653 | 0.9599 | 0.9707 |
| +SHAP Drift | 0.9625 | 0.9520 | 0.9731 |
| All Three | 0.9648 | 0.9587 | 0.9711 |
| Baseline adaptive | **0.9730** | 0.9615 | **0.9848** |

These numbers should be treated as targets to reproduce, not numbers to hard-code.

## 6.5 Important findings to verify

The reproduction should also check:

- Graph reduced false positives from 302 to 216 in the reported experiment.
- Adaptive threshold produced FPR outliers up to 0.34 in borderline windows.
- SHAP drift uses a 1000-sample window.
- The combined All-Three configuration was slightly below Graph alone in F1, suggesting information overlap.
- SHAP drift's major value is interpretability / attack-transition detection rather than necessarily improving F1.

## 6.6 Data4Cyber

Only after NCSRD reproduction is stable:

```text
Data4Cyber
    ↓
dataset-specific adaptation
    ↓
Saurabh methodology
    ↓
evaluation
```

Document every change required because the datasets do not have the same feature structure.

---

# 7. PHASE 4 — Implement Our Proposed Work

**Owner: Member 3**

This is the main research contribution.

The faculty has specified five directions:

1. Online correlation baseline drift adaptation
2. Precision-constrained threshold optimization
3. Real-time SHAP approximation
4. Model distillation for edge deployment
5. Evaluation on non-stationary traffic

---

## 7.1 Online Correlation Baseline

Current Saurabh approach:

```text
Static benign correlation matrix
```

Our approach:

```text
Cbase(t-1)
     ↓
EWMA update
     ↓
Cbase(t)
```

Add change-point detection, such as CUSUM, so the baseline updates only when benign traffic is confirmed.

Main problem to solve:

> Prevent attack traffic from contaminating the benign baseline.

---

## 7.2 Precision-Constrained Threshold

Current Saurabh method:

```text
maximize F1
```

Our target:

```text
maximize Recall
subject to FPR ≤ α
```

Start with:

```text
α = 0.05
```

The point is to prevent the high FPR spikes observed with small-window F1 optimization.

---

## 7.3 Lightweight SHAP

Compare:

```text
TreeExplainer
1000-sample window
```

against:

```text
FastSHAP / LinearSHAP
50–100 sample window
incremental Kendall tau
```

Measure:

- SHAP computation latency
- total drift detection latency
- number of detected drift events
- false drift rate
- sensitivity to attack transitions

---

## 7.4 Model Distillation

Teacher:

```text
XGBoost
51-feature augmented input
```

Student:

```text
Lightweight shallow neural network
```

Measure:

- F1
- precision
- recall
- FPR
- AUC
- model size
- memory
- inference latency
- CPU usage
- power where measurable

---

## 7.5 Proposed module outputs

Suggested structure:

```text
src/proposed/
│
├── ewma.py
├── cusum.py
├── constrained_threshold.py
├── fast_shap.py
└── distillation.py
```

The proposed implementation should expose the same model/prediction interface as the other methods.

---

# 8. PHASE 5 — Build the Unified Experiment Framework

**Owner: Member 5**

This person creates the integration layer.

We should eventually be able to run:

```bash
python run_experiment.py --dataset ncsrd --method base
python run_experiment.py --dataset ncsrd --method saurabh
python run_experiment.py --dataset ncsrd --method proposed
```

and:

```bash
python run_experiment.py --dataset data4cyber --method base
python run_experiment.py --dataset data4cyber --method saurabh
python run_experiment.py --dataset data4cyber --method proposed
```

## Common evaluator

Every run should produce a standardized record containing:

```text
dataset
method
configuration
precision
recall
F1
FPR
AUC
TP
FP
FN
TN
threshold
threshold variance
latency
model size
memory
power
```

This lets us compare the three approaches without manually combining unrelated notebooks.

---

# 9. PHASE 6 — 3×1 NCSRD Comparison

After all three methods run independently:

```text
                        NCSRD
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
          BASE         SAURABH       PROPOSED
            |             |             |
            +-------------+-------------+
                          |
                          v
                     COMPARISON
```

## Detection metrics

- Accuracy
- Precision
- Recall
- F1
- FPR
- AUC

## Stability metrics

- Mean FPR
- FPR variance
- Maximum FPR
- Threshold variance

## Real-time metrics

- preprocessing latency
- feature-generation latency
- model inference latency
- SHAP/drift latency
- end-to-end detection latency

## Edge metrics

- model size
- memory
- CPU
- power where measurable

---

# 10. PHASE 7 — Data4Cyber Adaptation

This phase begins **only after the NCSRD pipeline is stable**.

Final target:

```text
                         Data4Cyber
                             |
               +-------------+-------------+
               |             |             |
               v             v             v
             BASE         SAURABH       PROPOSED
               |             |             |
               +-------------+-------------+
                             |
                             v
                        COMPARISON
```

Document clearly:

### What transfers directly

- thresholding concept
- EWMA
- CUSUM
- SHAP drift concept
- evaluation framework
- distillation approach

### What requires adaptation

- feature schema
- labels
- preprocessing
- window construction
- graph dimension
- model input dimension

Do not claim that the algorithms were applied "unchanged" if dataset-specific modifications were necessary.

---

# 11. PHASE 8 — Final 3×2 Comparison

Final comparison matrix:

| Method | NCSRD | Data4Cyber |
|---|---|---|
| **Base** | Result | Result |
| **Saurabh** | Result | Result |
| **Proposed** | Result | Result |

This should answer:

### Q1
Does our method improve over the base paper?

### Q2
Does it improve over Saurabh?

### Q3
Does the improvement remain useful on another dataset?

### Q4
Does the proposed method improve operational metrics such as FPR stability and latency rather than only F1?

---

# 12. PHASE 9 — Proposed Ablation Study

**Lead: Member 5**

**Algorithm support: Member 3**

Do not only compare:

```text
Base
Saurabh
Proposed
```

Break our proposed system into components.

Suggested ablation:

| ID | Configuration |
|---|---|
| P0 | Saurabh baseline |
| P1 | + EWMA |
| P2 | + Constrained Threshold |
| P3 | + Lightweight SHAP |
| P4 | + EWMA + Constrained Threshold |
| P5 | + EWMA + Threshold + Lightweight SHAP |
| P6 | Full Proposed |
| P7 | Full Proposed + Distilled Edge Model |

The exact set can be reduced if some configurations become redundant, but the study should be designed before the final experiments.

The key question is:

> **Which component improves which property?**

For example:

```text
EWMA            → adaptation / FPR stability
Threshold       → FPR constraint / recall trade-off
FastSHAP        → latency
Distillation    → edge efficiency
```

---

# 13. PHASE 10 — Non-Stationary Traffic Experiments

**Owner: Member 4**

The faculty specifically wants synthetic normal-traffic variations.

Create controlled scenarios such as:

```text
Scenario 1
Stable benign traffic

Scenario 2
Benign traffic pattern changes

Scenario 3
Bursty mMTC-like traffic

Scenario 4
Periodic URLLC-like traffic

Scenario 5
Gradual attack onset

Scenario 6
Attack → benign transition

Scenario 7
Attack-type transition
```

Measure:

- FPR mean
- FPR variance
- maximum FPR
- detection latency
- drift detection latency
- number of drift alerts

The goal is to demonstrate that the proposed system adapts to changing benign traffic without letting attacks corrupt the baseline.

---

# 14. PHASE 11 — Final Edge Benchmark

Compare:

```text
Cloud/reference XGBoost
           vs
Distilled edge model
```

Final table:

| Metric | XGBoost | Distilled Model |
|---|---:|---:|
| Model size | | |
| Memory | | |
| Inference latency | | |
| End-to-end latency | | |
| F1 | | |
| Precision | | |
| Recall | | |
| FPR | | |
| Power | | |

The final project should aim to demonstrate a better overall operational trade-off:

> **Detection performance + low FPR + adaptation + low latency + edge efficiency**

rather than simply claiming that F1 is higher.

---

# 15. Recommended Work Order

The whole group should follow this order:

```text
PHASE 0
GitHub + environment + interfaces
        ↓
PHASE 1
Common dataset pipeline
        ↓
PHASE 2
Base paper reproduction
        ↓
PHASE 3
Saurabh reproduction
        ↓
PHASE 4
Proposed method implementation
        ↓
PHASE 5
Unified experiment runner
        ↓
PHASE 6
3×1 NCSRD comparison
        ↓
PHASE 7
Data4Cyber adaptation
        ↓
PHASE 8
3×2 final comparison
        ↓
PHASE 9
Proposed ablation
        ↓
PHASE 10
Non-stationary traffic experiments
        ↓
PHASE 11
Edge + latency benchmarking
        ↓
PHASE 12
Final plots + tables + report + Review 2 demo
```

---

# 16. Division of Work Among 5 Members

## MEMBER 1 — BASE PAPER / ORIGINAL SYSTEM

### Owns

**NCSRD**
- base paper preprocessing
- base XGBoost
- static threshold
- evaluation
- reproduction of reference results
- optional CNN/LSTM/MLP

**Data4Cyber**
- base preprocessing adapter
- base XGBoost
- evaluation

### Deliverables

```text
src/base/
base_ncsrd_results
base_data4cyber_results
base plots
```

### Boundary

Own only the **original/base approach**.

---

## MEMBER 2 — SAURABH / EXISTING EXTENSION

### Owns

**NCSRD**
- run supplied notebooks
- reproduce:
  - correlation graph
  - dynamic threshold
  - SHAP drift
  - integrated system
  - original five-configuration ablation

**Data4Cyber**
- adapt Saurabh methodology
- evaluate
- document modifications

### Deliverables

```text
src/saurabh/
saurabh_ncsrd_results
saurabh_data4cyber_results
original ablation results
```

### Boundary

Reproduce Saurabh's work faithfully.

Do not "improve" his implementation while reproducing it.

---

## MEMBER 3 — OUR PROPOSED RESEARCH

### Owns

- EWMA
- CUSUM/change-point detection
- FPR-constrained threshold
- FastSHAP / LinearSHAP
- reduced SHAP window
- Knowledge Distillation
- edge version

### Deliverables

```text
src/proposed/

ewma.py
cusum.py
constrained_threshold.py
fast_shap.py
distillation.py
```

plus:

```text
proposed_ncsrd_results
proposed_data4cyber_results
```

### Boundary

Own the **algorithmic contribution**, not the final master tables/plots.

---

## MEMBER 4 — DATA + DATA4CYBER + TRAFFIC GENERATION

### Owns

- NCSRD cleaning/preprocessing
- common data loader
- labels
- train/test split
- windowing
- Data4Cyber adapter
- synthetic non-stationary traffic
- dataset documentation

### Deliverables

```text
src/common/data/
src/common/windows/
src/data4cyber/
```

plus standardized dataset interfaces.

### Boundary

Do not independently alter ML algorithms.

---

## MEMBER 5 — INTEGRATION + EXPERIMENTS + ABLATION

### Owns

- common experiment runner
- common evaluator
- metric calculations
- experiment configuration
- comparison tables
- plots
- master ablation study
- latency benchmarking
- FPR variance analysis
- final result consolidation

### Deliverables

```text
run_experiment.py
evaluate.py
results/
plots/
tables/
```

### Boundary

Do not become a dumping ground for everyone else's code.

Every module should integrate through agreed interfaces.

---

# 17. GitHub Coordination Rules

## Repo ownership

One shared repository.

Each member should mainly own one area:

```text
M1 → src/base/
M2 → src/saurabh/
M3 → src/proposed/
M4 → src/common/ + src/data4cyber/
M5 → evaluation + experiments + results
```

## Never do this

```text
five independent notebooks
five different preprocessing pipelines
five different splits
five different metric implementations
```

## Always do this

```text
ONE dataset interface
ONE split
ONE windowing convention
ONE metric implementation
THREE interchangeable methods
```

---

# 18. Midpoint Milestones

By the midpoint, these five milestones should exist:

| Member | Required milestone |
|---|---|
| **M1** | Base XGBoost running on NCSRD |
| **M2** | Complete Saurabh pipeline reproduced on NCSRD |
| **M3** | At least one proposed improvement running on NCSRD |
| **M4** | NCSRD + Data4Cyber pipelines working |
| **M5** | One-command Base vs Saurabh comparison working |

Once these are achieved, the project is structurally safe.

---

# 19. Recommended Pairing

The two most tightly connected roles should be:

### Member 3 + Member 5

**Member 3 asks:**

> What did we change and how does the algorithm work?

**Member 5 asks:**

> Did the change actually improve anything, and can we prove it?

This pairing is especially important for the final research contribution.

---

# 20. One Important Data4Cyber Decision

**By: Member-4**

Before investing heavily in Data4Cyber implementation, confirm the exact faculty requirement.

There are two fundamentally different possibilities:

### A. Independent evaluation (Recommended)

Train each approach on Data4Cyber and evaluate on Data4Cyber.

This demonstrates performance in a second environment.

### B. Cross-dataset transfer

Train on NCSRD and test on Data4Cyber, or vice versa.

This is a much stronger generalization experiment and requires substantially more care because the datasets have different schemas and semantics.

Do not assume these are the same experiment.

The project architecture should support both, but the exact experiment should be confirmed before implementation.

**[updated] — DECIDED: Option A, independent evaluation on each dataset.**

Each approach is trained and tested within a dataset; results are compared
across the 3 × 2 matrix. Implemented as the Data4Cyber `block` split (D3).

Cross-dataset transfer (Option B) is **not** part of the project. Note also
that Data4Cyber is a smart-grid/ICS testbed, so a second-dataset result shows
that the *methodology* survives a domain change after adaptation — it is not
evidence of 5G generalization (D7).

---

# 21. Final Research Story

The final project should tell one coherent story:

```text
BASE PAPER
High DDoS detection accuracy
        |
        | limitations:
        | independent features
        | static threshold
        | passive SHAP
        v
SAURABH
Correlation graph
Dynamic threshold
SHAP drift detection
        |
        | remaining limitations:
        | FPR spikes
        | static correlation baseline
        | expensive/slow SHAP
        | edge deployment gap
        v
OUR METHOD
EWMA + CUSUM
FPR-constrained threshold
Lightweight SHAP
Knowledge Distillation
        |
        v
3×2 EVALUATION
NCSRD + Data4Cyber
        |
        v
ABLATION
Which component actually helps?
        |
        v
FINAL CLAIM
Better balance of
accuracy + FPR stability
+ adaptability + latency
+ edge efficiency
```

That should be the backbone of the implementation, final report and Review 2 demonstration.
