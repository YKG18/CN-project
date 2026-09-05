# Implementation Roadmap

Who does what, in what order. This is a **plan, not a specification** — change
it when a simpler or better approach appears.

Methodology (feature sets, splits, thresholds, metrics) is **not** here. That
lives in [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md), which wins wherever the
two disagree. Setup and repository layout are in the root `README.md`.

---

## The story we are telling

```
BASE PAPER            Xylouris et al. — XGBoost, static threshold
   |                  limits: features treated independently, static τ,
   |                          SHAP used only for offline auditing
   v
SAURABH               correlation behavioural graph, per-window F1-optimal τ,
   |                  SHAP drift via Kendall tau
   |                  limits: FPR spikes (up to 0.34), static correlation
   |                          baseline, 1000-sample SHAP window too slow
   v
OUR PROPOSED WORK     EWMA + CUSUM baseline adaptation, FPR-constrained τ,
                      lightweight SHAP, distillation for edge deployment
   v
3 × 2 EVALUATION + ABLATION
```

The final claim is a better **operational trade-off** — accuracy *plus* FPR
stability *plus* adaptability *plus* latency *plus* edge efficiency — not simply
a higher F1.

---

## Phase order and status

| Phase | What | Owner | Status |
|---|---|---|---|
| 0 | Repo, environment, shared interfaces | M4 | **done** |
| 1 | Common data pipelines (NCSRD + Data4Cyber) | M4 | **done** |
| 2 | Base paper reproduction, NCSRD | M1 | **next** |
| 3 | Saurabh reproduction, NCSRD | M2 | after 2 |
| 4 | Proposed method | M3 | after 3 |
| 5 | Unified runner + common evaluator | M5 | alongside 3–4 |
| 6 | 3 × 1 NCSRD comparison | M5 | after 4 |
| 7 | Data4Cyber adaptation, all three methods | all | after 6 |
| 8 | Final 3 × 2 comparison | M5 | after 7 |
| 9 | Proposed ablation | M5 + M3 | after 8 |
| 10 | Non-stationary traffic experiments | M4 | after 4 |
| 11 | Edge / latency benchmark | M3 + M5 | after 4 |
| 12 | Plots, tables, report, demo | all | last |

Phases 2, 3 and 4 can overlap once each member has the frozen split. Nothing
downstream of phase 4 should start before the 3 × 1 NCSRD comparison looks sane.

---

## Member 1 — Base paper

**Owns `src/base/`.** Two configurations, kept strictly separate (D1):

* `base_paper` — the `base38` feature set, random undersampling +
  `scale_pos_weight`, logloss, early stopping on validation.
* `base_saurabh_static` — the `saurabh49` feature set, SMOTE on train,
  `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8`. This is the anchor for Saurabh's ablation row A0.

Each runs under both split policies (D2): `reference` for comparison with
published numbers, `project_standard` for the headline tables.

Deliverables: `results/raw/base_ncsrd_results.csv` in the D6 schema, plus
confusion matrix, ROC curve, PR curve and inference timing. Then the same on
Data4Cyber once NCSRD is stable.

CNN / LSTM / MLP are optional and not on the critical path (D8).

---

## Member 2 — Saurabh's extension

**Owns `src/saurabh/`.** Reproduce faithfully; do not "improve" while
reproducing. Start from the supplied notebooks in
`docs/Saurabh's Project Files/`, run them as given, then port to `src/saurabh/`
against the shared pipeline.

**Module A — correlation behavioural graph.** Pearson correlation over the 49
features → benign baseline matrix → per-window correlation → Frobenius
divergence → `graph_frob_div`, appended as a 50th feature to XGBoost.

**Module B — dynamic threshold.** Per 500-sample window, F1-optimal τ from the
precision–recall curve. Fallback for all-benign / all-attack windows.

**Module C — SHAP drift.** TreeSHAP over 1000-sample windows → feature
importance ranking → Kendall tau between consecutive windows → drift flag.

**Ablation to reproduce** (targets in PROJECT_DECISIONS.md):

| ID | Configuration |
|---|---|
| A0 | Baseline static |
| A1 | + Graph |
| A2 | + SHAP drift |
| A3 | All three |
| A4 | Baseline adaptive |

Worth verifying: the graph cutting false positives 302 → 216; adaptive
thresholding producing FPR outliers up to 0.34; All-Three scoring slightly
*below* Graph alone on F1, which suggests the signals overlap; and SHAP drift
being valuable for interpretability and attack-transition detection rather than
for F1.

---

## Member 3 — Our proposed work

**Owns `src/proposed/`.** Five directions from the faculty brief:

| File | What |
|---|---|
| `ewma.py` | Replace Saurabh's static benign correlation matrix with an EWMA update |
| `cusum.py` | Change-point detection on the Frobenius divergence, so the baseline updates only when benign traffic is confirmed — this is what stops attack traffic contaminating the baseline |
| `constrained_threshold.py` | Maximize recall subject to FPR ≤ α (start α = 0.05) instead of maximizing F1. Directly targets Saurabh's FPR spikes |
| `fast_shap.py` | FastSHAP / LinearSHAP, 50–100 sample window, incremental Kendall tau. Measure the sensitivity-versus-latency trade-off |
| `distillation.py` | Distil the augmented XGBoost into a small shallow network; benchmark size, memory, latency, CPU, power |

Expose the same `fit()` / `predict_proba()` / `predict()` interface as everyone
else.

---

## Member 4 — Data and traffic generation

**Owns `src/common/`.** Pipelines are done. Remaining: **phase 10**, synthetic
non-stationary benign traffic to challenge the online baseline updater:

1. stable benign 2. benign pattern shift 3. bursty mMTC-like
4. periodic URLLC-like 5. gradual attack onset 6. attack → benign transition
7. attack-type transition

Measure FPR mean / variance / maximum, detection latency, drift latency and
drift-alert counts across 1000+ sliding windows. The point is to show the
system adapts to changing benign traffic *without* letting attacks corrupt the
baseline.

Do not change ML algorithms — that is M1/M2/M3 territory.

---

## Member 5 — Integration, experiments, ablation

**Owns `run_experiment.py`, the evaluator, and `results/`.** Target:

```bash
python run_experiment.py --dataset ncsrd --method base
python run_experiment.py --dataset data4cyber --method proposed
```

One evaluator implementing the D6 metric definitions — nobody else computes
metrics by hand. Every run emits the D6 results row.

**Proposed ablation** (design it before running the final experiments):

| ID | Configuration |
|---|---|
| P0 | Saurabh baseline |
| P1 | + EWMA |
| P2 | + Constrained threshold |
| P3 | + Lightweight SHAP |
| P4 | + EWMA + Constrained threshold |
| P5 | + EWMA + Threshold + Lightweight SHAP |
| P6 | Full proposed |
| P7 | Full proposed + distilled edge model |

The question is **which component improves which property** — roughly, EWMA →
adaptation and FPR stability; constrained τ → FPR bound versus recall;
FastSHAP → latency; distillation → edge efficiency. Drop redundant rows if some
turn out to say the same thing.

Beyond detection metrics, report stability (mean / variance / max FPR, τ
variance), real-time latency (preprocessing, feature generation, inference,
drift detection, end-to-end) and edge cost (model size, memory, CPU, power
where measurable).

M5 is an integrator, not a dumping ground — everything arrives through the
agreed interfaces.

---

## Midpoint milestones

| Member | Milestone |
|---|---|
| M1 | Base XGBoost running on NCSRD, both configurations |
| M2 | Full Saurabh pipeline reproduced on NCSRD |
| M3 | At least one proposed improvement running on NCSRD |
| M4 | Both pipelines working *(done)* + non-stationary scenarios |
| M5 | One-command Base vs Saurabh comparison |

Once these exist, the project is structurally safe.

**M3 and M5 should pair closely.** M3 asks "what did we change and how does it
work?"; M5 asks "did it actually improve anything, and can we prove it?" That
pairing is where the research contribution gets validated.

---

## Questions the final report must answer

1. Does our method improve on the base paper?
2. Does it improve on Saurabh?
3. Does the improvement hold on a second, different-domain dataset (D7)?
4. Does it improve *operational* metrics — FPR stability, latency, edge cost —
   and not only F1?
