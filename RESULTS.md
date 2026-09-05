# Results

Consolidated 3 × 2 comparison. Owner: M5. Raw per-run CSVs go in `results/raw/`
using the schema in [docs/PROJECT_DECISIONS.md](docs/PROJECT_DECISIONS.md) D6.

All metrics below are for the **attack class (label = 1)** unless stated.

## Reference targets (reproduce, do not hard-code)

| Source | Configuration | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| Base paper | XGBoost, 38 features | 0.97 | 0.96 | 0.98 |
| Saurabh | Baseline static | 0.9573 | 0.9448 | 0.9701 |
| Saurabh | + Graph | 0.9653 | 0.9599 | 0.9707 |
| Saurabh | + SHAP drift | 0.9625 | 0.9520 | 0.9731 |
| Saurabh | All three | 0.9648 | 0.9587 | 0.9711 |
| Saurabh | Baseline adaptive | 0.9730 | 0.9615 | 0.9848 |

Base paper also reports accuracy 0.996 and weighted F1 1.00.

## A. Reference reproduction — NCSRD

Random stratified 80/20, seed 42, global scaling. Compare against the table
above. **Optimistic; not the headline.**

| Method | Config | Feature set | P | R | F1 | FPR | AUC | τ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| | | | | | | | | |

## B. Project standard — NCSRD

Temporal 70/10/20, train-only scaling, one frozen split, τ chosen on
validation. **This is the headline 3 × 1 comparison.**

| Method | Config | Feature set | P | R | F1 | FPR | AUC | τ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| | | | | | | | | |

## C. Project standard — Data4Cyber (`block` split, primary)

| Method | Config | P | R | F1 | FPR | AUC | τ |
|---|---|---:|---:|---:|---:|---:|---:|
| | | | | | | | |

## D. Secondary — Data4Cyber (`scenario` split, unseen attack family)

Robustness only. S6 (MQTT supply-chain) never appears in training.

| Method | Config | P | R | F1 | FPR | AUC | τ |
|---|---|---:|---:|---:|---:|---:|---:|
| | | | | | | | |

## E. Stability, latency and edge metrics

Window-level (D4): FPR variance, max FPR, threshold variance, detection and
drift latency, drift-event counts. Plus model size, memory and inference
latency for the distillation comparison.

| Method | Mean FPR | FPR var | Max FPR | τ var | Latency (ms) | Model size |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |
