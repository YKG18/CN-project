# Table 8.1 — Incremental Ablation Results (multi-seed)

Mean ± std over seeds. Contamination-injection columns are owned by Member 2 and are not filled here.

| ID | Configuration | F1 | FPR | Precision | Recall | Latency (ms) | n |
|---|---|---|---|---|---|---|---|
| **P0** | Base XGBoost (no EWMA) | 0.8424 ± 0.0192 | 0.0192 ± 0.0034 | 0.7658 ± 0.0335 | 0.9370 ± 0.0047 | 0.0010 ± 0.0001 | 20 |
| **P1** | + EWMA baseline tracking | 0.9692 ± 0.0161 | 0.0033 ± 0.0023 | 0.9536 ± 0.0311 | 0.9860 ± 0.0097 | 0.0023 ± 0.0002 | 20 |
| **P2** | + CUSUM contamination guard | 0.9692 ± 0.0161 | 0.0033 ± 0.0023 | 0.9536 ± 0.0311 | 0.9860 ± 0.0097 | 0.0024 ± 0.0004 | 20 |
| **P3** | + FPR-constrained threshold | 0.9634 ± 0.0472 | 0.0015 ± 0.0011 | 0.9773 ± 0.0152 | 0.9549 ± 0.0858 | 0.0023 ± 0.0002 | 20 |
| **P4** | + FastSHAP drift detector | 0.9634 ± 0.0472 | 0.0015 ± 0.0011 | 0.9773 ± 0.0152 | 0.9549 ± 0.0858 | 0.0025 ± 0.0005 | 20 |
| **P5** | + Knowledge distillation | 0.9634 ± 0.0472 | 0.0015 ± 0.0011 | 0.9773 ± 0.0152 | 0.9549 ± 0.0858 | 0.0017 ± 0.0004 | 20 |
| **P6** | Full proposed teacher model | 0.9634 ± 0.0472 | 0.0015 ± 0.0011 | 0.9773 ± 0.0152 | 0.9549 ± 0.0858 | 0.0021 ± 0.0003 | 20 |
| **P7** | Edge student alone | 0.8632 ± 0.0701 | 0.0080 ± 0.0028 | 0.8807 ± 0.0330 | 0.8587 ± 0.1159 | 0.0018 ± 0.0003 | 20 |
