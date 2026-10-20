# Table 7.3 — Detailed Data4Cyber Comparison (multi-seed)

Values are mean ± std with 95% t-interval in brackets. Markers on the Proposed column are Bonferroni-adjusted Wilcoxon tests vs Saurabh (`*` p_adj < 0.05, `**` < 0.01, `***` < 0.001, `ns` not significant).

| Metric | Base Paper | Saurabh Extension | Proposed Work | vs Base | vs Saurabh |
|---|---|---|---|---|---|
| F1 (attack class) | 0.8910 ± 0.0043 [0.8890, 0.8930] | 0.8921 ± 0.0020 [0.8912, 0.8931] | 0.8919 ± 0.0021 [0.8910, 0.8929] | ns (p_adj=1) | ns (p_adj=1) |
| Precision | 0.8530 ± 0.0045 [0.8509, 0.8551] | 0.8606 ± 0.0035 [0.8590, 0.8623] | 0.8618 ± 0.0042 [0.8598, 0.8638] | *** (p_adj=0.0002537) | ns (p_adj=0.08203) |
| Recall | 0.9327 ± 0.0096 [0.9282, 0.9372] | 0.9261 ± 0.0029 [0.9247, 0.9274] | 0.9243 ± 0.0037 [0.9226, 0.9261] | ns (p_adj=1) | ns (p_adj=1) |
| FPR | 0.1984 ± 0.0081 [0.1947, 0.2022] | 0.1851 ± 0.0057 [0.1824, 0.1877] | 0.1830 ± 0.0068 [0.1798, 0.1862] | ** (p_adj=0.001423) | ns (p_adj=0.08131) |
| Accuracy | 0.8740 ± 0.0045 [0.8719, 0.8761] | 0.8763 ± 0.0024 [0.8752, 0.8775] | 0.8763 ± 0.0026 [0.8751, 0.8775] | ns (p_adj=0.5776) | ns (p_adj=1) |
| ROC-AUC | 0.8997 ± 0.0050 [0.8974, 0.9020] | 0.9046 ± 0.0021 [0.9037, 0.9056] | 0.9046 ± 0.0021 [0.9037, 0.9056] | * (p_adj=0.01183) | ns (p_adj=1) |
| Inference latency (ms/sample) | 0.0017 ± 0.0004 [0.0015, 0.0019] | 0.0341 ± 0.0053 [0.0316, 0.0365] | 0.0129 ± 0.0021 [0.0119, 0.0139] | ns (p_adj=1) | *** (p_adj=1.335e-05) |
