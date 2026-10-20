# Wilcoxon signed-rank tests (Proposed vs baselines)

Pairing is by seed on the frozen project-standard split. Bonferroni correction is applied within each dataset over 14 planned tests (7 metrics × 2 baselines).

| Dataset | Metric | Baseline | n | mean_diff | p | p_adj | marker | note |
|---|---|---|---|---|---|---|---|---|
| ncsrd | f1 | Base Paper | 20 | 0.142045 | 2.861e-06 | 4.005e-05 | *** |  |
| ncsrd | precision | Base Paper | 20 | 0.236258 | 9.537e-07 | 1.335e-05 | *** |  |
| ncsrd | recall | Base Paper | 20 | 0.0247949 | 0.01074 | 0.1504 | ns |  |
| ncsrd | fpr | Base Paper | 20 | -0.0209456 | 9.537e-07 | 1.335e-05 | *** |  |
| ncsrd | accuracy | Base Paper | 20 | 0.0211853 | 9.537e-07 | 1.335e-05 | *** |  |
| ncsrd | roc_auc | Base Paper | 20 | 0.0051088 | 9.537e-07 | 1.335e-05 | *** |  |
| ncsrd | inference_latency_ms | Base Paper | 20 | 0.00130255 | 1 | 1 | ns |  |
| ncsrd | f1 | Saurabh Extension | 20 | -0.0057995 | 0.1745 | 1 | ns |  |
| ncsrd | precision | Saurabh Extension | 20 | 0.0236741 | 9.805e-05 | 0.001373 | ** |  |
| ncsrd | recall | Saurabh Extension | 20 | -0.0311195 | 0.9995 | 1 | ns |  |
| ncsrd | fpr | Saurabh Extension | 20 | -0.00172485 | 9.805e-05 | 0.001373 | ** |  |
| ncsrd | accuracy | Saurabh Extension | 20 | -0.00032065 | 0.1636 | 1 | ns |  |
| ncsrd | roc_auc | Saurabh Extension | 20 | 0 | 1 | 1 | ns | all paired differences are zero |
| ncsrd | inference_latency_ms | Saurabh Extension | 20 | -0.0024038 | 9.537e-07 | 1.335e-05 | *** |  |
| data4cyber | f1 | Base Paper | 20 | 0.00094395 | 0.2375 | 1 | ns |  |
| data4cyber | precision | Base Paper | 20 | 0.0088077 | 1.812e-05 | 0.0002537 | *** |  |
| data4cyber | recall | Base Paper | 20 | -0.0083332 | 0.9988 | 1 | ns |  |
| data4cyber | fpr | Base Paper | 20 | -0.0154563 | 0.0001016 | 0.001423 | ** |  |
| data4cyber | accuracy | Base Paper | 20 | 0.00231595 | 0.04126 | 0.5776 | ns |  |
| data4cyber | roc_auc | Base Paper | 20 | 0.00494865 | 0.000845 | 0.01183 | * |  |
| data4cyber | inference_latency_ms | Base Paper | 20 | 0.0111286 | 1 | 1 | ns |  |
| data4cyber | f1 | Saurabh Extension | 20 | -0.00020365 | 0.8962 | 1 | ns |  |
| data4cyber | precision | Saurabh Extension | 20 | 0.00114165 | 0.005859 | 0.08203 | ns |  |
| data4cyber | recall | Saurabh Extension | 20 | -0.0017447 | 0.9942 | 1 | ns |  |
| data4cyber | fpr | Saurabh Extension | 20 | -0.00208865 | 0.005808 | 0.08131 | ns |  |
| data4cyber | accuracy | Saurabh Extension | 20 | -2.87e-05 | 0.3766 | 1 | ns |  |
| data4cyber | roc_auc | Saurabh Extension | 20 | 0 | 1 | 1 | ns | all paired differences are zero |
| data4cyber | inference_latency_ms | Saurabh Extension | 20 | -0.0212266 | 9.537e-07 | 1.335e-05 | *** |  |
