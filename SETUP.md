# Setup

Environment, dataset download links and the build commands are in
[README.md](README.md). This file is the frozen configuration everyone shares —
do not change any of it alone (Implementation.md §3.3).

| | | Where |
|---|---|---|
| Python | 3.10 – 3.12 | — |
| Packages | `requirements.txt` (root, the only one) | — |
| Random seed | `42` | `config.SEED` |
| Reference split | random stratified 80/20, global scaling | `config.REFERENCE_SPLIT` |
| Project-standard split | temporal 70/10/20, train-only scaling | `config.PROJECT_STANDARD_SPLIT` |
| Frozen split indices | shared by all three methods | `config.SPLIT_INDEX_FILE` |
| NCSRD feature sets | `base38` (base paper) / `saurabh49` (Saurabh) | `ncsrd_prep.FEATURE_SETS` |
| Data4Cyber split | `block` primary, `scenario` secondary | `data4cyber_prep.SPLIT_MODES` |
| Threshold | chosen on validation, reported on test | PROJECT_DECISIONS.md D5 |
| Reported class | attack (label = 1) | PROJECT_DECISIONS.md D6 |
| Windows | 500 (threshold/correlation), 1000 (SHAP drift) | `config.THRESHOLD_WINDOW`, `config.SHAP_WINDOW` |

Paths all resolve from the repository root via `src/common/config.py`. Never
hard-code an absolute path.

After changing anything under `src/common/`, run:

```bash
python tests/test_pipeline.py
```
