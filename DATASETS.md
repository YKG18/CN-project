# Benchmark datasets

The raw benchmark data are intentionally not included in the repository ZIP or Git history. Download them from their official Zenodo records into the expected project paths:

- **NCSRD-DS-5GDDoS v3.0** (used file: `amari_ue_data_classic_tabular.csv`): https://zenodo.org/records/13900057 — download into `data/ncsrd/raw/`.
- **Data4Cyber** (project-pinned release): https://zenodo.org/records/19965384 — the project's pipeline needs each scenario folder's `dataset.csv` under `data/data4cyber/raw/<scenario>/`.

From the repository root, run:

```bash
python scripts/download_datasets.py
```

Optional: `--ncsrd-only`, `--data4cyber-only`, or `--force`.

The script downloads from Zenodo and extracts only the Data4Cyber scenario `dataset.csv` files needed by this repository, not the large ancillary packet captures. It requires an internet connection. If the Zenodo file structure changes, download the files manually from the record pages and place them at the paths above.

**Important:** raw datasets are excluded from version control as specified in `docs/PROJECT_DECISIONS.md`. Do not commit the downloaded files to Git. Cite the dataset records in any report or publication and check the applicable terms before redistributing data.
