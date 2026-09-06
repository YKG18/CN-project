"""Table Generator Script — Member 5.

Reads raw result CSVs from results/raw/ and generates formatted Markdown tables
for the 3x2 evaluation matrix and the proposed ablation study.

Usage:
    python results/generate_tables.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
TABLES_DIR = ROOT / "results" / "tables"

D6_COLUMNS = [
    "dataset", "method", "config", "feature_set", "split_policy", "seed",
    "precision", "recall", "f1", "fpr", "accuracy", "weighted_f1", "roc_auc",
    "tp", "fp", "fn", "tn", "threshold", "threshold_selected_on",
    "n_train", "n_test", "inference_latency_ms", "model_size_kb", "notes"
]


def load_all_results() -> List[Dict[str, Any]]:
    """Load all rows from CSV files in results/raw/."""
    results = []
    if not RAW_DIR.exists():
        return results

    for csv_file in RAW_DIR.glob("*.csv"):
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    return results


def generate_3x2_matrix(results: List[Dict[str, Any]]) -> str:
    """Generate the 3x2 Method x Dataset evaluation matrix in Markdown."""
    header = "| Method | NCSRD-DS-5GDDoS (F1) | NCSRD (FPR) | Data4Cyber (F1) | Data4Cyber (FPR) |\n"
    header += "|---|---|---|---|---|\n"

    methods = ["base", "saurabh", "proposed"]
    rows = []

    for m in methods:
        ncsrd_row = next((r for r in results if r.get("method") == m and r.get("dataset") == "ncsrd"), None)
        d4c_row = next((r for r in results if r.get("method") == m and r.get("dataset") == "data4cyber"), None)

        ncsrd_f1 = f"{float(ncsrd_row['f1']):.4f}" if ncsrd_row else "TBD"
        ncsrd_fpr = f"{float(ncsrd_row['fpr']):.4f}" if ncsrd_row else "TBD"

        d4c_f1 = f"{float(d4c_row['f1']):.4f}" if d4c_row else "TBD"
        d4c_fpr = f"{float(d4c_row['fpr']):.4f}" if d4c_row else "TBD"

        rows.append(f"| **{m.capitalize()}** | {ncsrd_f1} | {ncsrd_fpr} | {d4c_f1} | {d4c_fpr} |")

    return header + "\n".join(rows) + "\n"


def generate_ablation_table(results: List[Dict[str, Any]]) -> str:
    """Generate the proposed ablation study matrix (P0 - P7) in Markdown."""
    header = "| ID | Configuration | F1 | FPR | Precision | Recall | Latency (ms) | Model Size (KB) |\n"
    header += "|---|---|---|---|---|---|---|---|\n"

    ablation_ids = [f"P{i}" for i in range(8)]
    rows = []

    for p_id in ablation_ids:
        row_data = next((r for r in results if r.get("config") == p_id), None)
        if row_data:
            f1 = f"{float(row_data['f1']):.4f}"
            fpr = f"{float(row_data['fpr']):.4f}"
            prec = f"{float(row_data['precision']):.4f}"
            rec = f"{float(row_data['recall']):.4f}"
            lat = f"{float(row_data['inference_latency_ms']):.4f}"
            size = f"{float(row_data['model_size_kb']):.1f}"
        else:
            f1 = fpr = prec = rec = lat = size = "TBD"

        config_name = f"Ablation {p_id}"
        rows.append(f"| **{p_id}** | {config_name} | {f1} | {fpr} | {prec} | {rec} | {lat} | {size} |")

    return header + "\n".join(rows) + "\n"


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_all_results()
    print(f"Loaded {len(results)} result rows from {RAW_DIR}")

    matrix_md = generate_3x2_matrix(results)
    ablation_md = generate_ablation_table(results)

    matrix_file = TABLES_DIR / "3x2_evaluation_matrix.md"
    ablation_file = TABLES_DIR / "proposed_ablation_matrix.md"

    matrix_file.write_text(matrix_md, encoding="utf-8")
    ablation_file.write_text(ablation_md, encoding="utf-8")

    print(f"Wrote {matrix_file}")
    print(f"Wrote {ablation_file}")


if __name__ == "__main__":
    main()
