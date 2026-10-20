"""Aggregate multi-seed D6 rows into summary tables, tests, and error-bar plots.

Usage:
    python scripts/analyze_multiseed.py
    python scripts/analyze_multiseed.py --raw-dir results/raw --min-seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.multiseed_stats import (  # noqa: E402
    HEADLINE_DATASETS,
    HEADLINE_METHODS,
    format_detailed_table,
    format_table_71,
    format_table_81,
    load_multiseed_csvs,
    planned_comparisons,
    summarize_configuration,
)

DATASET_LABEL = {"ncsrd": "NCSRD-DS-5GDDoS", "data4cyber": "Data4Cyber"}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def plot_grouped_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    use_ci: bool = True,
) -> None:
    methods = list(HEADLINE_METHODS)
    datasets = list(HEADLINE_DATASETS)
    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (method, config_name, label) in enumerate(methods):
        means = []
        lower_err = []
        upper_err = []
        for dataset in datasets:
            hit = summary[
                (summary["dataset"] == dataset)
                & (summary["method"] == method)
                & (summary["config"].astype(str) == config_name)
            ]
            if hit.empty or not np.isfinite(hit.iloc[0][f"{metric}_mean"]):
                means.append(np.nan)
                lower_err.append(0.0)
                upper_err.append(0.0)
                continue
            row = hit.iloc[0]
            mean = float(row[f"{metric}_mean"])
            means.append(mean)
            if use_ci:
                lo = float(row[f"{metric}_ci_low"])
                hi = float(row[f"{metric}_ci_high"])
                lower_err.append(max(0.0, mean - lo))
                upper_err.append(max(0.0, hi - mean))
            else:
                std = float(row[f"{metric}_std"])
                lower_err.append(0.0 if not np.isfinite(std) else std)
                upper_err.append(0.0 if not np.isfinite(std) else std)
        err = np.vstack([lower_err, upper_err])
        ax.bar(x + (i - 1) * width, means, width, label=label, yerr=err, capsize=4)
    ax.set_xticks(x, [DATASET_LABEL.get(d, d) for d in datasets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_n_seeds(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels = []
    counts = []
    for dataset in HEADLINE_DATASETS:
        for method, config_name, disp in HEADLINE_METHODS:
            hit = summary[
                (summary["dataset"] == dataset)
                & (summary["method"] == method)
                & (summary["config"].astype(str) == config_name)
            ]
            labels.append(f"{dataset}\n{disp}")
            counts.append(0 if hit.empty else int(hit.iloc[0]["n_seeds"]))
    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)), labels, rotation=0)
    ax.set_ylabel("Completed seeds")
    ax.set_title("Multi-seed coverage (target = 20)")
    ax.axhline(20, linestyle="--", linewidth=1, color="black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    p.add_argument("--tables-dir", type=Path, default=ROOT / "results" / "tables")
    p.add_argument("--plots-dir", type=Path, default=ROOT / "results" / "plots")
    p.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="Warn if any headline cell has fewer than this many seeds",
    )
    args = p.parse_args(argv)

    df = load_multiseed_csvs(args.raw_dir)
    if df.empty:
        print(
            f"No multi-seed CSVs found under {args.raw_dir}. "
            "Run python scripts/run_multiseed.py --seeds 1-20 --save first.",
            file=sys.stderr,
        )
        return 1

    summary = summarize_configuration(df)
    tests = planned_comparisons(df)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.tables_dir / "multiseed_summary.csv", index=False)
    tests.to_csv(args.tables_dir / "multiseed_wilcoxon.csv", index=False)
    df.to_csv(args.tables_dir / "multiseed_runs_deduped.csv", index=False)
    print(f"wrote {args.tables_dir / 'multiseed_summary.csv'}")
    print(f"wrote {args.tables_dir / 'multiseed_wilcoxon.csv'}")

    write_text(args.tables_dir / "table_7_1_multiseed.md", format_table_71(summary))
    write_text(
        args.tables_dir / "table_7_2_multiseed.md",
        format_detailed_table(
            summary,
            tests,
            dataset="ncsrd",
            title="Table 7.2 — Detailed NCSRD Comparison (multi-seed)",
        ),
    )
    write_text(
        args.tables_dir / "table_7_3_multiseed.md",
        format_detailed_table(
            summary,
            tests,
            dataset="data4cyber",
            title="Table 7.3 — Detailed Data4Cyber Comparison (multi-seed)",
        ),
    )
    write_text(args.tables_dir / "table_8_1_multiseed.md", format_table_81(summary))

    wilcoxon_md = [
        "# Wilcoxon signed-rank tests (Proposed vs baselines)",
        "",
        "Pairing is by seed on the frozen project-standard split. Bonferroni "
        "correction is applied within each dataset over 14 planned tests "
        "(7 metrics × 2 baselines).",
        "",
        "| Dataset | Metric | Baseline | n | mean_diff | p | p_adj | marker | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if not tests.empty:
        for _, row in tests.iterrows():
            p_raw = row["p_value"]
            p_adj = row["p_adj_bonferroni"]
            wilcoxon_md.append(
                "| "
                + " | ".join(
                    [
                        str(row["dataset"]),
                        str(row["metric"]),
                        str(row["baseline_label"]),
                        str(int(row["n"])),
                        "n/a" if not np.isfinite(row["mean_diff"]) else f"{row['mean_diff']:.6g}",
                        "n/a" if not np.isfinite(p_raw) else f"{p_raw:.4g}",
                        "n/a" if not np.isfinite(p_adj) else f"{p_adj:.4g}",
                        str(row["marker"]),
                        str(row["note"]).replace("|", "/"),
                    ]
                )
                + " |"
            )
    wilcoxon_md.append("")
    write_text(args.tables_dir / "multiseed_wilcoxon.md", "\n".join(wilcoxon_md))

    plot_grouped_metric(
        summary,
        "f1",
        "F1 (attack class)",
        "Attack-class F1 (mean, 95% CI) over seeds",
        args.plots_dir / "07_multiseed_f1_errorbars.png",
    )
    plot_grouped_metric(
        summary,
        "fpr",
        "False positive rate",
        "False positive rate (mean, 95% CI) over seeds",
        args.plots_dir / "08_multiseed_fpr_errorbars.png",
    )
    plot_grouped_metric(
        summary,
        "inference_latency_ms",
        "Inference latency (ms/sample)",
        "Inference latency (mean, 95% CI) over seeds",
        args.plots_dir / "09_multiseed_latency_errorbars.png",
    )
    plot_n_seeds(summary, args.plots_dir / "10_multiseed_coverage.png")

    warnings = []
    for dataset in HEADLINE_DATASETS:
        for method, config_name, label in HEADLINE_METHODS:
            hit = summary[
                (summary["dataset"] == dataset)
                & (summary["method"] == method)
                & (summary["config"].astype(str) == config_name)
            ]
            n = 0 if hit.empty else int(hit.iloc[0]["n_seeds"])
            if n < args.min_seeds:
                warnings.append(f"{dataset}/{label}: {n} seeds (need {args.min_seeds})")
    if warnings:
        print("WARNING: incomplete multi-seed coverage:")
        for line in warnings:
            print(f"  - {line}")
        return 0

    print("All headline cells meet the requested seed count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
