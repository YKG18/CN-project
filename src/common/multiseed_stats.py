"""Multi-seed aggregation and significance tests (Review 3, Member 1).

The frozen project-standard split is shared across seeds. A seed therefore
indexes training-time randomness, so paired Wilcoxon tests by seed are
justified: each pair compares two methods on the same partition and seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from common.evaluator import D6_COLUMNS
except ImportError:  # scripts may import via src.common.*
    from src.common.evaluator import D6_COLUMNS

METRICS: tuple[str, ...] = (
    "f1",
    "precision",
    "recall",
    "fpr",
    "accuracy",
    "roc_auc",
    "inference_latency_ms",
)

LOWER_IS_BETTER: frozenset[str] = frozenset({"fpr", "inference_latency_ms"})

HEADLINE_METHODS: tuple[tuple[str, str, str], ...] = (
    ("base", "base_paper", "Base Paper"),
    ("saurabh", "saurabh_full", "Saurabh Extension"),
    ("proposed", "P6", "Proposed Work"),
)

HEADLINE_DATASETS: tuple[str, ...] = ("ncsrd", "data4cyber")

METHOD_DISPLAY = {method: label for method, _cfg, label in HEADLINE_METHODS}
CONFIG_BY_METHOD = {method: cfg for method, cfg, _label in HEADLINE_METHODS}

RUN_KEY_COLUMNS = ("dataset", "method", "config", "split_policy", "seed")

CI_LEVEL = 0.95


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float

    def mean_pm_std(self, digits: int = 4) -> str:
        if self.n == 0 or not np.isfinite(self.mean):
            return "n/a"
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"

    def ci_str(self, digits: int = 4) -> str:
        if self.n < 2 or not np.isfinite(self.ci_low):
            return "n/a"
        return f"[{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}]"


def summarize_values(values: Sequence[float], ci_level: float = CI_LEVEL) -> MetricSummary:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return MetricSummary(0, float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n < 2:
        return MetricSummary(n, mean, std, mean, mean)
    se = std / np.sqrt(n)
    t_crit = float(stats.t.ppf((1.0 + ci_level) / 2.0, df=n - 1))
    return MetricSummary(n, mean, std, mean - t_crit * se, mean + t_crit * se)


def wilcoxon_paired(
    proposed: Sequence[float],
    baseline: Sequence[float],
    metric: str,
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test of Proposed vs one baseline on one metric.

    Alternative is one-sided in the direction that is better for ``metric``.
    """
    a = np.asarray(proposed, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError("proposed and baseline must be paired (same length)")
    if a.size == 0:
        return {
            "n": 0,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "alternative": "none",
            "mean_diff": float("nan"),
            "note": "no paired observations",
        }

    diff = a - b
    mean_diff = float(np.mean(diff))
    alternative = "less" if metric in LOWER_IS_BETTER else "greater"

    if np.allclose(diff, 0.0):
        return {
            "n": int(a.size),
            "statistic": 0.0,
            "p_value": 1.0,
            "alternative": alternative,
            "mean_diff": 0.0,
            "note": "all paired differences are zero",
        }

    try:
        try:
            result = stats.wilcoxon(
                a, b, alternative=alternative, zero_method="wilcox", method="auto"
            )
        except TypeError:
            result = stats.wilcoxon(
                a, b, alternative=alternative, zero_method="wilcox", mode="auto"
            )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        note = ""
    except ValueError as exc:
        statistic = float("nan")
        p_value = float("nan")
        note = str(exc)

    return {
        "n": int(a.size),
        "statistic": statistic,
        "p_value": p_value,
        "alternative": alternative,
        "mean_diff": mean_diff,
        "note": note,
    }


def bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Family-wise Bonferroni: p_adj = min(1, p * m)."""
    m = len(p_values)
    if m == 0:
        return []
    adjusted = []
    for p in p_values:
        if p is None or not np.isfinite(p):
            adjusted.append(float("nan"))
        else:
            adjusted.append(float(min(1.0, p * m)))
    return adjusted


def significance_marker(p_adj: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p_adj):
        return "n/a"
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < alpha:
        return "*"
    return "ns"


def dedupe_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last row for each (dataset, method, config, split, seed)."""
    present = [c for c in RUN_KEY_COLUMNS if c in df.columns]
    if not present:
        return df
    return df.drop_duplicates(subset=present, keep="last").reset_index(drop=True)


def load_multiseed_csvs(raw_dir: Path) -> pd.DataFrame:
    """Load every D6 CSV under results/raw/, including seed_<n>/ subfolders."""
    raw_dir = Path(raw_dir)
    paths = sorted(raw_dir.glob("seed_*/*_results.csv"))
    paths += sorted(raw_dir.glob("multiseed_all_runs.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(D6_COLUMNS) + ["source_file"])
    df = pd.concat(frames, ignore_index=True)
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    for metric in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return dedupe_runs(df)


def paired_metric_frame(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    proposed_method: str = "proposed",
    proposed_config: str = "P6",
    baseline_method: str = "base",
    baseline_config: str = "base_paper",
    split_policy: str = "project_standard",
) -> pd.DataFrame:
    left = df[
        (df["dataset"] == dataset)
        & (df["method"] == proposed_method)
        & (df["config"].astype(str) == str(proposed_config))
        & (df["split_policy"] == split_policy)
    ][["seed", metric]].rename(columns={metric: "proposed"})
    right = df[
        (df["dataset"] == dataset)
        & (df["method"] == baseline_method)
        & (df["config"].astype(str) == str(baseline_config))
        & (df["split_policy"] == split_policy)
    ][["seed", metric]].rename(columns={metric: "baseline"})
    return left.merge(right, on="seed", how="inner")


def summarize_configuration(df: pd.DataFrame, metrics: Iterable[str] = METRICS) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "method", "config", "split_policy"]
    for keys, part in df.groupby(group_cols, dropna=False):
        dataset, method, config_name, split_policy = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "config": config_name,
            "split_policy": split_policy,
            "n_seeds": int(part["seed"].nunique()) if "seed" in part.columns else len(part),
        }
        for metric in metrics:
            if metric not in part.columns:
                continue
            summary = summarize_values(part[metric].tolist())
            row[f"{metric}_mean"] = summary.mean
            row[f"{metric}_std"] = summary.std
            row[f"{metric}_ci_low"] = summary.ci_low
            row[f"{metric}_ci_high"] = summary.ci_high
            row[f"{metric}_n"] = summary.n
        rows.append(row)
    return pd.DataFrame(rows)


def planned_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon + Bonferroni over the pre-registered 3x2 headline comparisons.

    Family size per dataset is 7 metrics x 2 baselines = 14 tests.
    Bonferroni is applied within each dataset family.
    """
    records: list[dict[str, Any]] = []
    for dataset in HEADLINE_DATASETS:
        family: list[dict[str, Any]] = []
        for baseline_method, baseline_config, baseline_label in HEADLINE_METHODS:
            if baseline_method == "proposed":
                continue
            for metric in METRICS:
                paired = paired_metric_frame(
                    df,
                    dataset=dataset,
                    metric=metric,
                    baseline_method=baseline_method,
                    baseline_config=baseline_config,
                )
                test = wilcoxon_paired(
                    paired["proposed"].tolist() if not paired.empty else [],
                    paired["baseline"].tolist() if not paired.empty else [],
                    metric=metric,
                )
                family.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "proposed_method": "proposed",
                        "proposed_config": "P6",
                        "baseline_method": baseline_method,
                        "baseline_config": baseline_config,
                        "baseline_label": baseline_label,
                        **test,
                    }
                )
        p_adj = bonferroni_adjust([row["p_value"] for row in family])
        for row, adj in zip(family, p_adj):
            row["p_adj_bonferroni"] = adj
            row["m_tests"] = len(family)
            row["significant_0.05"] = bool(np.isfinite(adj) and adj < 0.05)
            row["marker"] = significance_marker(adj)
            records.append(row)
    return pd.DataFrame(records)


def format_table_71(summary: pd.DataFrame) -> str:
    """Markdown for report Table 7.1 (headline 3x2, mean ± std)."""
    lines = [
        "# Table 7.1 — Headline 3×2 Evaluation Matrix (multi-seed)",
        "",
        "All results are reported as mean ± standard deviation. Differences are "
        "statistically significant at Bonferroni-adjusted p < 0.05 unless otherwise noted "
        "(see the Wilcoxon table).",
        "",
        "| Method | NCSRD F1 | NCSRD FPR | Data4Cyber F1 | Data4Cyber FPR | n |",
        "|---|---|---|---|---|---|",
    ]
    for method, config_name, label in HEADLINE_METHODS:
        cells = [f"**{label}**"]
        n_vals = []
        for dataset in HEADLINE_DATASETS:
            hit = summary[
                (summary["dataset"] == dataset)
                & (summary["method"] == method)
                & (summary["config"].astype(str) == config_name)
            ]
            if hit.empty:
                cells.extend(["n/a", "n/a"])
                continue
            row = hit.iloc[0]
            cells.append(_fmt_mean_std(row, "f1"))
            cells.append(_fmt_mean_std(row, "fpr"))
            n_vals.append(int(row["n_seeds"]))
        cells.append(str(min(n_vals) if n_vals else 0))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def format_detailed_table(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    dataset: str,
    title: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        "Values are mean ± std with 95% t-interval in brackets. Markers on the "
        "Proposed column are Bonferroni-adjusted Wilcoxon tests vs Saurabh "
        "(`*` p_adj < 0.05, `**` < 0.01, `***` < 0.001, `ns` not significant).",
        "",
        "| Metric | Base Paper | Saurabh Extension | Proposed Work | vs Base | vs Saurabh |",
        "|---|---|---|---|---|---|",
    ]
    metric_labels = {
        "f1": "F1 (attack class)",
        "precision": "Precision",
        "recall": "Recall",
        "fpr": "FPR",
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "inference_latency_ms": "Inference latency (ms/sample)",
    }
    for metric, label in metric_labels.items():
        cells = [label]
        for method, config_name, _disp in HEADLINE_METHODS:
            hit = summary[
                (summary["dataset"] == dataset)
                & (summary["method"] == method)
                & (summary["config"].astype(str) == config_name)
            ]
            if hit.empty:
                cells.append("n/a")
            else:
                row = hit.iloc[0]
                cells.append(f"{_fmt_mean_std(row, metric)} {_fmt_ci(row, metric)}")
        for baseline in ("base", "saurabh"):
            test_hit = tests[
                (tests["dataset"] == dataset)
                & (tests["metric"] == metric)
                & (tests["baseline_method"] == baseline)
            ]
            if test_hit.empty:
                cells.append("n/a")
            else:
                trow = test_hit.iloc[0]
                p_adj = trow["p_adj_bonferroni"]
                p_txt = "n/a" if not np.isfinite(p_adj) else f"{p_adj:.4g}"
                cells.append(f"{trow['marker']} (p_adj={p_txt})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def format_table_81(summary: pd.DataFrame, dataset: str = "ncsrd") -> str:
    """Ablation table. Empty if P0–P7 multi-seed rows are missing."""
    lines = [
        "# Table 8.1 — Incremental Ablation Results (multi-seed)",
        "",
        "Mean ± std over seeds. Contamination-injection columns are owned by "
        "Member 2 and are not filled here.",
        "",
        "| ID | Configuration | F1 | FPR | Precision | Recall | Latency (ms) | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "P0": "Base XGBoost (no EWMA)",
        "P1": "+ EWMA baseline tracking",
        "P2": "+ CUSUM contamination guard",
        "P3": "+ FPR-constrained threshold",
        "P4": "+ FastSHAP drift detector",
        "P5": "+ Knowledge distillation",
        "P6": "Full proposed teacher model",
        "P7": "Edge student alone",
    }
    any_row = False
    for cfg, label in labels.items():
        hit = summary[
            (summary["dataset"] == dataset)
            & (summary["method"] == "proposed")
            & (summary["config"].astype(str) == cfg)
        ]
        if hit.empty:
            lines.append(f"| **{cfg}** | {label} | n/a | n/a | n/a | n/a | n/a | 0 |")
            continue
        any_row = True
        row = hit.iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"**{cfg}**",
                    label,
                    _fmt_mean_std(row, "f1"),
                    _fmt_mean_std(row, "fpr"),
                    _fmt_mean_std(row, "precision"),
                    _fmt_mean_std(row, "recall"),
                    _fmt_mean_std(row, "inference_latency_ms"),
                    str(int(row["n_seeds"])),
                ]
            )
            + " |"
        )
    if not any_row:
        lines.append("")
        lines.append(
            "_No multi-seed ablation rows were found. Headline P6 results are in "
            "Tables 7.1–7.3. Run `python scripts/run_multiseed.py --include-ablation` "
            "to populate this table._"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_mean_std(row: pd.Series, metric: str, digits: int = 4) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None or not np.isfinite(mean):
        return "n/a"
    std_v = 0.0 if std is None or not np.isfinite(std) else float(std)
    return f"{float(mean):.{digits}f} ± {std_v:.{digits}f}"


def _fmt_ci(row: pd.Series, metric: str, digits: int = 4) -> str:
    lo = row.get(f"{metric}_ci_low")
    hi = row.get(f"{metric}_ci_high")
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"[{float(lo):.{digits}f}, {float(hi):.{digits}f}]"
