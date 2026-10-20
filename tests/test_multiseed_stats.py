"""Unit tests for multi-seed statistics and CLI plumbing (Member 1)."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.evaluator import D6_COLUMNS  # noqa: E402
from common.multiseed_stats import (  # noqa: E402
    bonferroni_adjust,
    load_multiseed_csvs,
    planned_comparisons,
    significance_marker,
    summarize_configuration,
    summarize_values,
    wilcoxon_paired,
)
from scripts.run_multiseed import jobs_for, parse_seed_spec  # noqa: E402

import run_experiment  # noqa: E402


class TestSeedCli(unittest.TestCase):
    def test_run_experiment_accepts_seed(self):
        args = run_experiment.build_parser().parse_args(
            ["--dataset", "ncsrd", "--method", "base", "--seed", "7"]
        )
        self.assertEqual(args.seed, 7)

    def test_seed_json_name_includes_seed(self):
        name = run_experiment.result_json_name(
            "proposed", "ncsrd", "P6", "project_standard", 12
        )
        self.assertIn("seed12", name)
        self.assertTrue(name.endswith(".json"))

    def test_parse_seed_spec(self):
        self.assertEqual(parse_seed_spec("1-5"), [1, 2, 3, 4, 5])
        self.assertEqual(parse_seed_spec("1,3,3,5"), [1, 3, 5])
        self.assertEqual(parse_seed_spec("42"), [42])

    def test_headline_job_count(self):
        jobs = jobs_for(
            seeds=list(range(1, 21)),
            datasets=["ncsrd", "data4cyber"],
            methods=["base", "saurabh", "proposed"],
            include_ablation=False,
            split_policy="project_standard",
        )
        # 20 seeds x 2 datasets x 3 methods
        self.assertEqual(len(jobs), 120)
        seeds = {int(j["seed"]) for j in jobs}
        self.assertEqual(seeds, set(range(1, 21)))


class TestSummaries(unittest.TestCase):
    def test_mean_std_and_ci_known_sample(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = summarize_values(values)
        self.assertEqual(summary.n, 5)
        self.assertAlmostEqual(summary.mean, 3.0)
        self.assertAlmostEqual(summary.std, float(np.std(values, ddof=1)))
        self.assertLess(summary.ci_low, summary.mean)
        self.assertGreater(summary.ci_high, summary.mean)

    def test_single_value_ci_collapses(self):
        summary = summarize_values([0.8])
        self.assertEqual(summary.n, 1)
        self.assertEqual(summary.ci_low, 0.8)
        self.assertEqual(summary.ci_high, 0.8)

    def test_wilcoxon_greater_when_proposed_wins(self):
        rng = np.random.default_rng(0)
        baseline = rng.normal(0.80, 0.01, size=20)
        proposed = baseline + 0.05
        result = wilcoxon_paired(proposed, baseline, "f1")
        self.assertEqual(result["alternative"], "greater")
        self.assertLess(result["p_value"], 0.05)

    def test_wilcoxon_less_for_fpr(self):
        baseline = np.linspace(0.02, 0.04, 20)
        proposed = baseline - 0.01
        result = wilcoxon_paired(proposed, baseline, "fpr")
        self.assertEqual(result["alternative"], "less")
        self.assertLess(result["p_value"], 0.05)

    def test_wilcoxon_all_ties(self):
        values = [0.9] * 10
        result = wilcoxon_paired(values, values, "f1")
        self.assertEqual(result["p_value"], 1.0)
        self.assertIn("zero", result["note"])

    def test_bonferroni_and_markers(self):
        adjusted = bonferroni_adjust([0.001, 0.02, 0.4])
        self.assertEqual(len(adjusted), 3)
        self.assertAlmostEqual(adjusted[0], 0.003)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertEqual(adjusted[2], 1.0)
        self.assertEqual(significance_marker(0.0005), "***")
        self.assertEqual(significance_marker(0.005), "**")
        self.assertEqual(significance_marker(0.04), "*")
        self.assertEqual(significance_marker(0.06), "ns")


class TestCsvAggregation(unittest.TestCase):
    def _write_d6(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=D6_COLUMNS)
            writer.writeheader()
            for row in rows:
                full = {col: "" for col in D6_COLUMNS}
                full.update(row)
                writer.writerow(full)

    def test_load_and_planned_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            rows = []
            for seed in range(1, 6):
                for dataset in ("ncsrd", "data4cyber"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": "base",
                            "config": "base_paper",
                            "split_policy": "project_standard",
                            "seed": seed,
                            "f1": 0.80 + 0.001 * seed,
                            "precision": 0.81,
                            "recall": 0.79,
                            "fpr": 0.02,
                            "accuracy": 0.90,
                            "roc_auc": 0.95,
                            "inference_latency_ms": 0.01,
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": "saurabh",
                            "config": "saurabh_full",
                            "split_policy": "project_standard",
                            "seed": seed,
                            "f1": 0.90 + 0.001 * seed,
                            "precision": 0.91,
                            "recall": 0.89,
                            "fpr": 0.005,
                            "accuracy": 0.97,
                            "roc_auc": 0.99,
                            "inference_latency_ms": 0.004,
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": "proposed",
                            "config": "P6",
                            "split_policy": "project_standard",
                            "seed": seed,
                            "f1": 0.92 + 0.001 * seed,
                            "precision": 0.93,
                            "recall": 0.91,
                            "fpr": 0.004,
                            "accuracy": 0.98,
                            "roc_auc": 0.991,
                            "inference_latency_ms": 0.002,
                        }
                    )
            self._write_d6(raw / "seed_1" / "dummy_results.csv", rows)
            df = load_multiseed_csvs(raw)
            self.assertEqual(len(df), 30)
            summary = summarize_configuration(df)
            self.assertFalse(summary.empty)
            ncsrd_p6 = summary[
                (summary["dataset"] == "ncsrd")
                & (summary["method"] == "proposed")
                & (summary["config"] == "P6")
            ].iloc[0]
            self.assertEqual(int(ncsrd_p6["n_seeds"]), 5)
            tests = planned_comparisons(df)
            self.assertEqual(len(tests), 28)  # 2 datasets x 14 tests
            f1_vs_base = tests[
                (tests["dataset"] == "ncsrd")
                & (tests["metric"] == "f1")
                & (tests["baseline_method"] == "base")
            ].iloc[0]
            self.assertEqual(int(f1_vs_base["n"]), 5)
            self.assertAlmostEqual(float(f1_vs_base["p_adj_bonferroni"]), 0.4375)
            self.assertEqual(int(f1_vs_base["m_tests"]), 14)

    def test_dedupe_keeps_last_seed_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            self._write_d6(
                raw / "seed_1" / "base_ncsrd_results.csv",
                [
                    {
                        "dataset": "ncsrd",
                        "method": "base",
                        "config": "base_paper",
                        "split_policy": "project_standard",
                        "seed": 1,
                        "f1": 0.1,
                    },
                    {
                        "dataset": "ncsrd",
                        "method": "base",
                        "config": "base_paper",
                        "split_policy": "project_standard",
                        "seed": 1,
                        "f1": 0.9,
                    },
                ],
            )
            df = load_multiseed_csvs(raw)
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(float(df.iloc[0]["f1"]), 0.9)


if __name__ == "__main__":
    unittest.main()
