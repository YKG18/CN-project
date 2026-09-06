"""Unit tests for src/common/evaluator.py — Member 5."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from common.evaluator import D6_COLUMNS, Evaluator


class TestEvaluator(unittest.TestCase):
    def test_evaluator_basic_metrics(self):
        evaluator = Evaluator()
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.6, 0.4, 0.3, 0.1, 0.85])

        res = evaluator.evaluate_predictions(y_true, y_prob, threshold=0.5)

        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("f1", res)
        self.assertIn("fpr", res)
        self.assertIn("d6_row", res)
        self.assertEqual(len(res["d6_row"]), len(D6_COLUMNS))

    def test_evaluator_d6_schema_keys(self):
        evaluator = Evaluator()
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        meta = {
            "dataset": "ncsrd",
            "method": "base",
            "config": "base_paper",
            "feature_set": "base38",
            "split_policy": "project_standard",
            "seed": 42,
        }

        res = evaluator.evaluate_predictions(y_true, y_prob, threshold=0.5, meta=meta)
        row = res["d6_row"]

        for col in D6_COLUMNS:
            self.assertIn(col, row)

        self.assertEqual(row["dataset"], "ncsrd")
        self.assertEqual(row["method"], "base")
        self.assertEqual(row["seed"], 42)

    def test_evaluator_window_metrics(self):
        evaluator = Evaluator()
        y_true = np.random.randint(0, 2, size=1500)
        y_prob = np.random.rand(1500)
        block_ids = np.array([0] * 1000 + [1] * 500)

        res = evaluator.evaluate_windows(y_true, y_prob, window_size=500, block_ids=block_ids, thresholds=0.5)

        self.assertEqual(res["n_windows"], 3)
        self.assertTrue(0.0 <= res["fpr_mean"] <= 1.0)
        self.assertTrue(res["fpr_var"] >= 0.0)
        self.assertTrue(0.0 <= res["fpr_max"] <= 1.0)

    def test_save_d6_row(self):
        evaluator = Evaluator()
        y_true = np.array([0, 1])
        y_prob = np.array([0.2, 0.8])

        res = evaluator.evaluate_predictions(y_true, y_prob, threshold=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_results.csv"
            evaluator.save_d6_row(res["d6_row"], csv_path)

            self.assertTrue(csv_path.exists())
            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0].split(",")[0], "dataset")


if __name__ == "__main__":
    unittest.main()
