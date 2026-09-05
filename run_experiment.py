"""Unified experiment runner — OWNER: Member 5. Not implemented yet.

This file exists to fix the interface now, so that M1, M2 and M3 can write
their modules against it while M5 builds the evaluator.

Target usage (Implementation.md §8):

    python run_experiment.py --dataset ncsrd      --method base
    python run_experiment.py --dataset ncsrd      --method saurabh
    python run_experiment.py --dataset ncsrd      --method proposed
    python run_experiment.py --dataset data4cyber --method base

What every method must expose so the runner can treat them interchangeably:

    class SomeMethod:
        def fit(self, X, y) -> None: ...
        def predict_proba(self, X) -> np.ndarray:   # P(attack), shape (n,)
        def predict(self, X) -> np.ndarray:         # 0/1, shape (n,)

What every run must append to `results/raw/<method>_<dataset>_results.csv`
(PROJECT_DECISIONS.md D6 — keep this exact column order):

    dataset, method, config, feature_set, split_policy, seed,
    precision, recall, f1, fpr, accuracy, weighted_f1, roc_auc,
    tp, fp, fn, tn, threshold, threshold_selected_on,
    n_train, n_test, inference_latency_ms, model_size_kb, notes

Metrics refer to the ATTACK class (label 1); FPR = FP / (FP + TN).
`config` distinguishes e.g. base_paper / base_saurabh_static.
`split_policy` is "reference" or "project_standard".

Until M5 lands the evaluator, each member writes their own CSV in this schema
so the rows merge cleanly later.
"""

from __future__ import annotations

import argparse
import sys

DATASETS = ("ncsrd", "data4cyber")
METHODS = ("base", "saurabh", "proposed")

RESULT_COLUMNS = [
    "dataset", "method", "config", "feature_set", "split_policy", "seed",
    "precision", "recall", "f1", "fpr", "accuracy", "weighted_f1", "roc_auc",
    "tp", "fp", "fn", "tn", "threshold", "threshold_selected_on",
    "n_train", "n_test", "inference_latency_ms", "model_size_kb", "notes",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--split-policy", choices=["reference", "project_standard"],
                   default="project_standard")
    p.add_argument("--config", default=None,
                   help="method-specific configuration name, e.g. base_paper")
    args = p.parse_args(argv)

    print(f"run_experiment.py is not implemented yet (Member 5 owns it).\n"
          f"  requested: dataset={args.dataset} method={args.method} "
          f"split={args.split_policy} config={args.config}\n"
          f"For now, run the module directly, e.g. "
          f"experiments/{args.dataset}/run_{args.method}.py",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
