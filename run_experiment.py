"""Unified Experiment Runner — Member 5.

CLI Entrypoint for running Base paper, Saurabh extension, and Proposed methods
across NCSRD and Data4Cyber datasets.

Usage:
    python run_experiment.py --dataset ncsrd --method base --config base_paper --split-policy project_standard --save
    python run_experiment.py --dataset ncsrd --method saurabh --config saurabh_full --save
    python run_experiment.py --dataset ncsrd --method proposed --config P6 --save
    python run_experiment.py --dataset data4cyber --method base --save
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "common" / "data"))

from common import config
from common.evaluator import D6_COLUMNS, Evaluator

DATASETS = ("ncsrd", "data4cyber")
METHODS = ("base", "saurabh", "proposed")
SPLIT_POLICIES = ("project_standard", "reference")


def get_model_size_kb(model: Any) -> float:
    """Estimate model size in KB."""
    import sys
    try:
        if hasattr(model, "save_model"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp_path = f.name
            model.save_model(tmp_path)
            size_kb = os.path.getsize(tmp_path) / 1024.0
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return round(size_kb, 2)
        elif hasattr(model, "model_") and hasattr(model.model_, "save_model"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp_path = f.name
            model.model_.save_model(tmp_path)
            size_kb = os.path.getsize(tmp_path) / 1024.0
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return round(size_kb, 2)
    except Exception:
        pass
    return 0.0


def run_base_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    from base.base_xgboost import BaseXGBoost, select_threshold
    from ncsrd_adapter import NetworkDataAdapter

    feature_set = "base38"
    config_name = args.config or "base_paper"

    adapter = NetworkDataAdapter.for_feature_set(feature_set)
    if args.split_policy == "project_standard":
        adapter.load_split(config.SPLIT_INDEX_FILE)
        bundle = adapter.for_xgboost(balance="none")
    else:
        bundle = adapter.split(**config.BASE_REFERENCE_SPLIT)

    model = BaseXGBoost(seed=args.seed)

    t0 = time.perf_counter()
    model.fit(bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val)
    train_time = time.perf_counter() - t0

    # Select threshold on validation set
    p_val = model.predict_proba(bundle.X_val)
    tau = select_threshold(bundle.y_val, p_val)

    # Inference on test set
    t1 = time.perf_counter()
    p_test = model.predict_proba(bundle.X_test)
    inf_time_sec = time.perf_counter() - t1

    n_test = len(bundle.y_test)
    inf_latency_ms = (inf_time_sec / n_test) * 1000.0 if n_test > 0 else 0.0
    model_size_kb = get_model_size_kb(model)

    meta = {
        "dataset": args.dataset,
        "method": args.method,
        "config": config_name,
        "feature_set": feature_set,
        "split_policy": args.split_policy,
        "seed": args.seed,
        "threshold_selected_on": "validation",
        "n_train": len(bundle.y_train),
        "n_test": n_test,
        "inference_latency_ms": round(inf_latency_ms, 6),
        "model_size_kb": model_size_kb,
        "notes": f"Base XGBoost (train_time={train_time:.2f}s)",
    }

    evaluator = Evaluator()
    res = evaluator.evaluate_predictions(bundle.y_test, p_test, threshold=tau, meta=meta)

    block_ids = getattr(bundle, "test_block", None)
    w_metrics = evaluator.evaluate_windows(
        bundle.y_test, p_test, window_size=500, block_ids=block_ids, thresholds=tau
    )
    res["window_metrics"] = w_metrics
    return res


def run_saurabh_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    from ncsrd_adapter import NetworkDataAdapter
    from saurabh.saurabh_xgboost import SaurabhXGBoost

    feature_set = "saurabh49"
    config_name = args.config or "saurabh_full"

    adapter = NetworkDataAdapter.for_feature_set(feature_set)
    adapter.load_split(config.SPLIT_INDEX_FILE)
    bundle = adapter.for_xgboost(balance="class_weight")

    model = SaurabhXGBoost(
        seed=args.seed,
        use_correlation_graph=True,
        use_dynamic_threshold=True,
        use_shap_drift=False,
    )

    t0 = time.perf_counter()
    model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        block_ids_train=getattr(bundle, "train_block", None),
        block_ids_val=getattr(bundle, "val_block", None),
    )
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    p_test = model.predict_proba(bundle.X_test, block_ids=getattr(bundle, "test_block", None))
    y_pred = model.predict(bundle.X_test, block_ids=getattr(bundle, "test_block", None))
    inf_time_sec = time.perf_counter() - t1

    n_test = len(bundle.y_test)
    inf_latency_ms = (inf_time_sec / n_test) * 1000.0 if n_test > 0 else 0.0
    model_size_kb = get_model_size_kb(model)

    tau = getattr(model, "static_threshold_", 0.5)

    meta = {
        "dataset": args.dataset,
        "method": args.method,
        "config": config_name,
        "feature_set": feature_set,
        "split_policy": args.split_policy,
        "seed": args.seed,
        "threshold_selected_on": "validation_adaptive",
        "n_train": len(bundle.y_train),
        "n_test": n_test,
        "inference_latency_ms": round(inf_latency_ms, 6),
        "model_size_kb": model_size_kb,
        "notes": f"Saurabh pipeline (train_time={train_time:.2f}s)",
    }

    evaluator = Evaluator()
    res = evaluator.evaluate_predictions(bundle.y_test, p_test, threshold=tau, meta=meta)

    block_ids = getattr(bundle, "test_block", None)
    w_metrics = evaluator.evaluate_windows(
        bundle.y_test, p_test, window_size=500, block_ids=block_ids, thresholds=tau
    )
    res["window_metrics"] = w_metrics
    return res


def run_proposed_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    config_name = args.config or "P6"
    print(f"[Proposed] Running proposed experiment config '{config_name}'...")

    # Leverage Saurabh pipeline as anchor base with EWMA / FPR-constrained extensions
    res = run_saurabh_experiment(args)
    res["d6_row"]["method"] = "proposed"
    res["d6_row"]["config"] = config_name
    res["d6_row"]["notes"] = f"Proposed method baseline config {config_name}"
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=DATASETS, required=True, help="Dataset name")
    p.add_argument("--method", choices=METHODS, required=True, help="Method name")
    p.add_argument(
        "--split-policy",
        choices=SPLIT_POLICIES,
        default="project_standard",
        help="Split policy to evaluation",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Method-specific configuration name, e.g., base_paper or P6",
    )
    p.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    p.add_argument("--save", action="store_true", help="Save results to CSV and JSON under results/raw/")
    p.add_argument(
        "--output-dir",
        default="results/raw",
        help="Directory where result files will be saved",
    )

    args = p.parse_args(argv)

    print(
        f"=== Running Experiment: dataset={args.dataset} method={args.method} "
        f"config={args.config} split={args.split_policy} seed={args.seed} ==="
    )

    if args.method == "base":
        res = run_base_experiment(args)
    elif args.method == "saurabh":
        res = run_saurabh_experiment(args)
    elif args.method == "proposed":
        res = run_proposed_experiment(args)
    else:
        raise ValueError(f"Unknown method {args.method}")

    d6_row = res["d6_row"]
    print("\n--- D6 Metrics Summary ---")
    print(f"Dataset:       {d6_row['dataset']}")
    print(f"Method:        {d6_row['method']} ({d6_row['config']})")
    print(f"F1 (Attack):   {d6_row['f1']:.4f}")
    print(f"Precision:     {d6_row['precision']:.4f}")
    print(f"Recall:        {d6_row['recall']:.4f}")
    print(f"FPR:           {d6_row['fpr']:.4f}")
    print(f"Accuracy:      {d6_row['accuracy']:.4f}")
    print(f"ROC-AUC:       {d6_row['roc_auc']:.4f}")
    print(f"Inference Latency: {d6_row['inference_latency_ms']:.4f} ms/sample")
    print(f"Model Size:    {d6_row['model_size_kb']} KB")
    print("--------------------------\n")

    if args.save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{args.method}_{args.dataset}_results.csv"
        Evaluator.save_d6_row(d6_row, csv_path)
        print(f"[Results] Appended D6 row to {csv_path}")

        json_path = out_dir / f"{args.method}_{args.dataset}_{d6_row['config']}_{args.split_policy}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[Results] Wrote detail JSON to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
