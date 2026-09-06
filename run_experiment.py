"""Unified Experiment Runner — Member 5.

CLI Entrypoint for running Base paper, Saurabh extension, and Proposed methods
across NCSRD and Data4Cyber datasets.

Usage:
    # the full 3x2 matrix in one command (what the report needs)
    python run_experiment.py --all

    # or one cell at a time
    python run_experiment.py --dataset ncsrd --method base --config base_paper --save
    python run_experiment.py --dataset ncsrd --method saurabh --config saurabh_full --save
    python run_experiment.py --dataset ncsrd --method proposed --config P6 --save
    python run_experiment.py --dataset data4cyber --method base --save

    # base-paper reference reproduction (NCSRD only, published-number comparison)
    python run_experiment.py --dataset ncsrd --method base --split-policy reference

`--all` always uses the project-standard split on NCSRD and the primary `block`
split on Data4Cyber. The Data4Cyber `scenario` holdout is a SECONDARY
novel-attack robustness experiment (D3) and is deliberately never part of the
3x2 matrix; run it explicitly via experiments/data4cyber/run_base.py.
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
    """Serialized booster size in KB.

    Base/Saurabh/Proposed all wrap the fitted booster in `.model`; a bare
    XGBClassifier exposes `save_model` directly. `.model_` is also accepted so
    the helper keeps working if a wrapper renames the attribute.
    """
    import tempfile

    for candidate in (model, getattr(model, "model", None),
                      getattr(model, "model_", None)):
        if candidate is None or not hasattr(candidate, "save_model"):
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp_path = f.name
            candidate.save_model(tmp_path)
            return round(os.path.getsize(tmp_path) / 1024.0, 2)
        except Exception:
            continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return 0.0


def window_size_for(dataset: str) -> int:
    """Window length appropriate to the dataset.

    Windows are built inside one block (D2/D4), so the window must fit in a
    block. NCSRD blocks hold ~1,400 rows, so the agreed 500-sample threshold
    window fits. Data4Cyber blocks are 120 one-second rows, so a 500-sample
    window yields ZERO windows; use that dataset's own 60-second convention.
    """
    if dataset == "data4cyber":
        return 60
    return config.THRESHOLD_WINDOW


def load_bundle_for_dataset(args: argparse.Namespace, feature_set: str = "saurabh49", balance: str = "class_weight"):
    if args.dataset == "data4cyber":
        from data4cyber_adapter import Data4CyberAdapter
        adapter = Data4CyberAdapter(split_mode="block", verbose=True)
        bundle = adapter.for_xgboost(balance=balance)
        if hasattr(bundle, "meta") and isinstance(bundle.meta, dict):
            bundle.train_block = bundle.meta.get("train_block")
            bundle.val_block = bundle.meta.get("val_block")
            bundle.test_block = bundle.meta.get("test_block")
        actual_feature_set = f"data4cyber_{bundle.X_train.shape[1]}"
        return bundle, actual_feature_set
    else:
        from ncsrd_adapter import NetworkDataAdapter
        adapter = NetworkDataAdapter.for_feature_set(feature_set)
        if args.split_policy == "project_standard":
            # The frozen index guarantees Base/Saurabh/Proposed see identical rows.
            adapter.load_split(config.SPLIT_INDEX_FILE)
        else:
            # Base-paper reference reproduction. `split()` returns the adapter,
            # so the bundle still has to be requested from it.
            adapter.split(**config.BASE_REFERENCE_SPLIT)
        bundle = adapter.for_xgboost(balance=balance)
        return bundle, feature_set


def run_base_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    from base.base_xgboost import BaseXGBoost, select_threshold

    config_name = args.config or "base_paper"
    bundle, feature_set = load_bundle_for_dataset(args, feature_set="base38", balance="none")

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
        bundle.y_test, p_test, window_size=window_size_for(args.dataset),
        block_ids=block_ids, thresholds=tau,
    )
    res["window_metrics"] = w_metrics
    return res


def run_saurabh_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    from saurabh.saurabh_xgboost import SaurabhXGBoost

    config_name = args.config or "saurabh_full"
    bundle, feature_set = load_bundle_for_dataset(args, feature_set="saurabh49", balance="class_weight")

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
        train_block_ids=getattr(bundle, "train_block", None),
        val_block_ids=getattr(bundle, "val_block", None),
    )
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    p_test = model.predict_proba(bundle.X_test, block_ids=getattr(bundle, "test_block", None))
    y_pred = model.predict(bundle.X_test, block_ids=getattr(bundle, "test_block", None))
    inf_time_sec = time.perf_counter() - t1

    n_test = len(bundle.y_test)
    inf_latency_ms = (inf_time_sec / n_test) * 1000.0 if n_test > 0 else 0.0
    model_size_kb = get_model_size_kb(model)

    # Module B decides per window, so there is no single global cut. Report the
    # global threshold it learned on validation, and score the predictions the
    # model itself produced -- `getattr(model, "static_threshold_", 0.5)` used to
    # fall back to 0.5 silently, which bypassed the dynamic thresholding entirely.
    if model.use_dynamic_threshold:
        tau = float(getattr(model.dynamic_threshold, "global_threshold_", 0.5) or 0.5)
        selected_on = "validation_per_window"
    else:
        tau = 0.5
        selected_on = "fixed_0.5"

    meta = {
        "dataset": args.dataset,
        "method": args.method,
        "config": config_name,
        "feature_set": feature_set,
        "split_policy": args.split_policy,
        "seed": args.seed,
        "threshold_selected_on": selected_on,
        "n_train": len(bundle.y_train),
        "n_test": n_test,
        "inference_latency_ms": round(inf_latency_ms, 6),
        "model_size_kb": model_size_kb,
        "notes": (
            f"Saurabh pipeline (train_time={train_time:.2f}s); "
            f"graph={model.use_correlation_graph} "
            f"dynamic_tau={model.use_dynamic_threshold} "
            f"shap_drift={model.use_shap_drift} "
            f"(drift detection does not affect classification metrics)"
        ),
    }

    evaluator = Evaluator()
    res = evaluator.evaluate_predictions(
        bundle.y_test, p_test, threshold=tau, meta=meta, y_pred=y_pred
    )

    block_ids = getattr(bundle, "test_block", None)
    w_metrics = evaluator.evaluate_windows(
        bundle.y_test, p_test, window_size=window_size_for(args.dataset),
        block_ids=block_ids, thresholds=tau, y_pred=y_pred,
    )
    res["window_metrics"] = w_metrics
    return res


def run_proposed_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    from proposed.proposed_xgboost import ProposedXGBoost

    config_name = args.config or "P6"
    bundle, feature_set = load_bundle_for_dataset(args, feature_set="saurabh49", balance="class_weight")

    model = ProposedXGBoost.for_config(
        config_name=config_name,
        seed=args.seed,
    )

    t0 = time.perf_counter()
    model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        train_block_ids=getattr(bundle, "train_block", None),
        val_block_ids=getattr(bundle, "val_block", None),
    )
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    p_test = model.predict_proba(bundle.X_test, block_ids=getattr(bundle, "test_block", None))
    y_pred = model.predict(bundle.X_test, block_ids=getattr(bundle, "test_block", None))
    inf_time_sec = time.perf_counter() - t1

    n_test = len(bundle.y_test)
    inf_latency_ms = (inf_time_sec / n_test) * 1000.0 if n_test > 0 else 0.0
    model_size_kb = get_model_size_kb(model)

    # The FPR-constrained threshold is the method's own decision rule; score the
    # predictions it produced rather than re-cutting the probabilities.
    tau = float(getattr(model, "selected_threshold_", 0.5) or 0.5)

    meta = {
        "dataset": args.dataset,
        "method": "proposed",
        "config": config_name,
        "feature_set": feature_set,
        "split_policy": args.split_policy,
        "seed": args.seed,
        "threshold_selected_on": "validation_fpr_constrained",
        "n_train": len(bundle.y_train),
        "n_test": n_test,
        "inference_latency_ms": round(inf_latency_ms, 6),
        "model_size_kb": model_size_kb,
        "notes": (
            f"Proposed {config_name} (train_time={train_time:.2f}s); "
            f"ewma={model.use_ewma} cusum={model.use_cusum} "
            f"constrained_tau={model.use_constrained_threshold} "
            f"fast_shap={model.use_fast_shap} distill={model.use_distillation}; "
            f"val_ewma_updates={model.validation_ewma_updates_} "
            f"val_drift_events={model.validation_drift_events_}"
        ),
    }

    evaluator = Evaluator()
    res = evaluator.evaluate_predictions(
        bundle.y_test, p_test, threshold=tau, meta=meta, y_pred=y_pred
    )

    block_ids = getattr(bundle, "test_block", None)
    w_metrics = evaluator.evaluate_windows(
        bundle.y_test, p_test, window_size=window_size_for(args.dataset),
        block_ids=block_ids, thresholds=tau, y_pred=y_pred,
    )
    res["window_metrics"] = w_metrics
    return res


# ---------------------------------------------------------------------------
# 3x2 matrix
# ---------------------------------------------------------------------------

RUNNERS = {
    "base": run_base_experiment,
    "saurabh": run_saurabh_experiment,
    "proposed": run_proposed_experiment,
}


def run_matrix(args: argparse.Namespace) -> list[Dict[str, Any]]:
    """Run all six dataset x method cells and return their D6 rows.

    Thin on purpose: it reuses the same per-cell functions as a single run, so
    there is exactly one implementation of each method.
    """
    rows: list[Dict[str, Any]] = []
    total = len(DATASETS) * len(METHODS)
    i = 0
    for dataset in DATASETS:
        for method in METHODS:
            i += 1
            cell = argparse.Namespace(**vars(args))
            cell.dataset = dataset
            cell.method = method
            cell.config = None                      # each runner picks its default
            cell.split_policy = "project_standard"  # the matrix is always this
            print(f"\n{'=' * 78}")
            print(f"[{i}/{total}] {method} x {dataset}")
            print("=" * 78)
            try:
                res = RUNNERS[method](cell)
                row = dict(res["d6_row"])
                row["_window"] = res.get("window_metrics", {})
                rows.append(row)
                if args.save:
                    save_result(cell, res)
            except Exception as exc:                # noqa: BLE001
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                rows.append({"dataset": dataset, "method": method,
                             "config": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def print_matrix(rows: list[Dict[str, Any]]) -> None:
    """Final comparison table: one row per method x dataset."""
    print(f"\n{'=' * 108}")
    print("FINAL 3x2 COMPARISON — attack class (label 1), project-standard split")
    print("NCSRD: stratified 20-min blocks | Data4Cyber: primary `block` split")
    print("=" * 108)
    head = (f"{'dataset':<12}{'method':<10}{'config':<14}{'features':<10}"
            f"{'P':>8}{'R':>8}{'F1':>8}{'FPR':>8}{'acc':>8}{'wF1':>8}{'AUC':>8}")
    print(head)
    print("-" * 108)
    for r in rows:
        if r.get("config") == "FAILED":
            print(f"{r['dataset']:<12}{r['method']:<10}{'FAILED':<14}"
                  f"{r.get('error', '')[:60]}")
            continue
        print(f"{r['dataset']:<12}{r['method']:<10}{r['config']:<14}"
              f"{str(r['feature_set']):<10}"
              f"{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}"
              f"{r['fpr']:>8.4f}{r['accuracy']:>8.4f}{r['weighted_f1']:>8.4f}"
              f"{r['roc_auc']:>8.4f}")
    print("-" * 108)
    print(f"{'':<12}{'':<10}{'':<14}{'':<10}"
          f"{'tp':>8}{'fp':>8}{'fn':>8}{'tn':>8}{'thresh':>8}{'lat/ms':>9}{'size/KB':>9}")
    for r in rows:
        if r.get("config") == "FAILED":
            continue
        print(f"{r['dataset']:<12}{r['method']:<10}{r['config']:<14}{'':<10}"
              f"{r['tp']:>8}{r['fp']:>8}{r['fn']:>8}{r['tn']:>8}"
              f"{r['threshold']:>8.4f}{r['inference_latency_ms']:>9.4f}"
              f"{r['model_size_kb']:>9.1f}")
    # Window-level stability (D4): only meaningful inside a block, which is how
    # the evaluator builds them. This is where the proposed method is supposed to
    # differ from Saurabh, since both can share the same probability ranking.
    if any(r.get("_window") for r in rows):
        print(f"{'':<12}{'':<10}{'':<14}{'':<10}"
              f"{'windows':>9}{'fpr_mean':>10}{'fpr_var':>10}{'fpr_max':>10}{'tau_std':>10}")
        for r in rows:
            w = r.get("_window") or {}
            if r.get("config") == "FAILED" or not w:
                continue
            print(f"{r['dataset']:<12}{r['method']:<10}{r['config']:<14}{'':<10}"
                  f"{w.get('n_windows', 0):>9}{w.get('fpr_mean', 0):>10.4f}"
                  f"{w.get('fpr_var', 0):>10.6f}{w.get('fpr_max', 0):>10.4f}"
                  f"{w.get('threshold_std', 0):>10.4f}")
        print("=" * 108)
    ok = sum(1 for r in rows if r.get("config") != "FAILED")
    print(f"{ok}/{len(rows)} cells completed. "
          f"Thresholds selected on validation; FPR = FP/(FP+TN).")


def save_result(args: argparse.Namespace, res: Dict[str, Any]) -> None:
    """Append the D6 row to the per-method CSV and dump the detail JSON."""
    d6_row = res["d6_row"]
    out_dir = Path(getattr(args, "output_dir", "results/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{args.method}_{args.dataset}_results.csv"
    Evaluator.save_d6_row(d6_row, csv_path)
    print(f"[Results] Appended D6 row to {csv_path}")

    json_path = (out_dir /
                 f"{args.method}_{args.dataset}_{d6_row['config']}_"
                 f"{args.split_policy}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[Results] Wrote detail JSON to {json_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--all", action="store_true",
                   help="run the full 3x2 matrix and print the comparison table")
    p.add_argument("--dataset", choices=DATASETS, help="Dataset name")
    p.add_argument("--method", choices=METHODS, help="Method name")
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

    if args.all:
        rows = run_matrix(args)
        print_matrix(rows)
        return 0 if all(r.get("config") != "FAILED" for r in rows) else 1

    if not args.dataset or not args.method:
        p.error("--dataset and --method are required unless --all is given")

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
        save_result(args, res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
