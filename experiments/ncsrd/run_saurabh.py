"""
Unified Saurabh methodology experiment runner.

Combines:
    - Full A + B + C experiment
    - Controlled ablation study
    - SHAP feature-importance drift experiment

Usage:
    python experiments/ncsrd/run_saurabh.py --experiment full
    python experiments/ncsrd/run_saurabh.py --experiment ablation
    python experiments/ncsrd/run_saurabh.py --experiment shap-drift
    python experiments/ncsrd/run_saurabh.py --experiment all

All experiments use the frozen project-standard NSCRD split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import config
from common.data.ncsrd_adapter import NetworkDataAdapter
from base.base_xgboost import evaluate
from saurabh.saurabh_xgboost import SaurabhXGBoost


# ======================================================================
# EXPERIMENT CONFIGURATIONS
# ======================================================================

FULL_CONFIG = {
    "use_correlation_graph": True,
    "use_dynamic_threshold": True,
    "use_shap_drift": True,
}

ABLATIONS = {
    "full": FULL_CONFIG,
    "no_correlation": {
        "use_correlation_graph": False,
        "use_dynamic_threshold": True,
        "use_shap_drift": True,
    },
    "no_dynamic": {
        "use_correlation_graph": True,
        "use_dynamic_threshold": False,
        "use_shap_drift": True,
    },
    "no_shap": {
        "use_correlation_graph": True,
        "use_dynamic_threshold": True,
        "use_shap_drift": False,
    },
    "xgboost_only": {
        "use_correlation_graph": False,
        "use_dynamic_threshold": False,
        "use_shap_drift": False,
    },
    # --- Saurabh's published five-configuration ablation (paper Table) ------
    # The four entries above are a leave-one-out design of our own. These are
    # the ADDITIVE configurations the paper actually reports, so its numbers
    # (A0 0.9573, A1 0.9653, A2 0.9625, A3 0.9648, A4 0.9730) can be compared
    # row for row. A0 == xgboost_only and A3 == full; both are kept separately
    # so each design reads cleanly on its own.
    "A0_baseline_static": {
        "use_correlation_graph": False,
        "use_dynamic_threshold": False,
        "use_shap_drift": False,
    },
    "A1_graph": {
        "use_correlation_graph": True,
        "use_dynamic_threshold": False,
        "use_shap_drift": False,
    },
    "A2_shap_drift": {
        "use_correlation_graph": False,
        "use_dynamic_threshold": False,
        "use_shap_drift": True,
    },
    "A3_all_three": {
        "use_correlation_graph": True,
        "use_dynamic_threshold": True,
        "use_shap_drift": True,
    },
    "A4_baseline_adaptive": {
        "use_correlation_graph": False,
        "use_dynamic_threshold": True,
        "use_shap_drift": False,
    },
}


# ======================================================================
# SHARED HELPERS
# ======================================================================

def load_data():
    """Load the frozen project-standard Saurabh49 split once."""
    adapter = NetworkDataAdapter.for_feature_set("saurabh49")
    adapter.load_split(config.SPLIT_INDEX_FILE)

    bundle = adapter.for_xgboost(balance="class_weight")

    print()
    print("[data]")
    print("train:", bundle.X_train.shape)
    print("val:  ", bundle.X_val.shape)
    print("test: ", bundle.X_test.shape)

    return bundle


def build_model(switches: dict[str, bool]) -> SaurabhXGBoost:
    """Create a Saurabh model from module switches."""
    return SaurabhXGBoost(
        seed=config.SEED,
        use_correlation_graph=switches["use_correlation_graph"],
        use_dynamic_threshold=switches["use_dynamic_threshold"],
        use_shap_drift=switches["use_shap_drift"],
    )


def fit_model(
    model: SaurabhXGBoost,
    bundle,
) -> float:
    """Fit the model and return training time in seconds."""
    start = time.perf_counter()

    model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        train_block_ids=bundle.train_block,
        val_block_ids=bundle.val_block,
    )

    return time.perf_counter() - start


def predict_model(
    model: SaurabhXGBoost,
    bundle,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run test inference and return probabilities, predictions, elapsed time."""
    start = time.perf_counter()

    probabilities = model.predict_proba(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    predictions = model.predict(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    elapsed = time.perf_counter() - start

    return probabilities, predictions, elapsed


def confusion_counts(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, int]:
    """Compute confusion-matrix counts from actual pipeline predictions."""
    return {
        "true_positive": int(((predictions == 1) & (labels == 1)).sum()),
        "false_positive": int(((predictions == 1) & (labels == 0)).sum()),
        "false_negative": int(((predictions == 0) & (labels == 1)).sum()),
        "true_negative": int(((predictions == 0) & (labels == 0)).sum()),
    }


def pipeline_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Compute metrics.

    Classification metrics are derived from actual pipeline predictions,
    which matters when dynamic thresholding uses per-window thresholds.
    ROC-AUC is derived from probabilities.
    """
    cm = confusion_counts(predictions, labels)

    tp = cm["true_positive"]
    fp = cm["false_positive"]
    fn = cm["false_negative"]
    tn = cm["true_negative"]

    accuracy = (tp + tn) / len(labels)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # evaluate() is retained for ROC-AUC and weighted metrics.
    base = evaluate(labels, probabilities, 0.5)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "weighted_f1": float(base.get("weighted_f1", f1)),
        "roc_auc": float(base["roc_auc"]),
        "false_positive_rate": float(fpr),
    }

    return metrics, cm


def summarize_drift(drift_result, analysis_seconds: float) -> dict[str, Any]:
    """Convert a SHAP drift result into a JSON-safe summary."""
    tau = np.asarray(drift_result.tau, dtype=float)
    flags = np.asarray(drift_result.drift, dtype=bool)

    return {
        "enabled": True,
        "analysis_seconds": float(analysis_seconds),
        "n_windows": int(drift_result.n_windows),
        "drift_windows": int(flags.sum()),
        "drift_rate": float(flags.mean()) if len(flags) else 0.0,
        "min_kendall_tau": float(tau.min()) if len(tau) else None,
        "mean_kendall_tau": float(tau.mean()) if len(tau) else None,
        "max_kendall_tau": float(tau.max()) if len(tau) else None,
        "std_kendall_tau": float(tau.std()) if len(tau) else None,
        "tau_values": [float(value) for value in tau],
        "drift_flags": [bool(value) for value in flags],
    }


def run_drift_detection(
    model: SaurabhXGBoost,
    bundle,
) -> tuple[Any, float]:
    """
    Run SHAP drift detection on test features only.

    Test labels are never used for drift detection.
    """
    start = time.perf_counter()

    drift_result = model.detect_shap_drift(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    return drift_result, time.perf_counter() - start


SAVE_ENABLED = False


def save_json(filename: str, data: Any) -> Path | None:
    """Save experiment output under results/raw, only when --save is given.

    Result collection belongs to Member 5; a verification run must not silently
    overwrite committed artifacts.
    """
    if not SAVE_ENABLED:
        print(f"[skip] {filename} not written (pass --save to store it)")
        return None

    output_dir = ROOT / "results" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=float)

    print(f"saved -> {output_path.relative_to(ROOT)}")

    return output_path


def print_metrics(metrics: dict[str, float]) -> None:
    print()
    print("[metrics]")
    for name, value in metrics.items():
        print(f"{name:24s}: {value:.6f}")


# ======================================================================
# FULL A + B + C EXPERIMENT
# ======================================================================

def run_full_experiment(bundle) -> dict[str, Any]:
    """Run the complete Saurabh methodology: A + B + C."""
    print()
    print("=" * 72)
    print("SAURABH FULL DATASET EXPERIMENT")
    print("=" * 72)

    model = build_model(FULL_CONFIG)

    print("\n[training]")
    train_seconds = fit_model(model, bundle)
    print(f"training time: {train_seconds:.2f}s")

    print("\n[test prediction]")
    probabilities, predictions, inference_seconds = predict_model(model, bundle)

    metrics, cm = pipeline_metrics(
        predictions,
        probabilities,
        bundle.y_test,
    )

    print("\n[SHAP drift analysis]")
    drift_result, drift_seconds = run_drift_detection(model, bundle)
    drift_summary = summarize_drift(drift_result, drift_seconds)

    result = {
        "experiment": "saurabh_full",
        "dataset": "ncsrd",
        "feature_set": "saurabh49",
        "seed": int(config.SEED),
        "split_file": str(config.SPLIT_INDEX_FILE),
        "samples": {
            "train": int(len(bundle.y_train)),
            "validation": int(len(bundle.y_val)),
            "test": int(len(bundle.y_test)),
        },
        "model": model.summary(),
        "metrics": metrics,
        "confusion_matrix": cm,
        "performance": {
            "training_seconds": float(train_seconds),
            "test_inference_seconds": float(inference_seconds),
            "test_inference_ms_per_sample": float(
                inference_seconds * 1000 / len(bundle.y_test)
            ),
            "shap_drift_seconds": float(drift_seconds),
        },
        "shap_drift": drift_summary,
    }

    print_metrics(metrics)

    print("\n[confusion matrix]")
    for name, value in cm.items():
        print(f"{name}: {value}")

    print("\n[performance]")
    print(f"training seconds: {train_seconds:.2f}")
    print(
        "inference ms/sample: "
        f"{result['performance']['test_inference_ms_per_sample']:.6f}"
    )

    return result


# ======================================================================
# ABLATION STUDY
# ======================================================================

def run_single_ablation(
    name: str,
    switches: dict[str, bool],
    bundle,
) -> dict[str, Any]:
    """Run one controlled ablation configuration."""
    print()
    print("=" * 72)
    print(f"ABLATION: {name}")
    print("=" * 72)

    for module, enabled in switches.items():
        print(f"{module}: {enabled}")

    model = build_model(switches)

    print("\n[fit]")
    train_seconds = fit_model(model, bundle)
    print(f"training time: {train_seconds:.2f}s")

    print("\n[predict]")
    probabilities, predictions, inference_seconds = predict_model(model, bundle)

    metrics, cm = pipeline_metrics(
        predictions,
        probabilities,
        bundle.y_test,
    )

    threshold_stats = (
        model.validation_threshold_statistics_
        if switches["use_dynamic_threshold"]
        else None
    )

    record = {
        "ablation": name,
        **switches,
        "n_input_features": model.n_input_features_,
        "n_model_features": model.n_model_features_,
        "n_trees": model.n_trees,
        "scale_pos_weight": model.scale_pos_weight_,
        **metrics,
        "tp": cm["true_positive"],
        "fp": cm["false_positive"],
        "fn": cm["false_negative"],
        "tn": cm["true_negative"],
        "threshold_statistics": threshold_stats,
        "train_seconds": float(train_seconds),
        "test_inference_ms": float(inference_seconds * 1000),
    }

    print_metrics(metrics)

    return record


def run_ablation_experiment(bundle) -> list[dict[str, Any]]:
    """Run all controlled Saurabh methodology ablations."""
    print()
    print("=" * 72)
    print("SAURABH ABLATION EXPERIMENT")
    print("=" * 72)

    results = []

    for name, switches in ABLATIONS.items():
        results.append(
            run_single_ablation(name, switches, bundle)
        )

    return results


# ======================================================================
# SHAP DRIFT-ONLY EXPERIMENT
# ======================================================================

def run_shap_drift_experiment(bundle) -> dict[str, Any]:
    """
    Train full methodology and report detailed SHAP drift statistics.

    Policy:
        - SHAP reference ranking: training data only
        - Drift detection: test features only
        - Test labels never used
        - Windows respect temporal block boundaries
    """
    print()
    print("=" * 72)
    print("SAURABH SHAP DRIFT EXPERIMENT")
    print("=" * 72)

    model = build_model(FULL_CONFIG)

    print("\n[fit]")
    train_seconds = fit_model(model, bundle)
    print(f"training time: {train_seconds:.2f}s")

    print("\n[drift detection]")
    drift_result, drift_seconds = run_drift_detection(model, bundle)

    summary = summarize_drift(drift_result, drift_seconds)

    record = {
        "experiment": "saurabh_shap_drift",
        "train_seconds": float(train_seconds),
        "drift_seconds": float(drift_seconds),
        "drift_threshold": float(model.shap_drift_threshold),
        "n_windows": summary["n_windows"],
        "n_drift_windows": summary["drift_windows"],
        "drift_rate": summary["drift_rate"],
        "tau_min": summary["min_kendall_tau"],
        "tau_mean": summary["mean_kendall_tau"],
        "tau_max": summary["max_kendall_tau"],
        "tau_std": summary["std_kendall_tau"],
        "tau_values": summary["tau_values"],
        "drift_flags": summary["drift_flags"],
    }

    print("\n[results]")
    print("windows:", record["n_windows"])
    print("drift windows:", record["n_drift_windows"])
    print(f"drift rate: {record['drift_rate']:.4f}")
    print(f"tau min : {record['tau_min']:.4f}")
    print(f"tau mean: {record['tau_mean']:.4f}")
    print(f"tau max : {record['tau_max']:.4f}")

    return record


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Saurabh methodology experiment runner."
    )

    parser.add_argument(
        "--experiment",
        choices=["full", "ablation", "shap-drift", "all"],
        default="all",
        help="Experiment to run (default: all).",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="write results under results/raw/ (off by default: Member 5 "
             "owns result collection)",
    )

    return parser.parse_args()


def main() -> None:
    global SAVE_ENABLED
    args = parse_args()
    SAVE_ENABLED = args.save

    # Load once and reuse the identical frozen split for selected experiments.
    bundle = load_data()

    if args.experiment in ("full", "all"):
        result = run_full_experiment(bundle)
        save_json("saurabh_full_results.json", result)

    if args.experiment in ("ablation", "all"):
        results = run_ablation_experiment(bundle)
        save_json("saurabh_ablation_results.json", results)

    if args.experiment in ("shap-drift", "all"):
        result = run_shap_drift_experiment(bundle)
        save_json("saurabh_shap_drift_results.json", result)


if __name__ == "__main__":
    main()
