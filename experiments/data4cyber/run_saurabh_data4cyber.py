"""
Run the full Saurabh methodology on the Data4Cyber dataset.

Experiment pipeline
-------------------
1. Load the project's frozen Data4Cyber split.
2. Train SaurabhXGBoost:
      Module A - Correlation Behavioural Graph
      Module B - Dynamic Adaptive Threshold
      Module C - SHAP Drift Detection reference
3. Evaluate classification performance on the untouched test split.
4. Run SHAP drift detection on test windows.
5. Save reproducible experiment results.

Leakage policy
--------------
* Training data trains the correlation baseline and XGBoost model.
* Validation data is used for early stopping and threshold learning.
* Test labels are used ONLY for final evaluation metrics.
* SHAP drift detection uses training rows as its reference and does not
  use test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data.data4cyber_adapter import Data4CyberAdapter  # noqa: E402
from saurabh.saurabh_xgboost import SaurabhXGBoost  # noqa: E402


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def to_python(value: Any) -> Any:
    """Convert NumPy values recursively into JSON-safe Python values."""

    if isinstance(value, dict):
        return {
            str(key): to_python(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_python(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Calculate final binary classification metrics."""

    metrics: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }

    # ROC-AUC and PR-AUC require both classes.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    return metrics


def save_drift_csv(
    path: Path,
    tau: np.ndarray,
    drift: np.ndarray,
) -> None:
    """Save window-level SHAP drift results."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "window",
            "kendall_tau",
            "drift_detected",
        ])

        for index, (tau_value, drift_value) in enumerate(
            zip(tau, drift),
            start=1,
        ):
            writer.writerow([
                index,
                float(tau_value),
                bool(drift_value),
            ])


def print_metrics(
    metrics: dict[str, Any],
) -> None:
    """Print classification metrics neatly."""

    print("\nCLASSIFICATION RESULTS")
    print("-" * 60)

    print(
        f"Accuracy : {metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: {metrics['precision']:.6f}"
    )

    print(
        f"Recall   : {metrics['recall']:.6f}"
    )

    print(
        f"F1-score : {metrics['f1_score']:.6f}"
    )

    if metrics["roc_auc"] is not None:
        print(
            f"ROC-AUC  : {metrics['roc_auc']:.6f}"
        )

    if metrics["pr_auc"] is not None:
        print(
            f"PR-AUC   : {metrics['pr_auc']:.6f}"
        )

    cm = metrics["confusion_matrix"]

    print("\nCONFUSION MATRIX")
    print("-" * 60)

    print(
        f"TN={cm['tn']}  FP={cm['fp']}"
    )

    print(
        f"FN={cm['fn']}  TP={cm['tp']}"
    )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_experiment(
    split_mode: str,
    output_dir: Path,
    seed: int,
    shap_window_size: int,
    shap_background_size: int,
    shap_sample_size: int,
) -> dict[str, Any]:
    """
    Execute one complete Saurabh Data4Cyber experiment.
    """

    print("=" * 72)
    print("SAURABH METHODOLOGY - DATA4CYBER EXPERIMENT")
    print("=" * 72)

    # -------------------------------------------------------------
    # Load frozen split
    # -------------------------------------------------------------

    print("\n[1/4] Loading Data4Cyber frozen split...")

    adapter = Data4CyberAdapter(
        split_mode=split_mode,
        verbose=True,
    )

    bundle = adapter.for_xgboost(
        balance="class_weight",
    )

    train_blocks = bundle.meta["train_block"]
    val_blocks = bundle.meta["val_block"]
    test_blocks = bundle.meta["test_block"]

    print(
        f"Train      : {bundle.X_train.shape}"
    )

    print(
        f"Validation : {bundle.X_val.shape}"
    )

    print(
        f"Test       : {bundle.X_test.shape}"
    )

    print(
        f"Features   : {len(bundle.feature_names)}"
    )

    # -------------------------------------------------------------
    # Train full methodology
    # -------------------------------------------------------------

    print("\n[2/4] Training full Saurabh methodology...")

    model = SaurabhXGBoost(
        correlation_window_size=500,
        threshold_window_size=500,

        shap_window_size=shap_window_size,
        shap_background_size=shap_background_size,
        shap_sample_size=shap_sample_size,
        shap_drift_threshold=0.7,

        seed=seed,
        early_stopping_rounds=50,

        use_correlation_graph=True,
        use_dynamic_threshold=True,
        use_shap_drift=True,
    )

    model.fit(
        bundle.X_train,
        bundle.y_train,

        bundle.X_val,
        bundle.y_val,

        train_block_ids=train_blocks,
        val_block_ids=val_blocks,
    )

    print("Training complete.")

    summary = model.summary()

    print(
        f"Best iteration: {summary['best_iteration']}"
    )

    print(
        f"Effective trees: {summary['n_trees']}"
    )

    print(
        f"Model features: {summary['model_features']}"
    )

    # -------------------------------------------------------------
    # Test evaluation
    # -------------------------------------------------------------

    print("\n[3/4] Evaluating on untouched test split...")

    probabilities = model.predict_proba(
        bundle.X_test,
        block_ids=test_blocks,
    )

    predictions = model.predict(
        bundle.X_test,
        block_ids=test_blocks,
    )

    metrics = classification_metrics(
        bundle.y_test,
        predictions,
        probabilities,
    )

    print_metrics(metrics)

    # -------------------------------------------------------------
    # SHAP drift detection
    # -------------------------------------------------------------

    print("\n[4/4] Running SHAP drift detection...")

    drift_result = model.detect_drift(
        bundle.X_test,
        block_ids=test_blocks,
    )

    if drift_result.n_windows > 0:

        tau_mean = float(
            np.mean(drift_result.tau)
        )

        tau_min = float(
            np.min(drift_result.tau)
        )

        tau_max = float(
            np.max(drift_result.tau)
        )

        drifted_windows = int(
            np.sum(drift_result.drift)
        )

        drift_rate = float(
            np.mean(drift_result.drift)
        )

    else:

        tau_mean = None
        tau_min = None
        tau_max = None

        drifted_windows = 0
        drift_rate = 0.0

    drift_summary = {
        "n_windows": int(
            drift_result.n_windows
        ),
        "mean_kendall_tau": tau_mean,
        "min_kendall_tau": tau_min,
        "max_kendall_tau": tau_max,
        "drifted_windows": drifted_windows,
        "drift_rate": drift_rate,
        "drift_threshold": (
            model.shap_drift_threshold
        ),
    }

    print(
        f"Windows analysed : "
        f"{drift_summary['n_windows']}"
    )

    print(
        f"Mean Kendall Tau : "
        f"{drift_summary['mean_kendall_tau']}"
    )

    print(
        f"Drifted windows  : "
        f"{drift_summary['drifted_windows']}"
    )

    print(
        f"Drift rate       : "
        f"{drift_summary['drift_rate']:.4f}"
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    print("\nSaving results...")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment = {
        "experiment": {
            "name": (
                "Saurabh methodology "
                "on Data4Cyber"
            ),
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "split_mode": split_mode,
            "seed": seed,
        },

        "dataset": {
            "n_features": int(
                len(bundle.feature_names)
            ),

            "train_samples": int(
                len(bundle.y_train)
            ),

            "validation_samples": int(
                len(bundle.y_val)
            ),

            "test_samples": int(
                len(bundle.y_test)
            ),

            "train_class_distribution": np.bincount(
                bundle.y_train.astype(int),
                minlength=2,
            ).tolist(),

            "validation_class_distribution": np.bincount(
                bundle.y_val.astype(int),
                minlength=2,
            ).tolist(),

            "test_class_distribution": np.bincount(
                bundle.y_test.astype(int),
                minlength=2,
            ).tolist(),
        },

        "model": summary,

        "classification_metrics": metrics,

        "shap_drift": drift_summary,
    }

    metrics_path = (
        output_dir / "metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            to_python(experiment),
            indent=2,
        ),
        encoding="utf-8",
    )

    drift_path = (
        output_dir / "shap_drift.csv"
    )

    save_drift_csv(
        drift_path,
        drift_result.tau,
        drift_result.drift,
    )

    # Save confusion matrix separately for easy reporting.
    cm_path = (
        output_dir / "confusion_matrix.csv"
    )

    cm = metrics["confusion_matrix"]

    with cm_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "",
            "Predicted_Benign",
            "Predicted_Attack",
        ])

        writer.writerow([
            "Actual_Benign",
            cm["tn"],
            cm["fp"],
        ])

        writer.writerow([
            "Actual_Attack",
            cm["fn"],
            cm["tp"],
        ])

    print(
        f"\nmetrics.json          -> {metrics_path}"
    )

    print(
        f"shap_drift.csv        -> {drift_path}"
    )

    print(
        f"confusion_matrix.csv  -> {cm_path}"
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 72)

    return experiment


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Run the Saurabh methodology "
            "on Data4Cyber."
        )
    )

    parser.add_argument(
        "--split-mode",
        choices=[
            "block",
            "scenario",
        ],
        default="block",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/saurabh_data4cyber"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--shap-window-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--shap-background-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    run_experiment(
        split_mode=args.split_mode,
        output_dir=args.output_dir,
        seed=args.seed,
        shap_window_size=args.shap_window_size,
        shap_background_size=args.shap_background_size,
        shap_sample_size=args.shap_sample_size,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())