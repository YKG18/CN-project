"""
Full Saurabh methodology experiment.

Runs the complete A + B + C methodology on the frozen
project-standard NSCRD split and records:

- Classification metrics
- Confusion matrix
- ROC-AUC
- Training time
- Inference time
- Model configuration
- Dynamic threshold statistics
- SHAP drift statistics

The test set remains untouched until final evaluation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import config
from src.common.data.ncsrd_adapter import NetworkDataAdapter
from src.base.base_xgboost import evaluate
from src.saurabh.saurabh_xgboost import SaurabhXGBoost


def main() -> None:

    print("=" * 72)
    print("SAURABH FULL DATASET EXPERIMENT")
    print("=" * 72)

    # --------------------------------------------------------------
    # Load frozen project-standard split
    # --------------------------------------------------------------

    adapter = NetworkDataAdapter.for_feature_set(
        "saurabh49"
    )

    adapter.load_split(
        config.SPLIT_INDEX_FILE
    )

    bundle = adapter.for_xgboost(
        balance="class_weight"
    )

    print()
    print("[data]")
    print("train:", bundle.X_train.shape)
    print("val:  ", bundle.X_val.shape)
    print("test: ", bundle.X_test.shape)

    # --------------------------------------------------------------
    # Full methodology: A + B + C
    # --------------------------------------------------------------

    model = SaurabhXGBoost(
        seed=config.SEED,
        use_correlation_graph=True,
        use_dynamic_threshold=True,
        use_shap_drift=True,
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    print()
    print("[training]")

    start = time.perf_counter()

    model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        train_block_ids=bundle.train_block,
        val_block_ids=bundle.val_block,
    )

    train_seconds = (
        time.perf_counter() - start
    )

    print(
        f"training time: {train_seconds:.2f}s"
    )

    # --------------------------------------------------------------
    # Test inference
    # --------------------------------------------------------------

    print()
    print("[test prediction]")

    start = time.perf_counter()

    probabilities = model.predict_proba(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    predictions = model.predict(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    inference_seconds = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------------
    # Classification metrics
    #
    # Metrics are calculated directly from pipeline predictions.
    # This is important because Module B uses per-window thresholds.
    # --------------------------------------------------------------

    tp = int(
        (
            (predictions == 1)
            & (bundle.y_test == 1)
        ).sum()
    )

    fp = int(
        (
            (predictions == 1)
            & (bundle.y_test == 0)
        ).sum()
    )

    fn = int(
        (
            (predictions == 0)
            & (bundle.y_test == 1)
        ).sum()
    )

    tn = int(
        (
            (predictions == 0)
            & (bundle.y_test == 0)
        ).sum()
    )

    accuracy = (
        (tp + tn)
        / len(bundle.y_test)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    # ROC-AUC uses probabilities, not thresholded predictions.
    base_metrics = evaluate(
        bundle.y_test,
        probabilities,
        0.5,
    )

    roc_auc = float(
        base_metrics["roc_auc"]
    )

    # --------------------------------------------------------------
    # SHAP drift analysis
    # --------------------------------------------------------------

    print()
    print("[SHAP drift analysis]")

    start = time.perf_counter()

    drift_result = model.detect_shap_drift(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    drift_seconds = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------------
    # Extract drift results defensively
    # --------------------------------------------------------------

    drift_summary = {
        "enabled": True,
        "analysis_seconds": float(drift_seconds),
    }

    tau = np.asarray(
        drift_result.tau,
        dtype=float,
    )

    flags = np.asarray(
        drift_result.drift,
        dtype=bool,
    )

    drift_summary["mean_kendall_tau"] = (
        float(tau.mean())
        if len(tau) > 0
        else None
    )

    drift_summary["min_kendall_tau"] = (
        float(tau.min())
        if len(tau) > 0
        else None
    )

    drift_summary["max_kendall_tau"] = (
        float(tau.max())
        if len(tau) > 0
        else None
    )

    drift_summary["n_windows"] = int(
        drift_result.n_windows
    )

    drift_summary["drift_windows"] = int(
        flags.sum()
    )

    drift_summary["drift_rate"] = (
        float(flags.mean())
        if len(flags) > 0
        else 0.0
    )
    # --------------------------------------------------------------
    # Build experiment record
    # --------------------------------------------------------------

    result = {
        "experiment": "saurabh_full",

        "dataset": "ncsrd",

        "feature_set": "saurabh49",

        "seed": int(config.SEED),

        "split_file": str(
            config.SPLIT_INDEX_FILE
        ),

        "samples": {
            "train": int(
                len(bundle.y_train)
            ),
            "validation": int(
                len(bundle.y_val)
            ),
            "test": int(
                len(bundle.y_test)
            ),
        },

        "model": model.summary(),

        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "false_positive_rate": float(fpr),
        },

        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },

        "performance": {
            "training_seconds": float(
                train_seconds
            ),

            "test_inference_seconds": float(
                inference_seconds
            ),

            "test_inference_ms_per_sample": float(
                inference_seconds
                * 1000
                / len(bundle.y_test)
            ),

            "shap_drift_seconds": float(
                drift_seconds
            ),
        },

        "shap_drift": drift_summary,
    }

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("FULL EXPERIMENT RESULTS")
    print("=" * 72)

    for name, value in result["metrics"].items():
        print(
            f"{name:24s}: {value:.6f}"
        )

    print()
    print("[confusion matrix]")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)

    print()
    print("[performance]")
    print(
        f"training seconds: "
        f"{train_seconds:.2f}"
    )

    print(
        f"inference ms/sample: "
        f"{result['performance']['test_inference_ms_per_sample']:.6f}"
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output_dir = (
        ROOT
        / "results"
        / "raw"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "saurabh_full_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            default=float,
        )

    print()
    print(
        f"saved -> "
        f"{output_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()