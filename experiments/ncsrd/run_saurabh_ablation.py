"""
Saurabh methodology ablation experiments.

Runs controlled ablations on the frozen project-standard split.

Configurations:
    full               A + B + C
    no_correlation     B + C
    no_dynamic         A + C
    no_shap            A + B
    xgboost_only       XGBoost baseline using Saurabh49 features

All configurations use:
    - identical frozen split
    - identical seed
    - identical XGBoost parameters
    - validation-only threshold fitting where enabled
    - untouched test set for final evaluation
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


ABLATIONS = {
    "full": {
        "use_correlation_graph": True,
        "use_dynamic_threshold": True,
        "use_shap_drift": True,
    },
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
}


def run_ablation(
    name: str,
    switches: dict,
    bundle,
) -> dict:

    print()
    print("=" * 72)
    print(f"ABLATION: {name}")
    print("=" * 72)

    print("correlation graph:", switches["use_correlation_graph"])
    print("dynamic threshold:", switches["use_dynamic_threshold"])
    print("SHAP drift:", switches["use_shap_drift"])

    model = SaurabhXGBoost(
        seed=config.SEED,
        use_correlation_graph=switches["use_correlation_graph"],
        use_dynamic_threshold=switches["use_dynamic_threshold"],
        use_shap_drift=switches["use_shap_drift"],
    )

    print("\n[fit]")

    start = time.perf_counter()

    model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        train_block_ids=bundle.train_block,
        val_block_ids=bundle.val_block,
    )

    train_seconds = time.perf_counter() - start

    print(f"training time: {train_seconds:.2f}s")

    print("\n[predict]")

    start = time.perf_counter()

    probabilities = model.predict_proba(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    predictions = model.predict(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    inference_ms = (time.perf_counter() - start) * 1000

    if switches["use_dynamic_threshold"]:
        threshold_stats = model.validation_threshold_statistics_

        threshold = float(
            threshold_stats["mean"]
        )
    else:
        threshold_stats = None
        threshold = 0.5

    metrics = evaluate(
        bundle.y_test,
        probabilities,
        threshold,
    )

    # Use actual predictions from the pipeline for confusion matrix
    # because dynamic thresholding may vary by window.
    tp = int(
        ((predictions == 1) &
         (bundle.y_test == 1)).sum()
    )

    fp = int(
        ((predictions == 1) &
         (bundle.y_test == 0)).sum()
    )

    fn = int(
        ((predictions == 0) &
         (bundle.y_test == 1)).sum()
    )

    tn = int(
        ((predictions == 0) &
         (bundle.y_test == 0)).sum()
    )

    record = {
        "ablation": name,

        "use_correlation_graph":
            switches["use_correlation_graph"],

        "use_dynamic_threshold":
            switches["use_dynamic_threshold"],

        "use_shap_drift":
            switches["use_shap_drift"],

        "n_input_features":
            model.n_input_features_,

        "n_model_features":
            model.n_model_features_,

        "n_trees":
            model.n_trees,

        "scale_pos_weight":
            model.scale_pos_weight_,

        "accuracy":
            float(metrics["accuracy"]),

        "precision":
            float(metrics["precision"]),

        "recall":
            float(metrics["recall"]),

        "f1":
            float(metrics["f1"]),

        "weighted_f1":
            float(metrics["weighted_f1"]),

        "roc_auc":
            float(metrics["roc_auc"]),

        "fpr":
            float(metrics["fpr"]),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "threshold_statistics":
            threshold_stats,

        "train_seconds":
            float(train_seconds),

        "test_inference_ms":
            float(inference_ms),
    }

    print()
    print("[results]")
    print(f"accuracy : {record['accuracy']:.6f}")
    print(f"precision: {record['precision']:.6f}")
    print(f"recall   : {record['recall']:.6f}")
    print(f"f1       : {record['f1']:.6f}")
    print(f"roc_auc  : {record['roc_auc']:.6f}")
    print(f"fpr      : {record['fpr']:.6f}")

    return record


def main():

    print("=" * 72)
    print("SAURABH ABLATION EXPERIMENT")
    print("=" * 72)

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

    results = []

    for name, switches in ABLATIONS.items():

        result = run_ablation(
            name,
            switches,
            bundle,
        )

        results.append(result)

    output_dir = ROOT / "results" / "raw"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        "saurabh_ablation_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            default=float,
        )

    print()
    print("=" * 72)
    print("ABLATION COMPLETE")
    print("=" * 72)

    print(
        f"saved -> "
        f"{output_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()