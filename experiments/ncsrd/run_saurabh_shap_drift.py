"""
Saurabh Methodology - Module C SHAP Drift Experiment.

Trains the full Saurabh methodology on the frozen project-standard split,
then performs SHAP feature-importance drift detection on the untouched
test set.

Policy:
    - SHAP reference ranking: training data only
    - Drift detection: test features only
    - Test labels are never used for drift detection
    - Windows respect temporal block boundaries
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
from src.saurabh.saurabh_xgboost import SaurabhXGBoost


def main() -> None:

    print("=" * 72)
    print("SAURABH SHAP DRIFT EXPERIMENT")
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
    # Train full methodology
    # --------------------------------------------------------------

    print()
    print("[fit]")

    model = SaurabhXGBoost(
        seed=config.SEED,
        use_correlation_graph=True,
        use_dynamic_threshold=True,
        use_shap_drift=True,
    )

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

    # --------------------------------------------------------------
    # Run SHAP drift detection
    # --------------------------------------------------------------

    print()
    print("[drift detection]")

    start = time.perf_counter()

    drift_result = model.detect_drift(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    drift_seconds = time.perf_counter() - start

    tau = drift_result.tau
    drift = drift_result.drift

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    record = {
        "experiment": "saurabh_shap_drift",

        "train_seconds": float(train_seconds),

        "drift_seconds": float(drift_seconds),

        "n_windows": int(drift_result.n_windows),

        "drift_threshold": float(
            model.shap_drift_threshold
        ),

        "n_drift_windows": int(
            drift.sum()
        ),

        "drift_rate": float(
            drift.mean()
        ) if len(drift) > 0 else 0.0,

        "tau_min": float(
            tau.min()
        ) if len(tau) > 0 else None,

        "tau_mean": float(
            tau.mean()
        ) if len(tau) > 0 else None,

        "tau_max": float(
            tau.max()
        ) if len(tau) > 0 else None,

        "tau_std": float(
            tau.std()
        ) if len(tau) > 0 else None,

        "tau_values": [
            float(value)
            for value in tau
        ],

        "drift_flags": [
            bool(value)
            for value in drift
        ],
    }

    print()
    print("[results]")
    print(
        "windows:",
        record["n_windows"],
    )
    print(
        "drift windows:",
        record["n_drift_windows"],
    )
    print(
        f"drift rate: "
        f"{record['drift_rate']:.4f}"
    )
    print(
        f"tau min : "
        f"{record['tau_min']:.4f}"
    )
    print(
        f"tau mean: "
        f"{record['tau_mean']:.4f}"
    )
    print(
        f"tau max : "
        f"{record['tau_max']:.4f}"
    )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    output_dir = ROOT / "results" / "raw"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        "saurabh_shap_drift_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            record,
            f,
            indent=2,
        )

    print()
    print(
        "saved ->",
        output_path.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()