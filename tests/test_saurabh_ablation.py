"""
Smoke tests for Saurabh methodology ablation configurations.

Verifies that individual methodology modules can be enabled or disabled
without breaking the XGBoost training and prediction pipeline.
"""

from __future__ import annotations

import numpy as np

from src.common import config
from src.common.data.ncsrd_adapter import NetworkDataAdapter
from src.saurabh.saurabh_xgboost import SaurabhXGBoost


def run_configuration(
    name: str,
    use_correlation_graph: bool,
    use_dynamic_threshold: bool,
    use_shap_drift: bool,
    bundle,
):
    """Train and smoke-test one ablation configuration."""

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    train_n = min(5000, len(bundle.X_train))
    val_n = min(2000, len(bundle.X_val))
    test_n = min(2000, len(bundle.X_test))

    X_train = bundle.X_train[:train_n]
    y_train = bundle.y_train[:train_n]

    X_val = bundle.X_val[:val_n]
    y_val = bundle.y_val[:val_n]

    X_test = bundle.X_test[:test_n]

    train_blocks = (
        bundle.train_block[:train_n]
        if bundle.train_block is not None
        else None
    )

    val_blocks = (
        bundle.val_block[:val_n]
        if bundle.val_block is not None
        else None
    )

    test_blocks = (
        bundle.test_block[:test_n]
        if bundle.test_block is not None
        else None
    )

    model = SaurabhXGBoost(
        correlation_window_size=500,
        threshold_window_size=500,
        shap_window_size=1000,
        shap_background_size=100,
        shap_sample_size=50,
        seed=42,
        use_correlation_graph=use_correlation_graph,
        use_dynamic_threshold=use_dynamic_threshold,
        use_shap_drift=use_shap_drift,
    )

    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        train_block_ids=train_blocks,
        val_block_ids=val_blocks,
    )

    probabilities = model.predict_proba(
        X_test,
        block_ids=test_blocks,
    )

    predictions = model.predict(
        X_test,
        block_ids=test_blocks,
    )

    assert len(probabilities) == len(X_test)
    assert len(predictions) == len(X_test)

    assert np.isfinite(probabilities).all()
    assert np.isin(predictions, [0, 1]).all()

    expected_features = 50 if use_correlation_graph else 49

    assert model.n_model_features_ == expected_features

    if use_dynamic_threshold:
        assert model.validation_threshold_statistics_ is not None
    else:
        assert model.validation_threshold_statistics_ is None

    if use_shap_drift:
        drift = model.detect_shap_drift(
            X_test,
            block_ids=test_blocks,
        )

        assert drift.n_windows >= 1

        print(
            "SHAP windows:",
            drift.n_windows,
        )

    print(
        "model features:",
        model.n_model_features_,
    )

    print(
        "predicted attacks:",
        int(predictions.sum()),
    )

    print("PASS")


def main():
    print("=" * 70)
    print("Saurabh ablation configuration smoke test")
    print("=" * 70)

    adapter = NetworkDataAdapter.for_feature_set(
        "saurabh49"
    )

    adapter.load_split(
        config.SPLIT_INDEX_FILE
    )

    bundle = adapter.for_xgboost(
        balance="class_weight"
    )

    configurations = [
        (
            "Baseline XGBoost",
            False,
            False,
            False,
        ),
        (
            "Correlation Graph Only",
            True,
            False,
            False,
        ),
        (
            "Correlation + Dynamic Threshold",
            True,
            True,
            False,
        ),
        (
            "Full Methodology",
            True,
            True,
            True,
        ),
    ]

    for (
        name,
        use_correlation_graph,
        use_dynamic_threshold,
        use_shap_drift,
    ) in configurations:

        run_configuration(
            name=name,
            use_correlation_graph=use_correlation_graph,
            use_dynamic_threshold=use_dynamic_threshold,
            use_shap_drift=use_shap_drift,
            bundle=bundle,
        )

    print()
    print("=" * 70)
    print("ALL ABLATION CONFIGURATIONS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()