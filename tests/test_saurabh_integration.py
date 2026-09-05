"""
Integration smoke test for the complete Saurabh methodology.

This test verifies that:

1. The frozen project-standard split is loaded.
2. The SaurabhXGBoost pipeline trains successfully.
3. Correlation graph augmentation works end-to-end.
4. Dynamic thresholds are fitted ONLY on validation data.
5. Test prediction never uses test labels for threshold selection.
6. SHAP drift analysis runs after model training.

IMPORTANT:
This is a smoke/integration test, not a final experiment. Small,
reproducibly shuffled subsets are used so the model sees representative
samples rather than a potentially pathological contiguous temporal segment.
"""

from __future__ import annotations

import numpy as np

from src.common import config
from src.common.data.ncsrd_adapter import NetworkDataAdapter
from src.saurabh.saurabh_xgboost import SaurabhXGBoost


def stratified_subset_indices(
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Return a reproducible approximately class-stratified subset.

    This prevents the smoke test from accidentally selecting a contiguous
    temporal region containing only one class or an unusual distribution.
    """

    y = np.asarray(y)

    if n_samples >= len(y):
        return np.arange(len(y))

    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)

    selected = []

    for class_label, class_count in zip(classes, counts):
        class_indices = np.flatnonzero(y == class_label)

        # Allocate samples approximately proportional to class frequency.
        n_class = int(
            round(n_samples * class_count / len(y))
        )

        # Ensure every present class gets at least one sample.
        n_class = max(1, n_class)

        # Never request more than available.
        n_class = min(n_class, len(class_indices))

        chosen = rng.choice(
            class_indices,
            size=n_class,
            replace=False,
        )

        selected.append(chosen)

    indices = np.concatenate(selected)

    # Rounding can make the total slightly different from n_samples.
    if len(indices) > n_samples:
        indices = rng.choice(
            indices,
            size=n_samples,
            replace=False,
        )

    elif len(indices) < n_samples:
        remaining = np.setdiff1d(
            np.arange(len(y)),
            indices,
            assume_unique=False,
        )

        extra = rng.choice(
            remaining,
            size=n_samples - len(indices),
            replace=False,
        )

        indices = np.concatenate([indices, extra])

    # Shuffle final subset so classes are not grouped together.
    rng.shuffle(indices)

    return indices


def main():
    print("=" * 70)
    print("Saurabh end-to-end integration smoke test")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load the frozen project-standard split.
    #
    # IMPORTANT:
    # Never call split() here. All experiments must use the same
    # frozen split for fair comparison.
    # ------------------------------------------------------------

    adapter = NetworkDataAdapter.for_feature_set("saurabh49")

    adapter.load_split(config.SPLIT_INDEX_FILE)

    bundle = adapter.for_xgboost(balance="class_weight")

    print()
    print("[data]")
    print("train:", bundle.X_train.shape)
    print("val:  ", bundle.X_val.shape)
    print("test: ", bundle.X_test.shape)

    print(
        "train attacks:",
        int((bundle.y_train == 1).sum()),
    )
    print(
        "val attacks:  ",
        int((bundle.y_val == 1).sum()),
    )
    print(
        "test attacks: ",
        int((bundle.y_test == 1).sum()),
    )

    # ------------------------------------------------------------
    # Use SMALL but representative subsets for the smoke test.
    #
    # Do NOT simply take [:N], because the dataset is temporally
    # structured and a contiguous prefix may have an abnormal class
    # distribution.
    #
    # Fixed seeds make this completely reproducible.
    # ------------------------------------------------------------

    train_n = min(10000, len(bundle.X_train))
    val_n = min(5000, len(bundle.X_val))
    test_n = min(5000, len(bundle.X_test))

    train_idx = stratified_subset_indices(
        bundle.y_train,
        train_n,
        seed=42,
    )

    val_idx = stratified_subset_indices(
        bundle.y_val,
        val_n,
        seed=43,
    )

    test_idx = stratified_subset_indices(
        bundle.y_test,
        test_n,
        seed=44,
    )

    X_train = bundle.X_train[train_idx]
    y_train = bundle.y_train[train_idx]

    X_val = bundle.X_val[val_idx]
    y_val = bundle.y_val[val_idx]

    X_test = bundle.X_test[test_idx]
    y_test = bundle.y_test[test_idx]

    # ------------------------------------------------------------
    # IMPORTANT NOTE ABOUT BLOCK IDS
    #
    # Random sampling destroys temporal ordering inside blocks.
    # Therefore block IDs are intentionally not passed to this
    # shuffled smoke test.
    #
    # Block-aware behavior is tested separately in unit tests and
    # should be used in final full-data experiments.
    # ------------------------------------------------------------

    train_block = None
    val_block = None
    test_block = None

    print()
    print("[smoke subset]")

    print(
        f"train: {X_train.shape}  "
        f"benign={int((y_train == 0).sum())}  "
        f"attack={int((y_train == 1).sum())}"
    )

    print(
        f"val:   {X_val.shape}  "
        f"benign={int((y_val == 0).sum())}  "
        f"attack={int((y_val == 1).sum())}"
    )

    print(
        f"test:  {X_test.shape}  "
        f"benign={int((y_test == 0).sum())}  "
        f"attack={int((y_test == 1).sum())}"
    )

    # ------------------------------------------------------------
    # Create complete methodology pipeline.
    # ------------------------------------------------------------

    model = SaurabhXGBoost(
        correlation_window_size=500,
        threshold_window_size=500,
        shap_window_size=1000,
        shap_background_size=200,
        shap_sample_size=100,
        seed=42,
    )

    # ------------------------------------------------------------
    # Train.
    #
    # Correlation baseline -> TRAIN
    # XGBoost training -> TRAIN
    # Dynamic thresholds -> VALIDATION
    #
    # Test labels are NEVER used here.
    # ------------------------------------------------------------

    print()
    print("[fit] training complete Saurabh pipeline...")

    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        train_block_ids=train_block,
        val_block_ids=val_block,
    )

    print("[fit] complete")

    # ------------------------------------------------------------
    # Prediction.
    #
    # IMPORTANT:
    # y_test is NOT passed into prediction.
    # ------------------------------------------------------------

    print()
    print("[predict] predicting test samples...")

    probabilities = model.predict_proba(
        X_test,
        block_ids=test_block,
    )

    predictions = model.predict(
        X_test,
        block_ids=test_block,
    )

    # ------------------------------------------------------------
    # Basic integration assertions.
    # ------------------------------------------------------------

    assert len(probabilities) == len(X_test)
    assert len(predictions) == len(X_test)

    assert np.isfinite(probabilities).all()

    assert np.isin(predictions, [0, 1]).all()

    assert model.fitted_

    assert model.n_input_features_ == 49

    # Correlation module adds one feature.
    assert model.n_model_features_ == 50

    print()
    print("[pipeline]")
    print("input features: ", model.n_input_features_)
    print("model features: ", model.n_model_features_)
    print("best iteration:", model.best_iteration_)

    # ------------------------------------------------------------
    # Dynamic threshold verification.
    # ------------------------------------------------------------

    print()
    print("[dynamic threshold]")

    stats = model.validation_threshold_statistics_

    assert stats is not None

    for key, value in stats.items():
        print(f"{key}: {value}")

    assert stats["n_windows"] > 0

    # ------------------------------------------------------------
    # SHAP drift verification.
    # ------------------------------------------------------------

    print()
    print("[SHAP drift]")

    drift_result = model.detect_shap_drift(
        X_test,
        block_ids=test_block,
    )

    print("windows:", drift_result.n_windows)
    print("tau:", drift_result.tau)
    print("drift:", drift_result.drift)

    assert drift_result.n_windows >= 1

    # ------------------------------------------------------------
    # Final prediction sanity.
    # ------------------------------------------------------------

    print()
    print("[prediction distribution]")

    benign_predictions = int((predictions == 0).sum())
    attack_predictions = int((predictions == 1).sum())

    print("predicted benign:", benign_predictions)
    print("predicted attack:", attack_predictions)

    print()
    print("[probability distribution]")

    print("min: ", float(probabilities.min()))
    print("mean:", float(probabilities.mean()))
    print("max: ", float(probabilities.max()))
    print("std: ", float(probabilities.std()))

    # The probabilities should not all collapse to exactly one value.
    assert probabilities.std() > 0.0, (
        "All predicted probabilities are identical; "
        "the smoke-test model appears degenerate."
    )

    print()
    print("=" * 70)
    print("SAURABH END-TO-END INTEGRATION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()