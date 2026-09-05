"""
Tests for Saurabh methodology modules.

Currently covers:
    Module A - Correlation Behavioural Graph
"""

import numpy as np
from src.saurabh.shap_drift import SHAPDriftDetector

from src.saurabh.dynamic_threshold import DynamicAdaptiveThreshold

from src.saurabh.correlation_graph import CorrelationBehaviouralGraph


def test_fit_uses_only_benign_samples():
    """
    When labels are provided, the baseline must be built only from
    benign (label == 0) rows.
    """

    rng = np.random.default_rng(42)

    benign = rng.normal(0, 1, size=(100, 3))
    attack = rng.normal(100, 1, size=(20, 3))

    X = np.vstack([benign, attack])
    y = np.array([0] * len(benign) + [1] * len(attack))

    graph = CorrelationBehaviouralGraph(window_size=10)
    graph.fit(X, y)

    expected = graph._correlation_matrix(benign)

    assert np.allclose(
        graph.baseline_correlation_,
        expected,
    )


def test_transform_adds_one_feature():
    """Transform must append exactly graph_frob_div."""

    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))

    graph = CorrelationBehaviouralGraph(window_size=10)
    Z = graph.fit_transform(X)

    assert Z.shape == (100, 6)

    # Original features must remain unchanged.
    assert np.allclose(Z[:, :-1], X)


def test_same_window_has_same_score():
    """Every row inside one window receives the same divergence score."""

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))

    graph = CorrelationBehaviouralGraph(window_size=10)
    Z = graph.fit_transform(X)

    scores = Z[:, -1]

    assert np.allclose(scores[0:10], scores[0])
    assert np.allclose(scores[10:20], scores[10])
    assert np.allclose(scores[20:30], scores[20])


def test_windows_do_not_cross_block_boundaries():
    """
    Block-aware transformation must restart window construction at every
    block boundary.

    This specifically guards the project-standard split requirement.
    """

    rng = np.random.default_rng(42)

    # Two blocks of deliberately different distributions.
    block_1 = rng.normal(0, 1, size=(8, 3))
    block_2 = rng.normal(10, 1, size=(8, 3))

    X = np.vstack([block_1, block_2])

    block_ids = np.array(
        [0] * len(block_1) +
        [1] * len(block_2)
    )

    graph = CorrelationBehaviouralGraph(window_size=10)
    graph.fit(X)

    Z = graph.transform(X, block_ids=block_ids)

    scores = Z[:, -1]

    # Each block is shorter than window_size but >= 2,
    # so each entire block forms its own correlation window.
    assert np.allclose(scores[:8], scores[0])
    assert np.allclose(scores[8:], scores[8])

    # Different blocks should not be forced into one shared window.
    assert not np.isclose(scores[0], scores[8])


def test_block_boundary_restarts_window():
    """
    A new block must restart the window counter.

    With window_size=5:
        Block 0 -> rows 0-4, 5-7
        Block 1 -> rows 8-12, 13-15

    Windows must not continue counting across blocks.
    """

    rng = np.random.default_rng(42)
    X = rng.normal(size=(16, 4))

    block_ids = np.array(
        [0] * 8 +
        [1] * 8
    )

    graph = CorrelationBehaviouralGraph(window_size=5)
    Z = graph.fit_transform(
        X,
        block_ids=block_ids,
    )

    scores = Z[:, -1]

    # Block 0 first full window.
    assert np.allclose(scores[0:5], scores[0])

    # Block 0 remainder window.
    assert np.allclose(scores[5:8], scores[5])

    # Block 1 MUST restart at row 8.
    assert np.allclose(scores[8:13], scores[8])

    # Block 1 remainder.
    assert np.allclose(scores[13:16], scores[13])


def test_constant_columns_produce_finite_output():
    """
    Constant columns must not cause NaN graph divergence scores.
    """

    rng = np.random.default_rng(42)

    X = np.column_stack(
        [
            rng.normal(size=100),
            rng.normal(size=100),
            np.ones(100),  # constant feature
        ]
    )

    graph = CorrelationBehaviouralGraph(window_size=20)
    Z = graph.fit_transform(X)

    assert np.isfinite(Z).all()


def test_block_ids_length_must_match_X():
    """block_ids must contain exactly one value per row."""

    X = np.random.default_rng(42).normal(size=(20, 3))

    graph = CorrelationBehaviouralGraph()
    graph.fit(X)

    wrong_block_ids = np.array([0] * 10)

    try:
        graph.transform(X, block_ids=wrong_block_ids)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "block_ids" in str(exc)


def test_divergence_requires_fit():
    """Calling divergence before fit must fail."""

    X = np.random.default_rng(42).normal(size=(10, 3))

    graph = CorrelationBehaviouralGraph()

    try:
        graph.divergence(X)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass
def test_dynamic_threshold_finds_optimal_threshold():
    """F1-optimal threshold should separate an obvious dataset."""

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])

    detector = DynamicAdaptiveThreshold(
        window_size=3,
        threshold_step=0.01,
    )

    threshold, score = detector.find_optimal_threshold(
        y_true,
        y_score,
    )

    assert 0.20 < threshold <= 0.80
    assert np.isclose(score, 1.0)


def test_dynamic_threshold_fit_creates_complete_windows():
    """Only complete windows should produce learned thresholds."""

    y_true = np.array(
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    )

    y_score = np.array(
        [0.1, 0.2, 0.8, 0.9, 0.3,
         0.7, 0.2, 0.1, 0.9, 0.8, 0.2]
    )

    detector = DynamicAdaptiveThreshold(window_size=5)

    detector.fit(y_true, y_score)

    # 11 samples -> two complete windows of 5.
    assert len(detector.thresholds_) == 2
    assert len(detector.window_f1_) == 2


def test_dynamic_threshold_windows_do_not_cross_blocks():
    """Threshold windows restart independently at block boundaries."""

    # Two blocks of 6 samples each, window size 5.
    y_true = np.array([
        0, 0, 1, 1, 0, 0,
        1, 1, 0, 0, 1, 1,
    ])

    y_score = np.array([
        0.1, 0.2, 0.8, 0.9, 0.3, 0.2,
        0.9, 0.8, 0.1, 0.2, 0.9, 0.8,
    ])

    block_ids = np.array([
        10, 10, 10, 10, 10, 10,
        20, 20, 20, 20, 20, 20,
    ])

    detector = DynamicAdaptiveThreshold(window_size=5)

    detector.fit(
        y_true,
        y_score,
        block_ids=block_ids,
    )

    # One complete window from each block.
    assert len(detector.thresholds_) == 2


def test_dynamic_threshold_block_ids_length_must_match():
    """Invalid block ID length must raise an error."""

    detector = DynamicAdaptiveThreshold(window_size=2)

    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])

    try:
        detector.fit(
            y_true,
            y_score,
            block_ids=np.array([1, 1]),
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_dynamic_threshold_predict_requires_fit():
    """Prediction before fitting must fail."""

    detector = DynamicAdaptiveThreshold(window_size=2)

    try:
        detector.predict(np.array([0.2, 0.8]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_dynamic_threshold_prediction_is_binary():
    """Predictions should contain only binary values."""

    y_true = np.array([
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0,
    ])

    y_score = np.array([
        0.1, 0.2, 0.8, 0.9, 0.3,
        0.2, 0.8, 0.9, 0.1, 0.2,
    ])

    detector = DynamicAdaptiveThreshold(window_size=5)
    detector.fit(y_true, y_score)

    predictions = detector.predict(y_score)

    assert len(predictions) == len(y_score)
    assert np.isin(predictions, [0, 1]).all()


def test_dynamic_threshold_statistics():
    """Threshold statistics should reflect fitted windows."""

    y_true = np.array([
        0, 0, 1, 1,
        0, 1, 0, 1,
    ])

    y_score = np.array([
        0.1, 0.2, 0.8, 0.9,
        0.2, 0.8, 0.3, 0.9,
    ])

    detector = DynamicAdaptiveThreshold(window_size=4)
    detector.fit(y_true, y_score)

    stats = detector.threshold_statistics()

    assert stats["n_windows"] == 2
    assert 0.0 <= stats["min"] <= 1.0
    assert 0.0 <= stats["mean"] <= 1.0
    assert 0.0 <= stats["max"] <= 1.0
    assert stats["std"] >= 0.0
# ======================================================================
# MODULE C — SHAP DRIFT DETECTION
# ======================================================================

def test_shap_drift_requires_fit():
    """transform() must fail before fit()."""

    detector = SHAPDriftDetector(
        window_size=10,
        background_size=20,
        sample_size=10,
    )

    X = np.random.default_rng(42).normal(size=(20, 4))

    try:
        detector.transform(X)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_shap_drift_reference_is_finite():
    """Reference importance values must be finite."""

    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X[:100], y[:100])

    detector = SHAPDriftDetector(
        window_size=20,
        background_size=50,
        sample_size=20,
    )

    detector.fit(model, X[:100])

    assert detector.reference_importance_ is not None
    assert np.isfinite(detector.reference_importance_).all()
    assert len(detector.reference_importance_) == X.shape[1]


def test_shap_drift_identical_distribution_no_drift():
    """Same distribution should produce a high Kendall tau."""

    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)

    X = rng.normal(size=(1000, 5))
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)

    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X[:500], y[:500])

    detector = SHAPDriftDetector(
        window_size=200,
        background_size=100,
        sample_size=100,
        drift_threshold=0.5,
    )

    detector.fit(model, X[:500])

    result = detector.transform(X[500:])

    assert result.n_windows > 0
    assert np.isfinite(result.tau).all()

    # Same generating distribution should generally preserve ranking.
    assert np.mean(result.tau) > 0.0


def test_shap_drift_window_count():
    """Only complete windows should be evaluated."""

    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)

    X = rng.normal(size=(500, 4))
    y = (X[:, 0] > 0).astype(int)

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X[:200], y[:200])

    detector = SHAPDriftDetector(
        window_size=100,
        background_size=50,
        sample_size=50,
    )

    detector.fit(model, X[:200])

    # 250 rows -> 2 complete windows, 50 remainder ignored.
    result = detector.transform(X[200:450])

    assert result.n_windows == 2
    assert len(result.tau) == 2
    assert len(result.drift) == 2


def test_shap_drift_binary_flags():
    """Drift flags must contain boolean values."""

    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)

    X = rng.normal(size=(400, 4))
    y = (X[:, 0] > 0).astype(int)

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X[:200], y[:200])

    detector = SHAPDriftDetector(
        window_size=100,
        background_size=50,
        sample_size=50,
    )

    detector.fit(model, X[:200])

    result = detector.transform(X[200:])

    assert result.drift.dtype == bool


def test_shap_drift_feature_dimension_must_match():
    """Transform data must have same feature count as reference."""

    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X[:100], y[:100])

    detector = SHAPDriftDetector(
        window_size=20,
        background_size=50,
        sample_size=20,
    )

    detector.fit(model, X[:100])

    bad_X = rng.normal(size=(100, 5))

    try:
        detector.transform(bad_X)
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    print("=" * 70)
    print("Saurabh methodology checks")
    print("=" * 70)

    tests = [
        # Module A — Correlation Behavioural Graph
        ("benign samples only for baseline",
         test_fit_uses_only_benign_samples),
        ("transform adds one graph feature",
         test_transform_adds_one_feature),
        ("same window has same score",
         test_same_window_has_same_score),
        ("windows do not cross block boundaries",
         test_windows_do_not_cross_block_boundaries),
        ("block boundary restarts window",
         test_block_boundary_restarts_window),
        ("constant columns produce finite output",
         test_constant_columns_produce_finite_output),
        ("block_ids length must match X",
         test_block_ids_length_must_match_X),
        ("divergence requires fit",
         test_divergence_requires_fit),

         # Module B — Dynamic Adaptive Threshold
        ("dynamic threshold finds optimal threshold",
         test_dynamic_threshold_finds_optimal_threshold),
        ("dynamic threshold creates complete windows",
         test_dynamic_threshold_fit_creates_complete_windows),
        ("dynamic threshold windows do not cross blocks",
         test_dynamic_threshold_windows_do_not_cross_blocks),
        ("dynamic threshold block IDs length must match",
         test_dynamic_threshold_block_ids_length_must_match),
        ("dynamic threshold predict requires fit",
         test_dynamic_threshold_predict_requires_fit),
        ("dynamic threshold prediction is binary",
         test_dynamic_threshold_prediction_is_binary),
        ("dynamic threshold statistics are valid",
         test_dynamic_threshold_statistics),
	
        ("SHAP drift requires fit",
         test_shap_drift_requires_fit),
        ("SHAP reference importance is finite",
         test_shap_drift_reference_is_finite),
        ("SHAP same distribution has stable ranking",
         test_shap_drift_identical_distribution_no_drift),
        ("SHAP drift counts complete windows",
         test_shap_drift_window_count),
        ("SHAP drift flags are boolean",
         test_shap_drift_binary_flags),
        ("SHAP feature dimension must match",
         test_shap_drift_feature_dimension_must_match),
    ]

    passed = 0
    failed = 0

    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    print()
    print("=" * 70)
    print(f"{passed} passed, {failed} failed")
    print("=" * 70)

    if failed:
        raise SystemExit(1)