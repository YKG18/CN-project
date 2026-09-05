"""
Saurabh Methodology - Integrated XGBoost Pipeline for Data4Cyber.

Single-file implementation combining:

Module A — Correlation Behavioural Graph
    Builds a static benign Pearson-correlation baseline from TRAINING data
    and appends graph_frob_div as an additional feature.

Module B — Dynamic Adaptive Threshold
    Learns classification thresholds from VALIDATION probabilities and labels
    only. Test labels are never used for threshold selection.

Module C — SHAP Drift Detection
    Learns a reference SHAP feature-importance ranking from TRAINING data
    and detects ranking drift in later windows without using their labels.

IMPORTANT SPLIT POLICY
----------------------
This model NEVER creates train/validation/test splits. The caller must provide
the frozen Data4Cyber splits. Block identifiers can be supplied so temporal
windows never combine rows from different contiguous blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import f1_score


# =====================================================================
# MODULE A — CORRELATION BEHAVIOURAL GRAPH
# =====================================================================

class CorrelationBehaviouralGraph:
    """
    Static correlation behavioural graph.

    Workflow:
        1. Fit a Pearson correlation baseline using benign training samples.
        2. Compute a correlation matrix for each incoming window.
        3. Measure divergence from the baseline using Frobenius norm.
        4. Assign the resulting graph_frob_div score to every row in the
           corresponding window.

    Parameters
    ----------
    window_size : int, default=500
        Number of samples used to construct each correlation window.
        Project default is 500.

    Notes
    -----
    When block_ids are supplied to transform(), windows are constructed
    independently inside each contiguous block. This is required by the
    project-standard split policy because split rows may originate from blocks
    separated by large gaps in time.
    """

    def __init__(self, window_size: int = 500):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        self.window_size = window_size
        self.baseline_correlation_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "CorrelationBehaviouralGraph":
        """
        Fit the static baseline correlation matrix.

        If labels are supplied, only benign samples (label == 0) are used.
        If labels are omitted, all supplied samples are treated as benign.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        y : np.ndarray | None
            Binary labels where:
                0 = benign
                1 = attack

        Returns
        -------
        CorrelationBehaviouralGraph
            Fitted instance.
        """

        X = self._validate_X(X)

        if y is not None:
            y = np.asarray(y).reshape(-1)

            if len(y) != len(X):
                raise ValueError(
                    "X and y must contain the same number of samples"
                )

            benign_mask = y == 0
            X_baseline = X[benign_mask]

            if len(X_baseline) < 2:
                raise ValueError(
                    "At least two benign samples are required "
                    "to construct the correlation baseline"
                )
        else:
            X_baseline = X

        self.baseline_correlation_ = self._correlation_matrix(X_baseline)
        self.n_features_ = X.shape[1]

        return self

    def divergence(self, X_window: np.ndarray) -> float:
        """
        Calculate Frobenius divergence between a window correlation matrix
        and the fitted benign baseline.

        Parameters
        ----------
        X_window : np.ndarray
            Window of shape (n_samples, n_features).

        Returns
        -------
        float
            Frobenius norm divergence score.
        """

        self._check_fitted()

        X_window = self._validate_X(X_window)

        if X_window.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, "
                f"received {X_window.shape[1]}"
            )

        if len(X_window) < 2:
            raise ValueError(
                "At least two samples are required "
                "to calculate a correlation matrix"
            )

        window_correlation = self._correlation_matrix(X_window)

        return float(
            np.linalg.norm(
                window_correlation - self.baseline_correlation_,
                ord="fro",
            )
        )

    def transform(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute graph_frob_div and append it as one additional feature.

        Windows are fixed and non-overlapping.

        Without block_ids:
            The entire input is treated as one continuous sequence.

        With block_ids:
            Each block is processed independently. Windows never cross block
            boundaries. This is required for project-standard evaluation.

        Each row inside the same window receives the same divergence score.

        A final partial window is evaluated if it contains at least two rows.
        A one-row remainder reuses the previous score within that block.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        block_ids : np.ndarray | None
            Block identifier for each row.

            Must have length n_samples. Rows belonging to different blocks are
            never placed in the same correlation window.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_samples, n_features + 1), where the final
            column is graph_frob_div.
        """

        self._check_fitted()

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, "
                f"received {X.shape[1]}"
            )

        n_samples = len(X)
        scores = np.zeros(n_samples, dtype=np.float64)

        # Backwards-compatible behaviour:
        # treat the entire input as one continuous block.
        if block_ids is None:
            self._transform_block(X, scores, np.arange(n_samples))
        else:
            block_ids = np.asarray(block_ids).reshape(-1)

            if len(block_ids) != n_samples:
                raise ValueError(
                    "block_ids must contain one identifier per sample: "
                    f"expected {n_samples}, received {len(block_ids)}"
                )

            # np.unique returns deterministic block order.
            # DataBundle rows are already ordered by block and then time.
            for block_id in np.unique(block_ids):
                rows = np.flatnonzero(block_ids == block_id)

                self._transform_block(X, scores, rows)

        return np.column_stack((X, scores))

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit the benign correlation baseline and transform X.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        y : np.ndarray | None
            Binary labels used to select benign baseline samples.

        block_ids : np.ndarray | None
            Optional block identifier per row. When supplied, windows remain
            inside their respective blocks.
        """

        return self.fit(X, y).transform(X, block_ids=block_ids)

    def _transform_block(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        """
        Compute divergence scores for one contiguous block.

        Windows are non-overlapping and never extend beyond `rows`.

        Blocks shorter than two rows cannot form a Pearson correlation matrix,
        so their score is left as zero.
        """

        n_rows = len(rows)

        if n_rows == 0:
            return

        previous_score = 0.0

        for start in range(0, n_rows, self.window_size):
            end = min(start + self.window_size, n_rows)

            window_rows = rows[start:end]
            window = X[window_rows]

            if len(window) >= 2:
                score = self.divergence(window)
                previous_score = score
            else:
                # A single row cannot form a Pearson correlation matrix.
                # Reuse the previous window's score within THIS block only.
                score = previous_score

            scores[window_rows] = score

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
        Validate and convert input to a 2D floating-point array.
        """

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array of shape "
                "(n_samples, n_features)"
            )

        if len(X) < 2:
            raise ValueError(
                "At least two samples are required"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values"
            )

        return X

    @staticmethod
    def _correlation_matrix(X: np.ndarray) -> np.ndarray:
        """
        Calculate a stable Pearson correlation matrix.

        Constant columns have undefined Pearson correlations and NumPy emits
        RuntimeWarnings / NaNs for them.

        Undefined correlations are replaced with zero, while every diagonal
        entry is explicitly set to one.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            correlation = np.corrcoef(
                X,
                rowvar=False,
            )

        correlation = np.nan_to_num(
            correlation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        np.fill_diagonal(correlation, 1.0)

        return correlation

    def _check_fitted(self) -> None:
        """
        Raise an error if fit() has not been called.
        """

        if self.baseline_correlation_ is None:
            raise RuntimeError(
                "CorrelationBehaviouralGraph is not fitted. "
                "Call fit() before transform() or divergence()."
            )


# =====================================================================
# MODULE B — DYNAMIC ADAPTIVE THRESHOLD
# =====================================================================

class DynamicAdaptiveThreshold:
    """
    Dynamic adaptive classification threshold.

    A global F1-optimal threshold is first learned from the complete
    validation set. Per-window thresholds are then learned only for
    validation windows containing both classes.

    Windows containing only one class fall back to the global threshold.
    This prevents degenerate thresholds such as 0.0 from being selected
    simply because predicting every sample as positive maximizes F1 in
    an all-positive or class-missing window.
    """

    def __init__(
        self,
        window_size: int = 500,
        threshold_min: float = 0.01,
        threshold_max: float = 0.99,
        threshold_step: float = 0.01,
    ) -> None:
        if window_size < 1:
            raise ValueError(
                "window_size must be at least 1"
            )

        if not 0.0 <= threshold_min <= 1.0:
            raise ValueError(
                "threshold_min must be between 0 and 1"
            )

        if not 0.0 <= threshold_max <= 1.0:
            raise ValueError(
                "threshold_max must be between 0 and 1"
            )

        if threshold_min >= threshold_max:
            raise ValueError(
                "threshold_min must be smaller than threshold_max"
            )

        if threshold_step <= 0:
            raise ValueError(
                "threshold_step must be positive"
            )

        self.window_size = window_size
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step

        self.thresholds_: np.ndarray | None = None
        self.window_f1_: np.ndarray | None = None
        self.global_threshold_: float | None = None
        self.global_f1_: float | None = None

    # ------------------------------------------------------------------
    # Threshold search
    # ------------------------------------------------------------------

    def _candidate_thresholds(self) -> np.ndarray:
        """Return valid threshold candidates."""

        return np.arange(
            self.threshold_min,
            self.threshold_max + self.threshold_step / 2,
            self.threshold_step,
        )

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[float, float]:
        """
        Find the threshold maximizing attack-class F1.

        Tie-breaking prefers the threshold closest to 0.5, which avoids
        systematically selecting extreme thresholds when several thresholds
        produce identical F1 scores.
        """

        y_true, y_score = self._validate_inputs(
            y_true,
            y_score,
        )

        candidates = self._candidate_thresholds()

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in candidates:
            y_pred = (
                y_score >= threshold
            ).astype(np.int64)

            score = f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)

            elif np.isclose(score, best_f1):
                current_distance = abs(
                    float(threshold) - 0.5
                )

                best_distance = abs(
                    best_threshold - 0.5
                )

                if current_distance < best_distance:
                    best_threshold = float(threshold)

        return best_threshold, best_f1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> "DynamicAdaptiveThreshold":
        """
        Learn thresholds using validation data only.

        Strategy:
        1. Learn one global validation threshold.
        2. Learn per-window thresholds where both classes are present.
        3. Use the global threshold as fallback for one-class windows.
        """

        y_true, y_score = self._validate_inputs(
            y_true,
            y_score,
        )

        # --------------------------------------------------------------
        # Global fallback threshold
        # --------------------------------------------------------------

        (
            self.global_threshold_,
            self.global_f1_,
        ) = self.find_optimal_threshold(
            y_true,
            y_score,
        )

        # --------------------------------------------------------------
        # Construct groups
        # --------------------------------------------------------------

        if block_ids is None:
            groups = [
                (
                    y_true,
                    y_score,
                )
            ]
        else:
            block_ids = np.asarray(
                block_ids
            ).reshape(-1)

            if len(block_ids) != len(y_true):
                raise ValueError(
                    "block_ids must contain one value "
                    "per sample"
                )

            groups = []

            for block in np.unique(block_ids):
                rows = np.flatnonzero(
                    block_ids == block
                )

                groups.append(
                    (
                        y_true[rows],
                        y_score[rows],
                    )
                )

        thresholds: list[float] = []
        f1_scores: list[float] = []

        # --------------------------------------------------------------
        # Learn window thresholds
        # --------------------------------------------------------------

        for group_y, group_score in groups:
            n_samples = len(group_y)

            for start in range(
                0,
                n_samples,
                self.window_size,
            ):
                end = min(
                    start + self.window_size,
                    n_samples,
                )

                window_y = group_y[start:end]
                window_score = group_score[start:end]

                if len(window_y) == 0:
                    continue

                # ------------------------------------------------------
                # Only optimize locally when both classes exist.
                #
                # One-class windows produce unstable F1 thresholds.
                # Use the globally learned validation threshold instead.
                # ------------------------------------------------------

                unique_classes = np.unique(window_y)

                if len(unique_classes) < 2:
                    threshold = self.global_threshold_
                    score = f1_score(
                        window_y,
                        (
                            window_score >= threshold
                        ).astype(np.int64),
                        pos_label=1,
                        zero_division=0,
                    )
                else:
                    threshold, score = (
                        self.find_optimal_threshold(
                            window_y,
                            window_score,
                        )
                    )

                thresholds.append(
                    float(threshold)
                )

                f1_scores.append(
                    float(score)
                )

        # Safety fallback.
        if not thresholds:
            thresholds = [
                float(self.global_threshold_)
            ]

            f1_scores = [
                float(self.global_f1_)
            ]

        self.thresholds_ = np.asarray(
            thresholds,
            dtype=np.float64,
        )

        self.window_f1_ = np.asarray(
            f1_scores,
            dtype=np.float64,
        )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        y_score: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply validation-learned thresholds to new probabilities.

        Thresholds are mapped cyclically across prediction windows.
        No prediction labels are used.
        """

        self._check_fitted()

        y_score = np.asarray(
            y_score,
            dtype=np.float64,
        ).reshape(-1)

        if len(y_score) == 0:
            raise ValueError(
                "y_score must contain at least one sample"
            )

        if not np.isfinite(y_score).all():
            raise ValueError(
                "y_score contains NaN or infinite values"
            )

        if (
            np.any(y_score < 0.0)
            or np.any(y_score > 1.0)
        ):
            raise ValueError(
                "y_score must contain probabilities "
                "between 0 and 1"
            )

        predictions = np.zeros(
            len(y_score),
            dtype=np.int64,
        )

        # --------------------------------------------------------------
        # Build prediction groups
        # --------------------------------------------------------------

        if block_ids is None:
            groups = [
                np.arange(len(y_score))
            ]
        else:
            block_ids = np.asarray(
                block_ids
            ).reshape(-1)

            if len(block_ids) != len(y_score):
                raise ValueError(
                    "block_ids must contain one value "
                    "per sample"
                )

            groups = [
                np.flatnonzero(
                    block_ids == block
                )
                for block in np.unique(block_ids)
            ]

        threshold_index = 0

        # --------------------------------------------------------------
        # Apply thresholds window by window
        # --------------------------------------------------------------

        for rows in groups:
            n_samples = len(rows)

            for start in range(
                0,
                n_samples,
                self.window_size,
            ):
                end = min(
                    start + self.window_size,
                    n_samples,
                )

                window_rows = rows[start:end]

                threshold = self.thresholds_[
                    threshold_index
                    % len(self.thresholds_)
                ]

                predictions[window_rows] = (
                    y_score[window_rows] >= threshold
                ).astype(np.int64)

                threshold_index += 1

        return predictions

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def threshold_statistics(self) -> dict[str, float]:
        """
        Return summary statistics for learned thresholds.
        """

        self._check_fitted()

        return {
            "min": float(
                self.thresholds_.min()
            ),
            "mean": float(
                self.thresholds_.mean()
            ),
            "max": float(
                self.thresholds_.max()
            ),
            "std": float(
                self.thresholds_.std()
            ),
            "n_windows": int(
                len(self.thresholds_)
            ),
            "global_threshold": float(
                self.global_threshold_
            ),
            "global_f1": float(
                self.global_f1_
            ),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate binary labels and probability scores."""

        y_true = np.asarray(
            y_true,
            dtype=np.int64,
        ).reshape(-1)

        y_score = np.asarray(
            y_score,
            dtype=np.float64,
        ).reshape(-1)

        if len(y_true) != len(y_score):
            raise ValueError(
                "y_true and y_score must contain "
                "the same number of samples"
            )

        if len(y_true) == 0:
            raise ValueError(
                "At least one sample is required"
            )

        if not np.isin(
            y_true,
            [0, 1],
        ).all():
            raise ValueError(
                "y_true must contain binary labels "
                "0 and 1"
            )

        if not np.isfinite(y_score).all():
            raise ValueError(
                "y_score contains NaN or infinite values"
            )

        if (
            np.any(y_score < 0.0)
            or np.any(y_score > 1.0)
        ):
            raise ValueError(
                "y_score must contain probabilities "
                "between 0 and 1"
            )

        return y_true, y_score

    def _check_fitted(self) -> None:
        """Raise an error when called before fit."""

        if (
            self.thresholds_ is None
            or self.global_threshold_ is None
        ):
            raise RuntimeError(
                "DynamicAdaptiveThreshold is not fitted. "
                "Call fit() before predict() or "
                "threshold_statistics()."
            )


# =====================================================================
# MODULE C — SHAP DRIFT DETECTION
# =====================================================================

@dataclass
class SHAPDriftResult:
    """Results produced by SHAP drift analysis."""

    tau: np.ndarray
    drift: np.ndarray
    n_windows: int

@dataclass
class SHAPDriftDetector:
    """
    SHAP feature-importance ranking drift detector.

    Parameters
    ----------
    window_size : int
        Number of samples per SHAP analysis window.
        Project default is 1000.
    background_size : int
        Maximum number of training rows used to establish the SHAP reference.
    sample_size : int
        Maximum number of rows sampled from each evaluation window.
        SHAP values are expensive, so sampling keeps Module C practical.
    drift_threshold : float
        A window is flagged as drifted when Kendall tau is below this value.
    seed : int
        Random seed used for deterministic sampling.
    """

    window_size: int = 1000
    background_size: int = 1000
    sample_size: int = 500
    drift_threshold: float = 0.7
    seed: int = 42

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")

        if self.background_size < 1:
            raise ValueError("background_size must be at least 1")

        if self.sample_size < 1:
            raise ValueError("sample_size must be at least 1")

        if not -1.0 <= self.drift_threshold <= 1.0:
            raise ValueError(
                "drift_threshold must be between -1 and 1"
            )

        self.reference_importance_: np.ndarray | None = None
        self.reference_ranking_: np.ndarray | None = None
        self.n_features_: int | None = None
        self._explainer: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: Any,
        X_reference: np.ndarray,
    ) -> "SHAPDriftDetector":
        """
        Build the reference SHAP importance ranking.

        Under the project-standard policy, X_reference must come from
        TRAINING rows only.

        Parameters
        ----------
        model
            Fitted tree-based model compatible with shap.TreeExplainer.
        X_reference
            Training/reference feature matrix.

        Returns
        -------
        self
        """

        X_reference = self._validate_X(X_reference)

        self.n_features_ = X_reference.shape[1]

        rng = np.random.default_rng(self.seed)

        X_sample = self._sample_rows(
            X_reference,
            self.background_size,
            rng,
        )

        import shap

        self._explainer = shap.TreeExplainer(model)

        shap_values = self._compute_shap_values(X_sample)

        importance = np.mean(
            np.abs(shap_values),
            axis=0,
        )

        self.reference_importance_ = np.asarray(
            importance,
            dtype=np.float64,
        )

        self.reference_ranking_ = self._importance_to_ranking(
            self.reference_importance_
        )

        return self

    def transform(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> SHAPDriftResult:
        """
        Compute SHAP drift scores for complete windows.

        Parameters
        ----------
        X
            Evaluation feature matrix.
        block_ids
            Optional contiguous block identifier per row.

            When supplied, windows restart at every block boundary and
            therefore never combine discontinuous time periods.

        Returns
        -------
        SHAPDriftResult
            tau:
                Kendall tau for every complete window.
            drift:
                Boolean drift flag for every window.
            n_windows:
                Number of complete windows analysed.
        """

        self._check_fitted()

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "X has a different number of features than "
                "the fitted reference data"
            )

        groups = self._make_groups(
            X,
            block_ids,
        )

        rng = np.random.default_rng(self.seed)

        tau_scores: list[float] = []

        for rows in groups:
            n_rows = len(rows)

            for start in range(
                0,
                n_rows - self.window_size + 1,
                self.window_size,
            ):
                window_rows = rows[
                    start:start + self.window_size
                ]

                X_window = X[window_rows]

                X_sample = self._sample_rows(
                    X_window,
                    self.sample_size,
                    rng,
                )

                shap_values = self._compute_shap_values(
                    X_sample
                )

                importance = np.mean(
                    np.abs(shap_values),
                    axis=0,
                )

                ranking = self._importance_to_ranking(
                    importance
                )

                tau = kendalltau(
                    self.reference_ranking_,
                    ranking,
                ).statistic

                # Kendall tau can become NaN when rankings are degenerate.
                # Treat identical constant rankings as no measurable drift.
                if not np.isfinite(tau):
                    tau = 1.0

                tau_scores.append(float(tau))

        tau_array = np.asarray(
            tau_scores,
            dtype=np.float64,
        )

        drift_array = (
            tau_array < self.drift_threshold
        )

        return SHAPDriftResult(
            tau=tau_array,
            drift=drift_array,
            n_windows=len(tau_array),
        )

    def fit_transform(
        self,
        model: Any,
        X_reference: np.ndarray,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> SHAPDriftResult:
        """Fit the reference ranking and analyse evaluation windows."""

        return self.fit(
            model,
            X_reference,
        ).transform(
            X,
            block_ids,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_X(
        X: np.ndarray,
    ) -> np.ndarray:
        """Validate a numeric feature matrix."""

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional feature matrix"
            )

        if len(X) == 0:
            raise ValueError(
                "X must contain at least one sample"
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values"
            )

        return X

    @staticmethod
    def _sample_rows(
        X: np.ndarray,
        max_rows: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Deterministically sample rows without replacement.

        If X already contains fewer rows than max_rows, all rows are used.
        """

        if len(X) <= max_rows:
            return X

        indices = rng.choice(
            len(X),
            size=max_rows,
            replace=False,
        )

        return X[indices]

    @staticmethod
    def _importance_to_ranking(
        importance: np.ndarray,
    ) -> np.ndarray:
        """
        Convert feature importance values into rank positions.

        Highest importance receives rank 0.

        Stable sorting makes ties deterministic.
        """

        order = np.argsort(
            -importance,
            kind="stable",
        )

        ranking = np.empty(
            len(importance),
            dtype=np.int64,
        )

        ranking[order] = np.arange(
            len(importance)
        )

        return ranking

    @staticmethod
    def _make_groups(
        X: np.ndarray,
        block_ids: np.ndarray | None,
    ) -> list[np.ndarray]:
        """
        Construct row groups for windowing.

        Without block IDs the whole matrix is one group.
        With block IDs each block is an independent group.
        """

        if block_ids is None:
            return [
                np.arange(
                    len(X),
                    dtype=np.int64,
                )
            ]

        block_ids = np.asarray(
            block_ids
        ).reshape(-1)

        if len(block_ids) != len(X):
            raise ValueError(
                "block_ids must contain one value per X row"
            )

        return [
            np.flatnonzero(
                block_ids == block
            )
            for block in np.unique(block_ids)
        ]

    def _compute_shap_values(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SHAP values and normalize output shape.

        Different SHAP/XGBoost versions may return:
        * (samples, features)
        * (samples, features, classes)
        """

        values = self._explainer.shap_values(X)

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if values.ndim == 3:
            # Binary classification can expose a class axis.
            # Use the attack/positive class when available.
            if values.shape[2] > 1:
                values = values[:, :, 1]
            else:
                values = values[:, :, 0]

        if values.ndim != 2:
            raise RuntimeError(
                "Unexpected SHAP output shape: "
                f"{values.shape}"
            )

        return values

    def _check_fitted(self) -> None:
        """Raise an error when transform() is called before fit()."""

        if (
            self._explainer is None
            or self.reference_importance_ is None
            or self.reference_ranking_ is None
        ):
            raise RuntimeError(
                "SHAPDriftDetector is not fitted. "
                "Call fit() before transform()."
            )


# =====================================================================
# INTEGRATED XGBOOST PIPELINE
# =====================================================================

# ---------------------------------------------------------------------
# XGBoost configuration
# ---------------------------------------------------------------------

SAURABH_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}


def scale_pos_weight_of(
    y: np.ndarray,
) -> float:
    """
    Calculate negative / positive class ratio.

    Used to compensate for attack-class imbalance without random temporal
    oversampling, preserving the original ordering of rows inside blocks.
    """

    y = np.asarray(
        y,
        dtype=np.int64,
    ).reshape(-1)

    positives = int(
        (y == 1).sum()
    )
    negatives = int(
        (y == 0).sum()
    )

    if positives == 0:
        return 1.0

    return float(
        negatives / positives
    )


@dataclass
class SaurabhXGBoost:
    """
    Integrated Saurabh methodology classifier.

    Pipeline
    --------

    TRAINING
        X_train
           |
           v
        Module A: benign correlation baseline
           |
           v
        graph_frob_div appended
           |
           v
        XGBoost training
           |
           +----> Module C: SHAP reference ranking


    VALIDATION
        X_val
           |
           v
        Module A transform
           |
           v
        XGBoost probabilities
           |
           v
        Module B: validation threshold learning


    TEST
        X_test
           |
           v
        Module A transform
           |
           v
        XGBoost probabilities
           |
           +----> Module B: validation-learned thresholds
           |
           +----> Module C: SHAP ranking drift analysis

    Parameters
    ----------
    correlation_window_size:
        Number of contiguous rows used for each correlation window.

    threshold_window_size:
        Number of contiguous rows used for each dynamic threshold window.

    shap_window_size:
        Number of contiguous rows used for each SHAP drift window.

    shap_background_size:
        Maximum number of training rows sampled for the SHAP reference.

    shap_sample_size:
        Maximum number of rows sampled per SHAP evaluation window.

    shap_drift_threshold:
        Kendall tau below which a SHAP window is flagged as drifted.

    seed:
        Random seed.

    early_stopping_rounds:
        XGBoost validation patience.

    use_correlation_graph:
        Enable Module A.

    use_dynamic_threshold:
        Enable Module B.

    use_shap_drift:
        Enable Module C.
    """

    correlation_window_size: int = 500
    threshold_window_size: int = 500

    shap_window_size: int = 1000
    shap_background_size: int = 500
    shap_sample_size: int = 200
    shap_drift_threshold: float = 0.7

    seed: int = 42
    early_stopping_rounds: int = 50

    # -----------------------------------------------------------------
    # Ablation switches
    # -----------------------------------------------------------------

    use_correlation_graph: bool = True
    use_dynamic_threshold: bool = True
    use_shap_drift: bool = True

    params: dict[str, Any] = field(
        default_factory=lambda: dict(
            SAURABH_XGBOOST_PARAMS
        )
    )

    # -----------------------------------------------------------------
    # Learned components
    # -----------------------------------------------------------------

    correlation_graph: CorrelationBehaviouralGraph = field(
        init=False,
        repr=False,
    )

    dynamic_threshold: DynamicAdaptiveThreshold = field(
        init=False,
        repr=False,
    )

    shap_drift: SHAPDriftDetector = field(
        init=False,
        repr=False,
    )

    model: Any = field(
        default=None,
        init=False,
        repr=False,
    )

    scale_pos_weight_: float = field(
        default=1.0,
        init=False,
    )

    best_iteration_: int | None = field(
        default=None,
        init=False,
    )

    validation_threshold_statistics_: (
        dict[str, float] | None
    ) = field(
        default=None,
        init=False,
    )

    n_input_features_: int | None = field(
        default=None,
        init=False,
    )

    n_model_features_: int | None = field(
        default=None,
        init=False,
    )

    fitted_: bool = field(
        default=False,
        init=False,
    )

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def __post_init__(
        self,
    ) -> None:
        """Initialize methodology modules."""

        self.correlation_graph = (
            CorrelationBehaviouralGraph(
                window_size=self.correlation_window_size,
            )
        )

        self.dynamic_threshold = (
            DynamicAdaptiveThreshold(
                window_size=self.threshold_window_size,
            )
        )

        self.shap_drift = SHAPDriftDetector(
            window_size=self.shap_window_size,
            background_size=self.shap_background_size,
            sample_size=self.shap_sample_size,
            drift_threshold=self.shap_drift_threshold,
            seed=self.seed,
        )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_block_ids: np.ndarray | None = None,
        val_block_ids: np.ndarray | None = None,
    ) -> "SaurabhXGBoost":
        """
        Fit the complete Saurabh methodology.

        Leakage policy
        --------------
        * Module A baseline uses benign TRAINING rows only.
        * XGBoost trains on TRAINING rows only.
        * Early stopping observes VALIDATION data.
        * Module B learns thresholds from VALIDATION only.
        * Module C reference ranking uses TRAINING rows only.
        * TEST data is never used during fit().
        """

        X_train = np.asarray(
            X_train,
            dtype=np.float64,
        )

        y_train = np.asarray(
            y_train,
            dtype=np.int64,
        ).reshape(-1)

        X_val = np.asarray(
            X_val,
            dtype=np.float64,
        )

        y_val = np.asarray(
            y_val,
            dtype=np.int64,
        ).reshape(-1)

        self._validate_xy(
            X_train,
            y_train,
            "training",
        )

        self._validate_xy(
            X_val,
            y_val,
            "validation",
        )

        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                "Training and validation feature dimensions "
                "must match"
            )

        self._validate_block_ids(
            train_block_ids,
            len(X_train),
            "train_block_ids",
        )

        self._validate_block_ids(
            val_block_ids,
            len(X_val),
            "val_block_ids",
        )

        self.n_input_features_ = int(
            X_train.shape[1]
        )

        # -------------------------------------------------------------
        # MODULE A — Correlation Behavioural Graph
        # -------------------------------------------------------------

        if self.use_correlation_graph:

            # Static baseline fitted from benign TRAINING rows only.
            self.correlation_graph.fit(
                X_train,
                y_train,
            )

            X_train_model = (
                self.correlation_graph.transform(
                    X_train,
                    block_ids=train_block_ids,
                )
            )

            X_val_model = (
                self.correlation_graph.transform(
                    X_val,
                    block_ids=val_block_ids,
                )
            )

        else:

            # Ablation: original feature space only.
            X_train_model = X_train
            X_val_model = X_val

        self.n_model_features_ = int(
            X_train_model.shape[1]
        )

        # -------------------------------------------------------------
        # XGBOOST
        # -------------------------------------------------------------

        import xgboost as xgb

        self.scale_pos_weight_ = (
            scale_pos_weight_of(
                y_train
            )
        )

        kwargs = dict(
            self.params
        )

        kwargs["random_state"] = self.seed
        kwargs["scale_pos_weight"] = (
            self.scale_pos_weight_
        )

        # XGBoost versions differ slightly in where early stopping is
        # configured. Constructor configuration works for modern versions.
        kwargs["early_stopping_rounds"] = (
            self.early_stopping_rounds
        )

        self.model = xgb.XGBClassifier(
            **kwargs
        )

        self.model.fit(
            X_train_model,
            y_train,
            eval_set=[
                (
                    X_val_model,
                    y_val,
                )
            ],
            verbose=False,
        )

        best_iteration = getattr(
            self.model,
            "best_iteration",
            None,
        )

        self.best_iteration_ = (
            int(best_iteration)
            if best_iteration is not None
            else None
        )

        # -------------------------------------------------------------
        # MODULE B — Dynamic Adaptive Threshold
        #
        # Validation probabilities + validation labels ONLY.
        # -------------------------------------------------------------

        if self.use_dynamic_threshold:

            p_val = self.model.predict_proba(
                X_val_model
            )[:, 1]

            self.dynamic_threshold.fit(
                y_val,
                p_val,
                block_ids=val_block_ids,
            )

            self.validation_threshold_statistics_ = (
                self.dynamic_threshold.threshold_statistics()
            )

        else:

            self.validation_threshold_statistics_ = None

        # -------------------------------------------------------------
        # MODULE C — SHAP Drift Reference
        #
        # Reference feature importance is computed from TRAINING data only.
        # -------------------------------------------------------------

        if self.use_shap_drift:

            self.shap_drift.fit(
                self.model,
                X_train_model,
            )

        self.fitted_ = True

        return self

    # -----------------------------------------------------------------
    # Feature transformation
    # -----------------------------------------------------------------

    def transform_features(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform raw features into the model feature space.

        This is the single shared transformation path used by probability
        prediction and SHAP drift analysis.
        """

        self._check_fitted()

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        self._validate_x(
            X,
            "input",
        )

        self._validate_block_ids(
            block_ids,
            len(X),
            "block_ids",
        )

        if self.use_correlation_graph:

            return self.correlation_graph.transform(
                X,
                block_ids=block_ids,
            )

        return X

    # -----------------------------------------------------------------
    # Probability prediction
    # -----------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return attack-class probabilities.

        Module A is applied before XGBoost prediction.
        """

        X_model = self.transform_features(
            X,
            block_ids=block_ids,
        )

        return self.model.predict_proba(
            X_model
        )[:, 1]

    # -----------------------------------------------------------------
    # Final binary prediction
    # -----------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return binary attack predictions.

        Module B uses thresholds learned from validation data only.
        """

        probabilities = self.predict_proba(
            X,
            block_ids=block_ids,
        )

        if self.use_dynamic_threshold:

            return self.dynamic_threshold.predict(
                probabilities,
                block_ids=block_ids,
            )

        return (
            probabilities >= 0.5
        ).astype(np.int64)

    # -----------------------------------------------------------------
    # SHAP drift analysis
    # -----------------------------------------------------------------

    def detect_drift(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> SHAPDriftResult:
        """
        Run Module C: SHAP feature-importance drift detection.

        No labels are used. This is safe for test-time or deployment-time
        analysis.

        Block IDs are passed through to SHAPDriftDetector so SHAP windows
        never cross contiguous block boundaries.
        """

        self._check_fitted()

        if not self.use_shap_drift:
            raise RuntimeError(
                "SHAP drift detection is disabled "
                "for this ablation run"
            )

        X_model = self.transform_features(
            X,
            block_ids=block_ids,
        )

        return self.shap_drift.transform(
            X_model,
            block_ids=block_ids,
        )

    def detect_shap_drift(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> SHAPDriftResult:
        """Alias for detect_drift()."""

        return self.detect_drift(
            X,
            block_ids=block_ids,
        )

    # -----------------------------------------------------------------
    # Convenience information
    # -----------------------------------------------------------------

    @property
    def n_trees(
        self,
    ) -> int:
        """
        Number of trees effectively used after early stopping.
        """

        self._check_fitted()

        if self.best_iteration_ is not None:
            return (
                self.best_iteration_ + 1
            )

        return int(
            self.params["n_estimators"]
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return a compact summary of the fitted methodology.
        """

        self._check_fitted()

        return {
            "input_features": self.n_input_features_,
            "model_features": self.n_model_features_,
            "graph_feature_added": (
                self.n_model_features_
                - self.n_input_features_
            ),
            "scale_pos_weight": (
                self.scale_pos_weight_
            ),
            "best_iteration": (
                self.best_iteration_
            ),
            "n_trees": self.n_trees,

            "modules": {
                "correlation_graph": (
                    self.use_correlation_graph
                ),
                "dynamic_threshold": (
                    self.use_dynamic_threshold
                ),
                "shap_drift": (
                    self.use_shap_drift
                ),
            },

            "correlation_window_size": (
                self.correlation_window_size
            ),

            "threshold_window_size": (
                self.threshold_window_size
            ),

            "shap_window_size": (
                self.shap_window_size
            ),

            "shap_background_size": (
                self.shap_background_size
            ),

            "shap_sample_size": (
                self.shap_sample_size
            ),

            "shap_drift_threshold": (
                self.shap_drift_threshold
            ),

            "threshold_statistics": (
                self.validation_threshold_statistics_
            ),
        }

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    def _validate_xy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str,
    ) -> None:
        """Validate feature matrix and binary labels."""

        if X.ndim != 2:
            raise ValueError(
                f"{name} X must be a 2D matrix"
            )

        if len(X) == 0:
            raise ValueError(
                f"{name} data is empty"
            )

        if len(X) != len(y):
            raise ValueError(
                f"{name} X and y lengths must match"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} X contains NaN or infinite values"
            )

        if not np.isin(
            y,
            [0, 1],
        ).all():
            raise ValueError(
                f"{name} y must contain binary labels"
            )

    def _validate_x(
        self,
        X: np.ndarray,
        name: str,
    ) -> None:
        """Validate prediction feature matrix."""

        if X.ndim != 2:
            raise ValueError(
                f"{name} X must be a 2D matrix"
            )

        if len(X) == 0:
            raise ValueError(
                f"{name} data is empty"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} X contains NaN or infinite values"
            )

        if (
            self.n_input_features_ is not None
            and X.shape[1]
            != self.n_input_features_
        ):
            raise ValueError(
                f"{name} has {X.shape[1]} features, "
                f"expected {self.n_input_features_}"
            )

    @staticmethod
    def _validate_block_ids(
        block_ids: np.ndarray | None,
        n_samples: int,
        name: str,
    ) -> None:
        """Validate optional block identifiers."""

        if block_ids is None:
            return

        block_ids = np.asarray(
            block_ids
        ).reshape(-1)

        if len(block_ids) != n_samples:
            raise ValueError(
                f"{name} must contain one value "
                f"per sample"
            )

    def _check_fitted(
        self,
    ) -> None:
        """Raise an error when called before fit()."""

        if not self.fitted_:
            raise RuntimeError(
                "SaurabhXGBoost is not fitted. "
                "Call fit() first."
            )
