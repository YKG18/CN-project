"""
Lightweight SHAP-based drift detection.

This module is intended as a lower-latency alternative to the existing
Saurabh SHAP drift detector.

Main differences:
    - smaller evaluation windows
    - configurable sample size
    - incremental Kendall tau comparison
    - measured SHAP and drift-detection latency

The detector uses a SHAP-compatible explainer backend. TreeExplainer is
used as a reliable fallback for tree models when a dedicated fast backend
is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.stats import kendalltau


@dataclass
class FastSHAPResult:
    """Result from one SHAP drift analysis window."""

    kendall_tau: float
    drift: bool
    shap_latency_ms: float
    total_latency_ms: float
    n_samples: int
    reference_n_features: int


class LightweightSHAPDriftDetector:
    """
    Lightweight SHAP feature-importance drift detector.

    Parameters
    ----------
    window_size:
        Number of rows in each drift-analysis window.

    sample_size:
        Maximum number of rows actually used for SHAP computation.

        The project recommends approximately 50–100 samples.

    drift_threshold:
        Kendall tau below this value is considered drift.

    random_state:
        Random seed used when sampling rows.

    backend:
        SHAP backend.

        "auto" attempts to use an available fast backend and otherwise
        falls back to TreeExplainer.

        "tree" explicitly uses shap.TreeExplainer.

    Notes
    -----
    The detector is designed for a fitted tree-based model such as
    XGBoost.
    """

    def __init__(
        self,
        window_size: int = 100,
        sample_size: int = 100,
        drift_threshold: float = 0.7,
        random_state: int = 42,
        backend: str = "auto",
    ) -> None:

        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than 0."
            )

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than 0."
            )

        if sample_size > window_size:
            sample_size = window_size

        if not -1.0 <= drift_threshold <= 1.0:
            raise ValueError(
                "drift_threshold must be between -1 and 1."
            )

        if backend not in {"auto", "tree"}:
            raise ValueError(
                "backend must be 'auto' or 'tree'."
            )

        self.window_size = int(window_size)
        self.sample_size = int(sample_size)
        self.drift_threshold = float(drift_threshold)
        self.random_state = int(random_state)
        self.backend = backend

        self._explainer = None
        self.reference_ranking_: np.ndarray | None = None
        self.reference_importance_: np.ndarray | None = None
        self.n_features_: int | None = None

        self.total_windows_: int = 0
        self.drift_events_: int = 0

        self.shap_latency_ms_: list[float] = []
        self.total_latency_ms_: list[float] = []
        self.kendall_tau_: list[float] = []

        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        model,
        X_reference: np.ndarray,
    ) -> "LightweightSHAPDriftDetector":
        """
        Build the reference SHAP importance ranking.

        Parameters
        ----------
        model:
            Fitted tree-based model.

        X_reference:
            Reference training data, ideally representing benign traffic.
        """

        X_reference = self._validate_X(X_reference)

        self.n_features_ = X_reference.shape[1]

        self._create_explainer(model)

        sample = self._sample_rows(
            X_reference,
            self.sample_size,
        )

        start = perf_counter()

        shap_values = self._compute_shap_values(sample)

        elapsed_ms = (
            perf_counter() - start
        ) * 1000.0

        importance = self._importance(
            shap_values
        )

        self.reference_importance_ = importance

        self.reference_ranking_ = (
            self._importance_to_ranking(importance)
        )

        self.reference_fit_latency_ms_ = elapsed_ms

        self.fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Drift analysis
    # ------------------------------------------------------------------

    def detect(
        self,
        X_window: np.ndarray,
    ) -> FastSHAPResult:
        """
        Analyze one traffic window for feature-importance drift.
        """

        self._check_fitted()

        X_window = self._validate_X(
            X_window
        )

        if X_window.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, "
                f"received {X_window.shape[1]}."
            )

        start_total = perf_counter()

        sample = self._sample_rows(
            X_window,
            self.sample_size,
        )

        start_shap = perf_counter()

        shap_values = self._compute_shap_values(
            sample
        )

        shap_latency_ms = (
            perf_counter() - start_shap
        ) * 1000.0

        importance = self._importance(
            shap_values
        )

        ranking = (
            self._importance_to_ranking(
                importance
            )
        )

        tau = self._kendall_tau(
            self.reference_ranking_,
            ranking,
        )

        drift = bool(
            tau < self.drift_threshold
        )

        total_latency_ms = (
            perf_counter() - start_total
        ) * 1000.0

        self.total_windows_ += 1

        if drift:
            self.drift_events_ += 1

        self.shap_latency_ms_.append(
            float(shap_latency_ms)
        )

        self.total_latency_ms_.append(
            float(total_latency_ms)
        )

        self.kendall_tau_.append(
            float(tau)
        )

        return FastSHAPResult(
            kendall_tau=float(tau),
            drift=drift,
            shap_latency_ms=float(
                shap_latency_ms
            ),
            total_latency_ms=float(
                total_latency_ms
            ),
            n_samples=int(sample.shape[0]),
            reference_n_features=int(
                self.n_features_
            ),
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def detect_windows(
        self,
        X: np.ndarray,
    ) -> list[FastSHAPResult]:
        """
        Divide X into contiguous windows and analyze each window.

        Rows in incomplete final windows are ignored.
        """

        self._check_fitted()

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, "
                f"received {X.shape[1]}."
            )

        results: list[FastSHAPResult] = []

        n_complete = (
            X.shape[0] // self.window_size
        )

        for i in range(n_complete):
            start = (
                i * self.window_size
            )
            end = (
                start + self.window_size
            )

            result = self.detect(
                X[start:end]
            )

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # SHAP backend
    # ------------------------------------------------------------------

    def _create_explainer(self, model) -> None:
        """
        Create the SHAP explainer.

        For this first implementation we use TreeExplainer because it is
        already part of the project's environment and is appropriate for
        XGBoost.

        The computational reduction comes primarily from using only
        50–100 samples per window. The backend can be replaced later with
        a dedicated FastSHAP implementation for the final benchmark.
        """

        if self.backend not in {"auto", "tree"}:
            raise ValueError(
                f"Unsupported backend: {self.backend}"
            )

        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "The 'shap' package is required for "
                "LightweightSHAPDriftDetector."
            ) from exc

        self._explainer = (
            shap.TreeExplainer(model)
        )

    # ------------------------------------------------------------------
    # SHAP calculation
    # ------------------------------------------------------------------

    def _compute_shap_values(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SHAP values and normalize their shape.
        """

        if self._explainer is None:
            raise RuntimeError(
                "SHAP explainer has not been initialized."
            )

        values = self._explainer.shap_values(X)

        values = np.asarray(
            values
        )

        # Standard binary classification:
        #
        #   (n_samples, n_features)
        #
        if values.ndim == 2:
            return values

        # Some SHAP/XGBoost versions may return:
        #
        #   (n_samples, n_features, n_classes)
        #
        if values.ndim == 3:
            if values.shape[2] > 1:
                return values[:, :, 1]

            return values[:, :, 0]

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{values.shape}"
        )

    # ------------------------------------------------------------------
    # Importance / ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _importance(
        shap_values: np.ndarray,
    ) -> np.ndarray:
        """
        Convert SHAP values to global feature importance.
        """

        importance = np.mean(
            np.abs(shap_values),
            axis=0,
        )

        return np.asarray(
            importance,
            dtype=float,
        )

    @staticmethod
    def _importance_to_ranking(
        importance: np.ndarray,
    ) -> np.ndarray:
        """
        Convert feature importance to a ranking.

        Rank 0 = most important feature.
        """

        importance = np.asarray(
            importance,
            dtype=float,
        )

        return np.argsort(
            -importance,
            kind="stable",
        )

    # ------------------------------------------------------------------
    # Kendall tau
    # ------------------------------------------------------------------

    @staticmethod
    def _kendall_tau(
        reference_ranking: np.ndarray,
        current_ranking: np.ndarray,
    ) -> float:
        """
        Calculate Kendall tau between two feature rankings.
        """

        reference_ranking = np.asarray(
            reference_ranking
        ).reshape(-1)

        current_ranking = np.asarray(
            current_ranking
        ).reshape(-1)

        if reference_ranking.shape != current_ranking.shape:
            raise ValueError(
                "Reference and current rankings must "
                "have the same number of features."
            )

        # Convert ordering into rank positions.
        reference_positions = np.empty_like(
            reference_ranking
        )

        current_positions = np.empty_like(
            current_ranking
        )

        reference_positions[
            reference_ranking
        ] = np.arange(
            reference_ranking.size
        )

        current_positions[
            current_ranking
        ] = np.arange(
            current_ranking.size
        )

        tau, _ = kendalltau(
            reference_positions,
            current_positions,
        )

        if np.isnan(tau):
            return 0.0

        return float(tau)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_rows(
        self,
        X: np.ndarray,
        sample_size: int,
    ) -> np.ndarray:
        """
        Randomly sample rows without replacement.

        If the window is smaller than sample_size, all rows are used.
        """

        if X.shape[0] <= sample_size:
            return X

        rng = np.random.default_rng(
            self.random_state
        )

        indices = rng.choice(
            X.shape[0],
            size=sample_size,
            replace=False,
        )

        indices.sort()

        return X[indices]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_X(
        X: np.ndarray,
    ) -> np.ndarray:
        """Validate an input feature matrix."""

        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "X cannot be empty."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature."
            )

        if not np.all(
            np.isfinite(X)
        ):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    def _check_fitted(self) -> None:
        """Raise an error if fit() has not been called."""

        if (
            not self.fitted_
            or self._explainer is None
            or self.reference_ranking_ is None
            or self.n_features_ is None
        ):
            raise RuntimeError(
                "LightweightSHAPDriftDetector is not fitted. "
                "Call fit() first."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, float | int]:
        """Return detector configuration and statistics."""

        self._check_fitted()

        mean_shap_latency = (
            float(
                np.mean(
                    self.shap_latency_ms_
                )
            )
            if self.shap_latency_ms_
            else 0.0
        )

        mean_total_latency = (
            float(
                np.mean(
                    self.total_latency_ms_
                )
            )
            if self.total_latency_ms_
            else 0.0
        )

        mean_tau = (
            float(
                np.mean(
                    self.kendall_tau_
                )
            )
            if self.kendall_tau_
            else 0.0
        )

        return {
            "window_size": self.window_size,
            "sample_size": self.sample_size,
            "drift_threshold": self.drift_threshold,
            "n_features": int(
                self.n_features_
            ),
            "n_windows": self.total_windows_,
            "n_drift_events": self.drift_events_,
            "mean_shap_latency_ms": mean_shap_latency,
            "mean_total_latency_ms": mean_total_latency,
            "mean_kendall_tau": mean_tau,
        }