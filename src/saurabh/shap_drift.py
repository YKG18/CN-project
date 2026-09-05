"""
Saurabh Methodology - Module C: SHAP Drift Detection.

Detects behavioural drift by comparing feature-importance rankings across
time windows using Kendall's tau correlation.

Project-standard policy
-----------------------
* The SHAP reference ranking is fitted using TRAINING rows only.
* Test labels are never used.
* Windows are constructed entirely inside contiguous blocks.
* Windows shorter than window_size are skipped.

Method
------
1. Compute a reference SHAP feature-importance ranking.
2. Split incoming samples into fixed-size windows.
3. Compute SHAP importance ranking for every window.
4. Compare each window ranking against the reference ranking using
   Kendall's tau.
5. Low Kendall tau indicates feature-importance drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import kendalltau


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