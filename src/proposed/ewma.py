"""
EWMA-based online correlation baseline.

Maintains an exponentially weighted moving average of correlation
matrices for confirmed-benign traffic windows.
"""

from __future__ import annotations

import numpy as np


class EWMACorrelationBaseline:
    """
    Maintain an exponentially weighted moving-average correlation matrix.

    Parameters
    ----------
    alpha:
        Weight assigned to the newest correlation matrix.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")

        self.alpha = float(alpha)

        self.baseline_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_updates_: int = 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array of shape "
                "(n_samples, n_features)."
            )

        if X.shape[0] < 2:
            raise ValueError(
                "At least two samples are required to compute correlation."
            )

        if X.shape[1] < 1:
            raise ValueError(
                "X must contain at least one feature."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    # ------------------------------------------------------------------
    # Safe correlation
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation_matrix(X: np.ndarray) -> np.ndarray:
        """
        Compute a Pearson correlation matrix without np.corrcoef.

        Constant columns are assigned:
            diagonal = 1
            off-diagonal correlations = 0

        This avoids divide-by-zero warnings for constant columns that
        can occur inside individual traffic windows.
        """

        X = EWMACorrelationBaseline._validate_X(X)

        n_samples, n_features = X.shape

        mean = np.mean(X, axis=0)
        centered = X - mean

        std = np.std(
            X,
            axis=0,
            ddof=1,
        )

        valid = std > 1e-12

        correlation = np.zeros(
            (n_features, n_features),
            dtype=np.float64,
        )

        if np.any(valid):
            Z = centered[:, valid] / std[valid]

            correlation_valid = (
                Z.T @ Z
            ) / (n_samples - 1)

            valid_indices = np.flatnonzero(valid)

            correlation[
                np.ix_(
                    valid_indices,
                    valid_indices,
                )
            ] = correlation_valid

        np.fill_diagonal(
            correlation,
            1.0,
        )

        # Numerical symmetry
        correlation = (
            correlation + correlation.T
        ) / 2.0

        return correlation

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    def initialize(
        self,
        X: np.ndarray,
    ) -> "EWMACorrelationBaseline":
        """Initialize the baseline from the first benign window."""

        X = self._validate_X(X)

        self.baseline_ = (
            self._correlation_matrix(X)
        )

        self.n_features_ = int(
            X.shape[1]
        )

        self.n_updates_ = 1

        return self

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        X: np.ndarray,
    ) -> "EWMACorrelationBaseline":
        """
        Update the baseline with a confirmed-benign window.
        """

        X = self._validate_X(X)

        if self.baseline_ is None:
            return self.initialize(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but baseline expects {self.n_features_}."
            )

        new_correlation = (
            self._correlation_matrix(X)
        )

        self.baseline_ = (
            self.alpha * new_correlation
            + (1.0 - self.alpha) * self.baseline_
        )

        self.baseline_ = (
            self.baseline_
            + self.baseline_.T
        ) / 2.0

        np.fill_diagonal(
            self.baseline_,
            1.0,
        )

        self.n_updates_ += 1

        return self

    # ------------------------------------------------------------------
    # Get baseline
    # ------------------------------------------------------------------

    def get_baseline(self) -> np.ndarray:
        """Return a copy of the current correlation baseline."""

        if self.baseline_ is None:
            raise RuntimeError(
                "EWMA baseline has not been initialized."
            )

        return self.baseline_.copy()

    # ------------------------------------------------------------------
    # Divergence
    # ------------------------------------------------------------------

    def divergence(
        self,
        X: np.ndarray,
    ) -> float:
        """
        Compute Frobenius divergence between the current window
        correlation matrix and the EWMA baseline.
        """

        if self.baseline_ is None:
            raise RuntimeError(
                "EWMA baseline has not been initialized."
            )

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but baseline expects {self.n_features_}."
            )

        current = (
            self._correlation_matrix(X)
        )

        return float(
            np.linalg.norm(
                current - self.baseline_,
                ord="fro",
            )
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, float | int]:
        """Return baseline statistics."""

        if self.baseline_ is None:
            raise RuntimeError(
                "EWMA baseline has not been initialized."
            )

        return {
            "alpha": float(self.alpha),
            "n_features": int(self.n_features_),
            "n_updates": int(self.n_updates_),
        }