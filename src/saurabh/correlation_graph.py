"""
Saurabh Methodology - Module A: Correlation Behavioural Graph.

Builds a static Pearson correlation baseline from benign training data and
measures the structural deviation of incoming feature windows using Frobenius
norm divergence.

This intentionally implements Saurabh's original static-baseline approach.
EWMA/CUSUM online adaptation belongs to Member 3's proposed methodology.
"""

from __future__ import annotations

import numpy as np


class CorrelationBehaviouralGraph:
    """
    Static correlation behavioural graph.

    Workflow:
        1. Fit a Pearson correlation baseline using benign training samples.
        2. Compute a correlation matrix for each incoming window.
        3. Measure divergence from the baseline using Frobenius norm.
        4. Assign the resulting graph_frob_div score to every row in the window.

    Parameters
    ----------
    window_size : int
        Number of samples used to construct each correlation window.
        Project default is 500.
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
            Binary labels where 0 = benign and 1 = attack.

        Returns
        -------
        self
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute graph_frob_div for each fixed non-overlapping window and
        append it as one additional feature.

        Each row inside the same window receives the same divergence score.

        If the final window contains fewer than `window_size` samples, it is
        still evaluated as long as it contains at least two samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

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

        for start in range(0, n_samples, self.window_size):
            end = min(start + self.window_size, n_samples)
            window = X[start:end]

            if len(window) >= 2:
                score = self.divergence(window)
            else:
                # A single sample cannot form a Pearson correlation matrix.
                # Reuse the previous score when possible.
                score = scores[start - 1] if start > 0 else 0.0

            scores[start:end] = score

        return np.column_stack((X, scores))

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit the benign correlation baseline and transform X.
        """

        return self.fit(X, y).transform(X)

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """Validate and convert input to a 2D floating-point array."""

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

    Constant columns have undefined Pearson correlation. They receive
    zero off-diagonal correlation, while all diagonal entries remain one.
    """

    n_features = X.shape[1]
    correlation = np.zeros((n_features, n_features), dtype=np.float64)

    # Identify features with non-zero variance in this specific window.
    variable_mask = np.std(X, axis=0) > 1e-12
    variable_indices = np.flatnonzero(variable_mask)

    if len(variable_indices) >= 2:
        variable_data = X[:, variable_mask]

        variable_correlation = np.corrcoef(
            variable_data,
            rowvar=False,
        )

        correlation[np.ix_(variable_indices, variable_indices)] = (
            variable_correlation
        )

    np.fill_diagonal(correlation, 1.0)

    return correlation
    def _check_fitted(self) -> None:
        """Raise an error if fit() has not been called."""

        if self.baseline_correlation_ is None:
            raise RuntimeError(
                "CorrelationBehaviouralGraph is not fitted. "
                "Call fit() before transform() or divergence()."
            )