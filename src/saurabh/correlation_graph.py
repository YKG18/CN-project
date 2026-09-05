"""
Saurabh Methodology - Module A: Correlation Behavioural Graph.

Builds a static Pearson correlation baseline from benign training data and
measures the structural deviation of incoming feature windows using Frobenius
norm divergence.

Project-standard policy:
    - Baseline is fitted using benign TRAINING rows only.
    - Correlation windows must remain inside contiguous split blocks.
    - No window may splice samples from different blocks.

Reference reproduction may still call transform(X) without block IDs.

This intentionally implements Saurabh's original static-baseline approach.
EWMA/CUSUM online adaptation belongs to Member 3's proposed methodology.
"""

from __future__ import annotations

import warnings

import numpy as np


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