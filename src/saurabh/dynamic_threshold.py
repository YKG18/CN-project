"""
Saurabh Methodology - Module B: Dynamic Adaptive Threshold.

Selects an F1-optimal classification threshold independently for each
fixed-size probability window.

For project-standard experiments, thresholds are selected on validation data
only. Test labels must never be used for threshold selection.

All windows can optionally be constrained by block IDs so that discontinuous
time periods are never combined into one window.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


class DynamicAdaptiveThreshold:
    """
    Per-window F1-optimal adaptive threshold.

    Workflow
    --------
    1. Receive model probabilities and true labels.
    2. Construct fixed-size windows.
    3. Optionally restart windows at every block boundary.
    4. Find the threshold maximizing attack-class F1 in each window.
    5. Return one threshold for every complete window.

    Parameters
    ----------
    window_size : int
        Number of samples per threshold window.
        Project default is 500.
    threshold_min : float
        Minimum candidate threshold.
    threshold_max : float
        Maximum candidate threshold.
    threshold_step : float
        Step size used when searching candidate thresholds.
    """

    def __init__(
        self,
        window_size: int = 500,
        threshold_min: float = 0.0,
        threshold_max: float = 1.0,
        threshold_step: float = 0.01,
    ):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        if not 0.0 <= threshold_min <= 1.0:
            raise ValueError("threshold_min must be between 0 and 1")

        if not 0.0 <= threshold_max <= 1.0:
            raise ValueError("threshold_max must be between 0 and 1")

        if threshold_min >= threshold_max:
            raise ValueError(
                "threshold_min must be smaller than threshold_max"
            )

        if threshold_step <= 0:
            raise ValueError("threshold_step must be positive")

        self.window_size = window_size
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step

        self.thresholds_: np.ndarray | None = None
        self.window_f1_: np.ndarray | None = None

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[float, float]:
        """
        Find the threshold maximizing attack-class F1.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground-truth labels.
        y_score : np.ndarray
            Attack probabilities.

        Returns
        -------
        tuple[float, float]
            (best_threshold, best_f1)
        """

        y_true, y_score = self._validate_inputs(y_true, y_score)

        candidates = np.arange(
            self.threshold_min,
            self.threshold_max + self.threshold_step / 2,
            self.threshold_step,
        )

        best_threshold = float(candidates[0])
        best_f1 = -1.0

        for threshold in candidates:
            y_pred = (y_score >= threshold).astype(np.int64)

            score = f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )

            # Deterministic tie-break:
            # keep the lower threshold when F1 is identical.
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)

        return best_threshold, best_f1

    def fit(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> "DynamicAdaptiveThreshold":
        """
        Select F1-optimal thresholds for validation windows.

        This method is intended to be called on VALIDATION data only under
        the project-standard split policy.

        Windows never cross block boundaries when block_ids are supplied.
        Blocks shorter than window_size are skipped.

        Parameters
        ----------
        y_true : np.ndarray
            Validation labels.
        y_score : np.ndarray
            Validation attack probabilities.
        block_ids : np.ndarray | None
            Block identifier for each sample.

        Returns
        -------
        self
        """

        y_true, y_score = self._validate_inputs(y_true, y_score)

        if block_ids is None:
            groups = [(y_true, y_score)]
        else:
            block_ids = np.asarray(block_ids).reshape(-1)

            if len(block_ids) != len(y_true):
                raise ValueError(
                    "block_ids must contain one value per sample"
                )

            groups = []

            for block in np.unique(block_ids):
                rows = np.flatnonzero(block_ids == block)

                groups.append(
                    (y_true[rows], y_score[rows])
                )

        thresholds: list[float] = []
        f1_scores: list[float] = []

        for group_y, group_score in groups:
            n_samples = len(group_y)

            # Only complete windows are used.
            for start in range(
                0,
                n_samples - self.window_size + 1,
                self.window_size,
            ):
                end = start + self.window_size

                window_y = group_y[start:end]
                window_score = group_score[start:end]

                threshold, score = self.find_optimal_threshold(
                    window_y,
                    window_score,
                )

                thresholds.append(threshold)
                f1_scores.append(score)

        if not thresholds:
            raise ValueError(
                "No complete windows were available. "
                "Ensure at least one block contains window_size samples."
            )

        self.thresholds_ = np.asarray(
            thresholds,
            dtype=np.float64,
        )

        self.window_f1_ = np.asarray(
            f1_scores,
            dtype=np.float64,
        )

        return self

    def predict(
        self,
        y_score: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply learned adaptive thresholds to probabilities.

        The learned validation threshold sequence is mapped cyclically to
        complete test windows. This keeps test labels completely out of the
        threshold-selection process.

        Parameters
        ----------
        y_score : np.ndarray
            Attack probabilities.
        block_ids : np.ndarray | None
            Optional block identifiers.

        Returns
        -------
        np.ndarray
            Binary predictions.
        """

        self._check_fitted()

        y_score = np.asarray(
            y_score,
            dtype=np.float64,
        ).reshape(-1)

        if not np.isfinite(y_score).all():
            raise ValueError(
                "y_score contains NaN or infinite values"
            )

        if block_ids is not None:
            block_ids = np.asarray(block_ids).reshape(-1)

            if len(block_ids) != len(y_score):
                raise ValueError(
                    "block_ids must contain one value per sample"
                )

        predictions = np.zeros(
            len(y_score),
            dtype=np.int64,
        )

        threshold_index = 0

        if block_ids is None:
            groups = [np.arange(len(y_score))]
        else:
            groups = [
                np.flatnonzero(block_ids == block)
                for block in np.unique(block_ids)
            ]

        for rows in groups:
            n_samples = len(rows)

            for start in range(
                0,
                n_samples - self.window_size + 1,
                self.window_size,
            ):
                end = start + self.window_size

                window_rows = rows[start:end]

                threshold = self.thresholds_[
                    threshold_index % len(self.thresholds_)
                ]

                predictions[window_rows] = (
                    y_score[window_rows] >= threshold
                ).astype(np.int64)

                threshold_index += 1

            # Remaining incomplete samples use the final learned threshold.
            remainder_start = (
                n_samples // self.window_size
            ) * self.window_size

            if remainder_start < n_samples:
                remainder_rows = rows[remainder_start:]

                threshold = self.thresholds_[
                    threshold_index % len(self.thresholds_)
                ]

                predictions[remainder_rows] = (
                    y_score[remainder_rows] >= threshold
                ).astype(np.int64)

                threshold_index += 1

        return predictions

    def threshold_statistics(self) -> dict[str, float]:
        """
        Return summary statistics of validation-selected thresholds.
        """

        self._check_fitted()

        return {
            "min": float(self.thresholds_.min()),
            "mean": float(self.thresholds_.mean()),
            "max": float(self.thresholds_.max()),
            "std": float(self.thresholds_.std()),
            "n_windows": int(len(self.thresholds_)),
        }

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate labels and probabilities."""

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
                "y_true and y_score must contain the same number "
                "of samples"
            )

        if len(y_true) == 0:
            raise ValueError(
                "At least one sample is required"
            )

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError(
                "y_true must contain binary labels 0 and 1"
            )

        if not np.isfinite(y_score).all():
            raise ValueError(
                "y_score contains NaN or infinite values"
            )

        if np.any(y_score < 0.0) or np.any(y_score > 1.0):
            raise ValueError(
                "y_score must contain probabilities between 0 and 1"
            )

        return y_true, y_score

    def _check_fitted(self) -> None:
        """Raise an error if fit() has not been called."""

        if self.thresholds_ is None:
            raise RuntimeError(
                "DynamicAdaptiveThreshold is not fitted. "
                "Call fit() before predict() or threshold_statistics()."
            )