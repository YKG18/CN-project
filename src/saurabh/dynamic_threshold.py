"""
Saurabh Methodology - Module B: Dynamic Adaptive Threshold.

Learns classification thresholds from validation probabilities only.

Important policy:
- Threshold selection uses validation labels/probabilities.
- Test labels are never used.
- Thresholds are constrained to avoid degenerate all-positive predictions
  when validation windows contain no attack samples.
- A global validation threshold is used as a stable fallback for windows
  that do not contain both classes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


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