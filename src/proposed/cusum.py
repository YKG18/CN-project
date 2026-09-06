"""
CUSUM change-point detector.

The detector is intended to monitor a scalar signal such as correlation
divergence over successive traffic windows.

The detector is trained/calibrated using benign reference scores and then
used online to decide whether a new window represents a significant change.

A detected change can later be used to prevent the EWMA correlation baseline
from being updated.
"""

from __future__ import annotations

import numpy as np


class CUSUMChangeDetector:
    """
    One-sided upward CUSUM detector.

    Parameters
    ----------
    k:
        Allowance parameter in standardized-score units.

        Small values make the detector more sensitive to small changes.
        Larger values require larger sustained changes.

    h:
        Decision threshold for the cumulative positive sum.

        Larger values make the detector less sensitive and reduce false
        alarms, at the cost of slower detection.
    """

    def __init__(
        self,
        k: float = 0.5,
        h: float = 5.0,
    ) -> None:

        if k < 0:
            raise ValueError("k must be non-negative.")

        if h <= 0:
            raise ValueError("h must be greater than 0.")

        self.k = float(k)
        self.h = float(h)

        self.reference_mean_: float | None = None
        self.reference_std_: float | None = None

        self.cusum_: float = 0.0
        self.n_samples_: int = 0
        self.n_changes_: int = 0

        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        benign_scores: np.ndarray,
    ) -> "CUSUMChangeDetector":
        """
        Calibrate the detector using benign reference scores.

        Parameters
        ----------
        benign_scores:
            Scalar divergence scores obtained from known-benign traffic.

        Returns
        -------
        CUSUMChangeDetector
            Fitted detector.
        """

        scores = np.asarray(
            benign_scores,
            dtype=float,
        ).reshape(-1)

        if scores.size == 0:
            raise ValueError(
                "benign_scores cannot be empty."
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "benign_scores contains NaN or infinite values."
            )

        self.reference_mean_ = float(
            np.mean(scores)
        )

        std = float(
            np.std(scores, ddof=1)
        ) if scores.size > 1 else 0.0

        # Prevent division by zero when benign divergence is constant.
        if std < 1e-12:
            std = 1.0

        self.reference_std_ = std

        self.reset()

        self.fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Sequential detection
    # ------------------------------------------------------------------

    def update(
        self,
        score: float,
    ) -> bool:
        """
        Process one new divergence score.

        Returns
        -------
        bool
            True if a change is detected.
            False otherwise.
        """

        self._check_fitted()

        score = float(score)

        if not np.isfinite(score):
            raise ValueError(
                "score must be finite."
            )

        # Standardize relative to benign reference traffic.
        z = (
            score - self.reference_mean_
        ) / self.reference_std_

        # One-sided upward CUSUM.
        self.cusum_ = max(
            0.0,
            self.cusum_ + z - self.k,
        )

        self.n_samples_ += 1

        if self.cusum_ >= self.h:
            self.n_changes_ += 1

            # Reset after an alarm so that a persistent change can
            # be detected as a new event later.
            self.cusum_ = 0.0

            return True

        return False

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Process a sequence of divergence scores.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates a detected change.
        """

        scores = np.asarray(
            scores,
            dtype=float,
        ).reshape(-1)

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "scores contains NaN or infinite values."
            )

        detections = np.zeros(
            scores.shape[0],
            dtype=bool,
        )

        for i, score in enumerate(scores):
            detections[i] = self.update(
                float(score)
            )

        return detections

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the cumulative statistic and counters."""

        self.cusum_ = 0.0
        self.n_samples_ = 0
        self.n_changes_ = 0

    # ------------------------------------------------------------------
    # Status / diagnostics
    # ------------------------------------------------------------------

    def current_statistic(self) -> float:
        """Return the current cumulative CUSUM statistic."""

        return float(self.cusum_)

    def summary(self) -> dict[str, float | int | bool]:
        """Return detector configuration and current state."""

        self._check_fitted()

        return {
            "k": self.k,
            "h": self.h,
            "reference_mean": float(
                self.reference_mean_
            ),
            "reference_std": float(
                self.reference_std_
            ),
            "cusum": float(
                self.cusum_
            ),
            "n_samples": self.n_samples_,
            "n_changes": self.n_changes_,
            "fitted": self.fitted_,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise an error if fit() has not been called."""

        if (
            not self.fitted_
            or self.reference_mean_ is None
            or self.reference_std_ is None
        ):
            raise RuntimeError(
                "CUSUMChangeDetector is not fitted. "
                "Call fit() first."
            )