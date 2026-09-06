"""
Precision-constrained threshold selection.

The threshold is selected using validation data only.

Objective:
    Maximize recall for the attack class (label 1)
    subject to:
    FPR <= alpha
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class ConstrainedThreshold:
    """
    Select a classification threshold that maximizes recall while
    satisfying a maximum false-positive-rate constraint.

    Parameters
    ----------
    alpha:
        Maximum allowed false-positive rate.
    threshold_min:
        Minimum threshold to consider.
    threshold_max:
        Maximum threshold to consider.
    threshold_step:
        Step size for threshold search.
    """

    alpha: float = 0.05
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_step: float = 0.01

    threshold_: float | None = None
    recall_: float | None = None
    fpr_: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")

        if not 0.0 <= self.threshold_min <= 1.0:
            raise ValueError("threshold_min must be between 0 and 1.")

        if not 0.0 <= self.threshold_max <= 1.0:
            raise ValueError("threshold_max must be between 0 and 1.")

        if self.threshold_min >= self.threshold_max:
            raise ValueError(
                "threshold_min must be smaller than threshold_max."
            )

        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be positive.")

    def _candidate_thresholds(self) -> np.ndarray:
        """Return the thresholds tested during optimization."""
        thresholds = np.arange(
            self.threshold_min,
            self.threshold_max + self.threshold_step / 2,
            self.threshold_step,
        )

        return np.round(thresholds, 10)

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate labels and attack probabilities."""

        y_true = np.asarray(y_true).reshape(-1)
        y_score = np.asarray(y_score, dtype=float).reshape(-1)

        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError(
                "y_true and y_score must contain the same number of samples."
            )

        if y_true.size == 0:
            raise ValueError("Validation data cannot be empty.")

        unique_labels = np.unique(y_true)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("y_true must contain only labels 0 and 1.")

        if not np.all(np.isfinite(y_score)):
            raise ValueError("y_score contains NaN or infinite values.")

        if np.any((y_score < 0.0) | (y_score > 1.0)):
            raise ValueError("y_score must contain probabilities in [0, 1].")

        if not np.any(y_true == 1):
            raise ValueError(
                "Validation data must contain at least one attack sample."
            )

        if not np.any(y_true == 0):
            raise ValueError(
                "Validation data must contain at least one benign sample."
            )

        return y_true.astype(int), y_score

    @staticmethod
    def _metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[float, float]:
        """
        Return recall and FPR.

        Recall = TP / (TP + FN)
        FPR    = FP / (FP + TN)
        """

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        recall_denominator = tp + fn
        fpr_denominator = fp + tn

        recall = (
            tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        fpr = (
            fp / fpr_denominator
            if fpr_denominator > 0
            else 0.0
        )

        return float(recall), float(fpr)

    def find_optimal_threshold(
        self,
        y_val: np.ndarray,
        p_val: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Find the threshold that maximizes recall while keeping FPR <= alpha.

        Returns
        -------
        threshold, recall, fpr
        """

        y_val, p_val = self._validate_inputs(y_val, p_val)

        best_threshold: float | None = None
        best_recall = -1.0
        best_fpr = float("inf")

        for threshold in self._candidate_thresholds():
            y_pred = (p_val >= threshold).astype(int)

            recall, fpr = self._metrics(y_val, y_pred)

            # Constraint
            if fpr > self.alpha:
                continue

            # Primary objective: maximum recall
            # Secondary objective: lower FPR
            # Tertiary objective: threshold closer to 0.5
            if (
                recall > best_recall
                or (
                    np.isclose(recall, best_recall)
                    and fpr < best_fpr
                )
                or (
                    np.isclose(recall, best_recall)
                    and np.isclose(fpr, best_fpr)
                    and (
                        best_threshold is None
                        or abs(threshold - 0.5)
                        < abs(best_threshold - 0.5)
                    )
                )
            ):
                best_threshold = float(threshold)
                best_recall = recall
                best_fpr = fpr

        if best_threshold is None:
            raise ValueError(
                f"No threshold in the configured range satisfies "
                f"FPR <= {self.alpha:.4f}."
            )

        return best_threshold, best_recall, best_fpr

    def fit(
        self,
        y_val: np.ndarray,
        p_val: np.ndarray,
    ) -> "ConstrainedThreshold":
        """
        Learn the threshold from validation labels and probabilities.

        Test labels must never be used here.
        """

        (
            self.threshold_,
            self.recall_,
            self.fpr_,
        ) = self.find_optimal_threshold(y_val, p_val)

        return self

    def predict(self, p_score: np.ndarray) -> np.ndarray:
        """Convert attack probabilities into binary predictions."""

        if self.threshold_ is None:
            raise RuntimeError(
                "ConstrainedThreshold is not fitted. "
                "Call fit() first."
            )

        p_score = np.asarray(p_score, dtype=float).reshape(-1)

        if not np.all(np.isfinite(p_score)):
            raise ValueError(
                "p_score contains NaN or infinite values."
            )

        if np.any((p_score < 0.0) | (p_score > 1.0)):
            raise ValueError("p_score must contain probabilities in [0, 1].")

        return (p_score >= self.threshold_).astype(int)

    def summary(self) -> dict[str, float]:
        """Return the learned threshold and validation metrics."""

        if self.threshold_ is None:
            raise RuntimeError(
                "ConstrainedThreshold is not fitted."
            )

        return {
            "threshold": float(self.threshold_),
            "validation_recall": float(self.recall_),
            "validation_fpr": float(self.fpr_),
            "alpha": float(self.alpha),
        }