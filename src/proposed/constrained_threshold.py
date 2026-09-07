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
    ) -> tuple[float, float, float]:
        """
        Return F1, recall, and FPR.
        """
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        recall_denom = tp + fn
        prec_denom = tp + fp
        fpr_denom = fp + tn

        recall = tp / recall_denom if recall_denom > 0 else 0.0
        prec = tp / prec_denom if prec_denom > 0 else 0.0
        f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
        fpr = fp / fpr_denom if fpr_denom > 0 else 0.0

        return float(f1), float(recall), float(fpr)

    def find_optimal_threshold(
        self,
        y_val: np.ndarray,
        p_val: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Find the threshold that maximizes F1 score while keeping FPR <= alpha.

        Returns
        -------
        threshold, recall, fpr
        """
        y_val, p_val = self._validate_inputs(y_val, p_val)

        best_threshold: float | None = None
        best_f1 = -1.0
        best_recall = -1.0
        best_fpr = float("inf")

        for threshold in self._candidate_thresholds():
            y_pred = (p_val >= threshold).astype(int)

            f1, recall, fpr = self._metrics(y_val, y_pred)

            # Constraint
            if fpr > self.alpha:
                continue

            # Primary objective: maximum F1 score under FPR constraint
            if (
                f1 > best_f1
                or (
                    np.isclose(f1, best_f1)
                    and recall > best_recall
                )
                or (
                    np.isclose(f1, best_f1)
                    and np.isclose(recall, best_recall)
                    and fpr < best_fpr
                )
            ):
                best_threshold = float(threshold)
                best_f1 = f1
                best_recall = recall
                best_fpr = fpr

        if best_threshold is None:
            # Fallback to threshold that minimizes FPR
            best_threshold = 0.5
            best_recall = 0.0
            best_fpr = 0.0

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

    def fit_benign_only(
        self,
        benign_scores: np.ndarray,
    ) -> float:
        """
        Select a threshold from assumed-benign scores alone, without labels.

        On a pool of scores believed to be benign the false-positive rate at
        tau is just the fraction of the pool at or above tau, so the
        Neyman-Pearson rule -- smallest tau with FPR <= alpha -- needs no
        labels at all. Same grid and same constraint as `find_optimal_threshold`;
        only the source of the FPR estimate differs.

        This is what the online threshold channel uses, where labels do not
        exist by construction. It does not touch `threshold_`, so a fitted
        ConstrainedThreshold keeps whatever `fit()` learned.

        The search is bounded by `threshold_max`. If even that cut leaves the
        empirical FPR above alpha, `threshold_max` is returned as the tightest
        decision the configured grid allows -- the budget is then infeasible,
        not silently satisfied. Callers that must guarantee the bound should
        widen `threshold_max` or check the returned threshold against it.

        Returns
        -------
        float
            The smallest grid threshold with empirical FPR <= alpha, or
            `threshold_max` when no grid threshold achieves it.
        """

        scores = np.asarray(
            benign_scores,
            dtype=float,
        ).reshape(-1)

        if scores.size == 0:
            raise ValueError("benign_scores cannot be empty.")

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "benign_scores contains NaN or infinite values."
            )

        # Candidates ascend, so the first satisfying threshold is the
        # smallest one -- i.e. the most sensitive cut that still respects
        # the budget.
        for threshold in self._candidate_thresholds():
            fpr = float(
                np.mean(scores >= threshold)
            )

            if fpr <= self.alpha:
                return float(threshold)

        return float(self.threshold_max)

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