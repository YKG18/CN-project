"""
Knowledge distillation for a lightweight edge model.

The student learns from a combination of:

    1. Teacher soft probabilities
    2. Ground-truth training labels

The student remains a small shallow MLP and is intended for
edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor


@dataclass
class DistillationMetrics:
    """Student deployment metrics."""

    model_size_kb: float
    inference_latency_ms: float
    n_samples: int


class DistilledEdgeModel:
    """
    Lightweight student model trained from teacher probabilities
    and, optionally, ground-truth labels.

    Parameters
    ----------
    hidden_layer_sizes:
        Hidden-layer architecture.

    learning_rate_init:
        Initial learning rate.

    max_iter:
        Maximum training iterations.

    random_state:
        Random seed.

    alpha:
        L2 regularization.

    threshold:
        Probability threshold for binary prediction.

    teacher_weight:
        Weight assigned to the teacher's soft probability.

    label_weight:
        Weight assigned to the ground-truth label.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        learning_rate_init: float = 0.001,
        max_iter: int = 300,
        random_state: int = 42,
        alpha: float = 0.0001,
        threshold: float = 0.5,
        teacher_weight: float = 0.7,
        label_weight: float = 0.3,
    ) -> None:

        if not hidden_layer_sizes:
            raise ValueError(
                "hidden_layer_sizes must contain at least one layer."
            )

        if any(
            size <= 0
            for size in hidden_layer_sizes
        ):
            raise ValueError(
                "Hidden-layer sizes must be positive."
            )

        if learning_rate_init <= 0:
            raise ValueError(
                "learning_rate_init must be greater than 0."
            )

        if max_iter <= 0:
            raise ValueError(
                "max_iter must be greater than 0."
            )

        if alpha < 0:
            raise ValueError(
                "alpha must be non-negative."
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if teacher_weight < 0:
            raise ValueError(
                "teacher_weight must be non-negative."
            )

        if label_weight < 0:
            raise ValueError(
                "label_weight must be non-negative."
            )

        if (
            teacher_weight == 0
            and label_weight == 0
        ):
            raise ValueError(
                "At least one distillation weight "
                "must be positive."
            )

        self.hidden_layer_sizes = tuple(
            hidden_layer_sizes
        )

        self.learning_rate_init = float(
            learning_rate_init
        )

        self.max_iter = int(
            max_iter
        )

        self.random_state = int(
            random_state
        )

        self.alpha = float(
            alpha
        )

        self.threshold = float(
            threshold
        )

        # Normalize the weights so they sum to 1.
        total_weight = (
            teacher_weight
            + label_weight
        )

        self.teacher_weight = (
            float(teacher_weight)
            / total_weight
        )

        self.label_weight = (
            float(label_weight)
            / total_weight
        )

        self.teacher_weight_effective_: float = (
            0.0
        )

        self.label_weight_effective_: float = (
            0.0
        )

        self.student_: MLPRegressor | None = None

        self.n_features_: int | None = None

        self.fitted_: bool = False

        self.training_loss_: float | None = None

        self.teacher_target_loss_: float | None = None

        self.label_target_loss_: float | None = None

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        teacher,
        y_true: np.ndarray | None = None,
    ) -> "DistilledEdgeModel":
        """
        Train the student.

        The distillation target is:

            target =
                teacher_weight * teacher_probability
                +
                label_weight * ground_truth_label

        When y_true is not supplied, pure teacher distillation
        is used.
        """

        X = self._validate_X(
            X
        )

        teacher_probabilities = (
            self._teacher_probabilities(
                teacher,
                X,
            )
        )

        if y_true is None:

            targets = teacher_probabilities

            self.teacher_weight_effective_ = 1.0
            self.label_weight_effective_ = 0.0

        else:

            y_true = self._validate_y(
                y_true
            )

            if y_true.shape[0] != X.shape[0]:
                raise ValueError(
                    "y_true and X must contain "
                    "the same number of samples."
                )

            targets = (
                self.teacher_weight
                * teacher_probabilities
                + self.label_weight
                * y_true
            )

            self.teacher_weight_effective_ = (
                self.teacher_weight
            )

            self.label_weight_effective_ = (
                self.label_weight
            )

        self.n_features_ = int(
            X.shape[1]
        )

        # --------------------------------------------------------------
        # Create the lightweight student.
        # --------------------------------------------------------------

        self.student_ = MLPRegressor(
            hidden_layer_sizes=(
                self.hidden_layer_sizes
            ),
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=(
                self.learning_rate_init
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
        )

        # --------------------------------------------------------------
        # Train student.
        # --------------------------------------------------------------

        self.student_.fit(
            X,
            targets,
        )

        self.training_loss_ = float(
            self.student_.loss_
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT call self.predict_proba(X) here.
        #
        # predict_proba() checks self.fitted_, but fit() has not yet
        # completed. Predict directly using the fitted sklearn model.
        # --------------------------------------------------------------

        student_predictions = np.clip(
            self.student_.predict(
                X
            ),
            0.0,
            1.0,
        ).astype(float)

        # --------------------------------------------------------------
        # Teacher reconstruction error
        # --------------------------------------------------------------

        self.teacher_target_loss_ = float(
            np.mean(
                (
                    student_predictions
                    - teacher_probabilities
                ) ** 2
            )
        )

        # --------------------------------------------------------------
        # Hard-label reconstruction error
        # --------------------------------------------------------------

        if y_true is not None:

            self.label_target_loss_ = float(
                np.mean(
                    (
                        student_predictions
                        - y_true
                    ) ** 2
                )
            )

        else:

            self.label_target_loss_ = None

        # --------------------------------------------------------------
        # Now the student is officially fitted.
        # --------------------------------------------------------------

        self.fitted_ = True

        return self

    # ------------------------------------------------------------------
    # PREDICT PROBA
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return student attack probabilities.
        """

        self._check_fitted()

        X = self._validate_X(
            X
        )

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, "
                f"received {X.shape[1]}."
            )

        probabilities = (
            self.student_.predict(
                X
            )
        )

        return np.clip(
            probabilities,
            0.0,
            1.0,
        ).astype(float)

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return binary student predictions.
        """

        probabilities = (
            self.predict_proba(
                X
            )
        )

        return (
            probabilities
            >= self.threshold
        ).astype(np.int64)

    # ------------------------------------------------------------------
    # TEACHER PROBABILITIES
    # ------------------------------------------------------------------

    @staticmethod
    def _teacher_probabilities(
        teacher,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Extract attack probabilities from the teacher.
        """

        if not hasattr(
            teacher,
            "predict_proba",
        ):
            raise TypeError(
                "Teacher must expose predict_proba(X)."
            )

        probabilities = np.asarray(
            teacher.predict_proba(X)
        )

        if probabilities.ndim == 1:

            attack_probability = (
                probabilities
            )

        elif (
            probabilities.ndim == 2
            and probabilities.shape[1] >= 2
        ):

            attack_probability = (
                probabilities[:, 1]
            )

        else:

            raise ValueError(
                "Teacher predict_proba() must return "
                "(n_samples,) or (n_samples, 2)."
            )

        attack_probability = np.asarray(
            attack_probability,
            dtype=float,
        ).reshape(-1)

        if not np.all(
            np.isfinite(
                attack_probability
            )
        ):
            raise ValueError(
                "Teacher probabilities contain "
                "NaN or infinite values."
            )

        if np.any(
            (attack_probability < 0.0)
            | (attack_probability > 1.0)
        ):
            raise ValueError(
                "Teacher probabilities must be "
                "between 0 and 1."
            )

        return attack_probability

    # ------------------------------------------------------------------
    # BENCHMARK
    # ------------------------------------------------------------------

    def benchmark_inference(
        self,
        X: np.ndarray,
        repeats: int = 10,
    ) -> DistillationMetrics:
        """
        Benchmark student inference.
        """

        self._check_fitted()

        X = self._validate_X(
            X
        )

        if repeats <= 0:
            raise ValueError(
                "repeats must be positive."
            )

        # Warm-up.
        self.predict_proba(
            X
        )

        start = perf_counter()

        for _ in range(repeats):
            self.predict_proba(
                X
            )

        elapsed = (
            perf_counter() - start
        )

        latency_ms = (
            elapsed
            * 1000.0
            / repeats
        )

        return DistillationMetrics(
            model_size_kb=self.model_size_kb(),
            inference_latency_ms=float(
                latency_ms
            ),
            n_samples=int(
                X.shape[0]
            ),
        )

    # ------------------------------------------------------------------
    # MODEL SIZE
    # ------------------------------------------------------------------

    def model_size_kb(self) -> float:
        """
        Estimate serialized student model size.
        """

        self._check_fitted()

        payload = pickle.dumps(
            self.student_,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        return len(payload) / 1024.0

    # ------------------------------------------------------------------
    # PARAMETER COUNT
    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        """
        Count trainable neural-network parameters.
        """

        self._check_fitted()

        total = 0

        for weights, bias in zip(
            self.student_.coefs_,
            self.student_.intercepts_,
        ):
            total += int(
                weights.size
            )

            total += int(
                bias.size
            )

        return total

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, float | int | tuple[int, ...]]:
        """
        Return student information.
        """

        self._check_fitted()

        return {
            "hidden_layer_sizes": (
                self.hidden_layer_sizes
            ),
            "n_features": int(
                self.n_features_
            ),
            "parameter_count": int(
                self.parameter_count()
            ),
            "model_size_kb": float(
                self.model_size_kb()
            ),
            "training_loss": float(
                self.training_loss_
            ),
            "teacher_target_loss": (
                float(
                    self.teacher_target_loss_
                )
            ),
            "label_target_loss": (
                None
                if self.label_target_loss_
                is None
                else float(
                    self.label_target_loss_
                )
            ),
            "teacher_weight": float(
                self.teacher_weight_effective_
            ),
            "label_weight": float(
                self.label_weight_effective_
            ),
            "threshold": float(
                self.threshold
            ),
        }

    # ------------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_X(
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Validate feature matrix.
        """

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

    @staticmethod
    def _validate_y(
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Validate binary labels.
        """

        y = np.asarray(
            y,
            dtype=np.float64,
        ).reshape(-1)

        if y.size == 0:
            raise ValueError(
                "y cannot be empty."
            )

        if not np.all(
            np.isin(
                y,
                [0, 1],
            )
        ):
            raise ValueError(
                "y must contain only 0 and 1."
            )

        return y

    # ------------------------------------------------------------------
    # FITTED CHECK
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """
        Ensure the student has been fitted.
        """

        if (
            not self.fitted_
            or self.student_ is None
            or self.n_features_ is None
        ):
            raise RuntimeError(
                "DistilledEdgeModel is not fitted. "
                "Call fit() first."
            )