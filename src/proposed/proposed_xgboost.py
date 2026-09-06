"""
Complete proposed DDoS detection model.

Integrated components
---------------------
1. EWMA adaptive correlation baseline
2. CUSUM change-point detection
3. FPR-constrained threshold
4. Lightweight SHAP drift detection
5. XGBoost teacher
6. Knowledge-distilled edge student

Public interface
----------------
fit()
predict_proba()
predict()
detect_drift()
edge_predict_proba()
edge_predict()
benchmark_edge()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from src.saurabh.saurabh_xgboost import SaurabhXGBoost

from src.proposed.constrained_threshold import (
    ConstrainedThreshold,
)

from src.proposed.ewma import (
    EWMACorrelationBaseline,
)

from src.proposed.cusum import (
    CUSUMChangeDetector,
)

from src.proposed.fast_shap import (
    LightweightSHAPDriftDetector,
    FastSHAPResult,
)

from src.proposed.distillation import (
    DistilledEdgeModel,
    DistillationMetrics,
)


@dataclass
class ProposedXGBoost:
    """
    Complete proposed DDoS detection model.

    Classification pipeline:

        Input
          |
          v
        EWMA correlation baseline
          |
          v
        Correlation divergence
          |
          v
        XGBoost teacher
          |
          v
        FPR-constrained threshold
          |
          v
        Attack prediction

    Supporting components:

        CUSUM
            Controls when EWMA may adapt.

        Lightweight SHAP
            Detects feature-importance drift.

        Distillation
            Creates a lightweight edge student.
    """

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    correlation_window_size: int = 500

    # ------------------------------------------------------------------
    # EWMA
    # ------------------------------------------------------------------

    ewma_alpha: float = 0.1

    # ------------------------------------------------------------------
    # CUSUM
    # ------------------------------------------------------------------

    cusum_k: float = 0.5
    cusum_h: float = 5.0

    # Minimum fraction of samples that must be predicted benign
    # before a window is allowed to update EWMA.
    benign_update_fraction: float = 0.90

    # ------------------------------------------------------------------
    # FPR constrained threshold
    # ------------------------------------------------------------------

    fpr_alpha: float = 0.05

    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_step: float = 0.01

    # ------------------------------------------------------------------
    # Lightweight SHAP
    # ------------------------------------------------------------------

    shap_window_size: int = 100
    shap_sample_size: int = 50
    shap_drift_threshold: float = 0.7

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    enable_distillation: bool = True

    student_hidden_layer_sizes: tuple[int, ...] = (32, 16)

    student_learning_rate: float = 0.001

    student_max_iter: int = 300

    student_alpha: float = 0.0001

    # None means the student uses the teacher's selected
    # FPR-constrained threshold.
    student_threshold: float | None = None

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    seed: int = 42
    early_stopping_rounds: int = 50

    params: dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Learned objects
    # ------------------------------------------------------------------

    backbone: SaurabhXGBoost = field(
        default=None,
        init=False,
        repr=False,
    )

    ewma: EWMACorrelationBaseline = field(
        default=None,
        init=False,
        repr=False,
    )

    cusum: CUSUMChangeDetector = field(
        default=None,
        init=False,
        repr=False,
    )

    constrained_threshold: ConstrainedThreshold = field(
        default=None,
        init=False,
        repr=False,
    )

    fast_shap: LightweightSHAPDriftDetector = field(
        default=None,
        init=False,
        repr=False,
    )

    distilled_edge_model: DistilledEdgeModel | None = field(
        default=None,
        init=False,
        repr=False,
    )

    model: Any = field(
        default=None,
        init=False,
        repr=False,
    )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    fitted_: bool = field(
        default=False,
        init=False,
    )

    selected_threshold_: float | None = field(
        default=None,
        init=False,
    )

    validation_threshold_metrics_: (
        dict[str, float] | None
    ) = field(
        default=None,
        init=False,
    )

    n_input_features_: int | None = field(
        default=None,
        init=False,
    )

    n_model_features_: int | None = field(
        default=None,
        init=False,
    )

    validation_drift_events_: int = field(
        default=0,
        init=False,
    )

    validation_ewma_updates_: int = field(
        default=0,
        init=False,
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Initialize all proposed components."""

        if self.correlation_window_size < 2:
            raise ValueError(
                "correlation_window_size must be >= 2."
            )

        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(
                "ewma_alpha must be in (0, 1]."
            )

        if self.cusum_k < 0:
            raise ValueError(
                "cusum_k must be non-negative."
            )

        if self.cusum_h <= 0:
            raise ValueError(
                "cusum_h must be positive."
            )

        if not 0.0 < self.benign_update_fraction <= 1.0:
            raise ValueError(
                "benign_update_fraction must be in (0, 1]."
            )

        if self.shap_window_size < 2:
            raise ValueError(
                "shap_window_size must be >= 2."
            )

        if self.shap_sample_size <= 0:
            raise ValueError(
                "shap_sample_size must be positive."
            )

        if not self.student_hidden_layer_sizes:
            raise ValueError(
                "student_hidden_layer_sizes cannot be empty."
            )

        if any(
            size <= 0
            for size in self.student_hidden_layer_sizes
        ):
            raise ValueError(
                "student hidden-layer sizes must be positive."
            )

        # --------------------------------------------------------------
        # Saurabh XGBoost backbone
        # --------------------------------------------------------------
        #
        # We deliberately disable Saurabh's own:
        #   - correlation graph
        #   - dynamic threshold
        #   - SHAP drift
        #
        # because ProposedXGBoost provides its own versions.
        # --------------------------------------------------------------

        backbone_kwargs: dict[str, Any] = {
            "correlation_window_size": (
                self.correlation_window_size
            ),
            "threshold_window_size": 500,
            "shap_window_size": 1000,
            "shap_background_size": 500,
            "shap_sample_size": 200,
            "shap_drift_threshold": 0.7,
            "seed": self.seed,
            "early_stopping_rounds": (
                self.early_stopping_rounds
            ),
            "use_correlation_graph": False,
            "use_dynamic_threshold": False,
            "use_shap_drift": False,
        }

        if self.params:
            backbone_kwargs["params"] = dict(
                self.params
            )

        self.backbone = SaurabhXGBoost(
            **backbone_kwargs
        )

        # --------------------------------------------------------------
        # Proposed components
        # --------------------------------------------------------------

        self.ewma = EWMACorrelationBaseline(
            alpha=self.ewma_alpha
        )

        self.cusum = CUSUMChangeDetector(
            k=self.cusum_k,
            h=self.cusum_h,
        )

        self.constrained_threshold = (
            ConstrainedThreshold(
                alpha=self.fpr_alpha,
                threshold_min=self.threshold_min,
                threshold_max=self.threshold_max,
                threshold_step=self.threshold_step,
            )
        )

        self.fast_shap = (
            LightweightSHAPDriftDetector(
                window_size=self.shap_window_size,
                sample_size=min(
                    self.shap_sample_size,
                    self.shap_window_size,
                ),
                drift_threshold=self.shap_drift_threshold,
                random_state=self.seed,
                backend="tree",
            )
        )

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_block_ids: np.ndarray | None = None,
        val_block_ids: np.ndarray | None = None,
    ) -> "ProposedXGBoost":
        """
        Fit the complete proposed system.

        Training steps:

        1. Establish an initial benign EWMA baseline.
        2. Calibrate CUSUM using benign training windows.
        3. Create training correlation-divergence features.
        4. Train the XGBoost teacher.
        5. Learn an FPR-constrained threshold from validation.
        6. Build a lightweight SHAP reference.
        7. Replay validation traffic through EWMA/CUSUM.
        8. Distill the teacher into a lightweight student.
        """

        # --------------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------------

        X_train = self._validate_X(
            X_train,
            "X_train",
        )

        X_val = self._validate_X(
            X_val,
            "X_val",
        )

        y_train = self._validate_y(
            y_train,
            "y_train",
        )

        y_val = self._validate_y(
            y_val,
            "y_val",
        )

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train lengths differ."
            )

        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError(
                "X_val and y_val lengths differ."
            )

        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                "X_train and X_val must have the same "
                "number of features."
            )

        self.n_input_features_ = int(
            X_train.shape[1]
        )

        # --------------------------------------------------------------
        # 1. Initial benign EWMA baseline
        # --------------------------------------------------------------

        initial_benign_window = (
            self._first_benign_window(
                X_train,
                y_train,
                train_block_ids,
            )
        )

        self.ewma = EWMACorrelationBaseline(
            alpha=self.ewma_alpha
        )

        self.ewma.initialize(
            initial_benign_window
        )

        # --------------------------------------------------------------
        # 2. CUSUM calibration
        # --------------------------------------------------------------

        benign_scores = (
            self._collect_initial_benign_scores(
                X_train,
                y_train,
                train_block_ids,
            )
        )

        self.cusum = CUSUMChangeDetector(
            k=self.cusum_k,
            h=self.cusum_h,
        )

        self.cusum.fit(
            benign_scores
        )

        # Reset EWMA after calibration so training begins from
        # the same initial baseline.
        self.ewma = EWMACorrelationBaseline(
            alpha=self.ewma_alpha
        )

        self.ewma.initialize(
            initial_benign_window
        )

        self.cusum.reset()

        # --------------------------------------------------------------
        # 3. Training augmented features
        # --------------------------------------------------------------

        X_train_model = (
            self._build_adaptive_training_features(
                X_train,
                y_train,
                train_block_ids,
            )
        )

        # --------------------------------------------------------------
        # Reset state for validation.
        #
        # The final training EWMA becomes the starting validation
        # baseline.
        # --------------------------------------------------------------

        validation_ewma = copy.deepcopy(
            self.ewma
        )

        X_val_model = (
            self._build_stream_features(
                X_val,
                val_block_ids,
                validation_ewma,
            )
        )

        self.n_model_features_ = int(
            X_train_model.shape[1]
        )

        # --------------------------------------------------------------
        # 4. Train XGBoost teacher
        # --------------------------------------------------------------

        self.backbone.fit(
            X_train_model,
            y_train,
            X_val_model,
            y_val,
        )

        self.model = self.backbone.model

        # --------------------------------------------------------------
        # 5. FPR-constrained threshold
        # --------------------------------------------------------------

        p_val = self.model.predict_proba(
            X_val_model
        )[:, 1]

        self.constrained_threshold.fit(
            y_val,
            p_val,
        )

        self.selected_threshold_ = (
            self.constrained_threshold.threshold_
        )

        self.validation_threshold_metrics_ = {
            "threshold": float(
                self.constrained_threshold.threshold_
            ),
            "validation_recall": float(
                self.constrained_threshold.recall_
            ),
            "validation_fpr": float(
                self.constrained_threshold.fpr_
            ),
            "fpr_alpha": float(
                self.fpr_alpha
            ),
        }

        # --------------------------------------------------------------
        # 6. Lightweight SHAP reference
        # --------------------------------------------------------------

        self.fast_shap = (
            LightweightSHAPDriftDetector(
                window_size=self.shap_window_size,
                sample_size=min(
                    self.shap_sample_size,
                    self.shap_window_size,
                ),
                drift_threshold=self.shap_drift_threshold,
                random_state=self.seed,
                backend="tree",
            )
        )

        self.fast_shap.fit(
            self.model,
            X_train_model,
        )

        # --------------------------------------------------------------
        # 7. Validation adaptive-state replay
        # --------------------------------------------------------------

        self.ewma = validation_ewma

        self.cusum.reset()

        self.validation_drift_events_ = 0
        self.validation_ewma_updates_ = 0

        self._adapt_validation(
            X_val,
            val_block_ids,
        )

        # --------------------------------------------------------------
        # 8. Knowledge distillation
        # --------------------------------------------------------------

        if self.enable_distillation:

            if self.student_threshold is None:
                effective_student_threshold = (
                    self.selected_threshold_
                )
            else:
                effective_student_threshold = (
                    self.student_threshold
                )

            self.distilled_edge_model = (
                DistilledEdgeModel(
                    hidden_layer_sizes=(
                        self.student_hidden_layer_sizes
                    ),
                    learning_rate_init=(
                        self.student_learning_rate
                    ),
                    max_iter=self.student_max_iter,
                    random_state=self.seed,
                    alpha=self.student_alpha,
                    threshold=float(
                        effective_student_threshold
                    ),
                )
            )

            self.distilled_edge_model.fit(
                X_train_model,
                self.model,
                y_true=y_train,
            )

        else:
            self.distilled_edge_model = None

        self.fitted_ = True

        return self

    # ------------------------------------------------------------------
    # BUILD TRAINING FEATURES
    # ------------------------------------------------------------------

    def _build_adaptive_training_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        block_ids: np.ndarray | None,
    ) -> np.ndarray:
        """
        Build training features sequentially.

        Ground-truth labels may be used during training to identify
        confirmed-benign windows for baseline adaptation.
        """

        output = np.zeros(
            (
                X.shape[0],
                X.shape[1] + 1,
            ),
            dtype=np.float64,
        )

        for start, end in self._window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            divergence = (
                self.ewma.divergence(
                    window
                )
            )

            output[start:end] = (
                self._append_feature(
                    window,
                    np.full(
                        window.shape[0],
                        divergence,
                        dtype=np.float64,
                    ),
                )
            )

            window_labels = y[start:end]

            benign_fraction = float(
                np.mean(
                    window_labels == 0
                )
            )

            change_detected = (
                self.cusum.update(
                    divergence
                )
            )

            if (
                not change_detected
                and benign_fraction
                >= self.benign_update_fraction
            ):
                self.ewma.update(
                    window
                )

        return output

    # ------------------------------------------------------------------
    # BUILD STREAM FEATURES
    # ------------------------------------------------------------------

    def _build_stream_features(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None,
        ewma: EWMACorrelationBaseline,
    ) -> np.ndarray:
        """
        Build augmented features from the supplied EWMA state.

        This method does NOT update EWMA.
        Adaptation is deliberately handled separately.
        """

        output = np.zeros(
            (
                X.shape[0],
                X.shape[1] + 1,
            ),
            dtype=np.float64,
        )

        for start, end in self._window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            if window.shape[0] < 2:
                divergence = 0.0
            else:
                divergence = ewma.divergence(
                    window
                )

            output[start:end] = (
                self._append_feature(
                    window,
                    np.full(
                        window.shape[0],
                        divergence,
                        dtype=np.float64,
                    ),
                )
            )

        return output

    # ------------------------------------------------------------------
    # VALIDATION ADAPTATION
    # ------------------------------------------------------------------

    def _adapt_validation(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None,
    ) -> None:
        """
        Replay validation traffic and adapt EWMA where allowed.

        Validation labels are NOT used for EWMA updates.
        """

        for start, end in self._window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            divergence = (
                self.ewma.divergence(
                    window
                )
            )

            X_model = self._append_feature(
                window,
                np.full(
                    window.shape[0],
                    divergence,
                    dtype=np.float64,
                ),
            )

            probabilities = (
                self.model.predict_proba(
                    X_model
                )[:, 1]
            )

            change_detected = (
                self.cusum.update(
                    divergence
                )
            )

            if change_detected:
                self.validation_drift_events_ += 1

            benign_fraction = float(
                np.mean(
                    probabilities
                    < self.selected_threshold_
                )
            )

            if (
                not change_detected
                and benign_fraction
                >= self.benign_update_fraction
            ):
                self.ewma.update(
                    window
                )

                self.validation_ewma_updates_ += 1

    # ------------------------------------------------------------------
    # PREDICT PROBA
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict attack probabilities using the XGBoost teacher.

        A copy of EWMA/CUSUM state is used, so evaluation does not
        mutate the fitted model.
        """

        self._check_fitted()

        X = self._validate_X(
            X,
            "X",
        )

        if X.shape[1] != self.n_input_features_:
            raise ValueError(
                f"Expected {self.n_input_features_} features, "
                f"received {X.shape[1]}."
            )

        ewma = copy.deepcopy(
            self.ewma
        )

        cusum = copy.deepcopy(
            self.cusum
        )

        probabilities: list[np.ndarray] = []

        for start, end in self._window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            if window.shape[0] < 2:
                divergence = 0.0
            else:
                divergence = ewma.divergence(
                    window
                )

            X_model = self._append_feature(
                window,
                np.full(
                    window.shape[0],
                    divergence,
                    dtype=np.float64,
                ),
            )

            p = self.model.predict_proba(
                X_model
            )[:, 1]

            probabilities.append(p)

            change_detected = (
                cusum.update(
                    divergence
                )
            )

            benign_fraction = float(
                np.mean(
                    p
                    < self.selected_threshold_
                )
            )

            if (
                not change_detected
                and benign_fraction
                >= self.benign_update_fraction
            ):
                ewma.update(
                    window
                )

        if not probabilities:
            return np.empty(
                0,
                dtype=np.float64,
            )

        return np.concatenate(
            probabilities
        )

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return binary teacher predictions using the constrained threshold.
        """

        self._check_fitted()

        probabilities = self.predict_proba(
            X,
            block_ids=block_ids,
        )

        return (
            self.constrained_threshold.predict(
                probabilities
            )
        )

    # ------------------------------------------------------------------
    # SHAP DRIFT
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> list[FastSHAPResult]:
        """
        Detect SHAP feature-importance drift in complete SHAP windows.
        """

        self._check_fitted()

        X = self._validate_X(
            X,
            "X",
        )

        if X.shape[1] != self.n_input_features_:
            raise ValueError(
                f"Expected {self.n_input_features_} features, "
                f"received {X.shape[1]}."
            )

        ewma = copy.deepcopy(
            self.ewma
        )

        cusum = copy.deepcopy(
            self.cusum
        )

        results: list[FastSHAPResult] = []

        for start, end in self._shap_window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            divergence = (
                ewma.divergence(
                    window
                )
            )

            X_model = self._append_feature(
                window,
                np.full(
                    window.shape[0],
                    divergence,
                    dtype=np.float64,
                ),
            )

            result = self.fast_shap.detect(
                X_model
            )

            results.append(
                result
            )

            probabilities = (
                self.model.predict_proba(
                    X_model
                )[:, 1]
            )

            change_detected = (
                cusum.update(
                    divergence
                )
            )

            benign_fraction = float(
                np.mean(
                    probabilities
                    < self.selected_threshold_
                )
            )

            if (
                not change_detected
                and benign_fraction
                >= self.benign_update_fraction
            ):
                ewma.update(
                    window
                )

        return results

    # ------------------------------------------------------------------
    # EDGE PROBABILITY
    # ------------------------------------------------------------------

    def edge_predict_proba(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict attack probabilities using the distilled edge student.
        """

        self._check_fitted()

        if self.distilled_edge_model is None:
            raise RuntimeError(
                "Distilled edge model is not available."
            )

        X = self._validate_X(
            X,
            "X",
        )

        if X.shape[1] != self.n_input_features_:
            raise ValueError(
                f"Expected {self.n_input_features_} features, "
                f"received {X.shape[1]}."
            )

        ewma = copy.deepcopy(
            self.ewma
        )

        cusum = copy.deepcopy(
            self.cusum
        )

        probabilities: list[np.ndarray] = []

        for start, end in self._window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            if window.shape[0] < 2:
                divergence = 0.0
            else:
                divergence = ewma.divergence(
                    window
                )

            X_model = self._append_feature(
                window,
                np.full(
                    window.shape[0],
                    divergence,
                    dtype=np.float64,
                ),
            )

            p = (
                self.distilled_edge_model
                .predict_proba(
                    X_model
                )
            )

            probabilities.append(p)

            change_detected = (
                cusum.update(
                    divergence
                )
            )

            benign_fraction = float(
                np.mean(
                    p
                    < self.selected_threshold_
                )
            )

            if (
                not change_detected
                and benign_fraction
                >= self.benign_update_fraction
            ):
                ewma.update(
                    window
                )

        if not probabilities:
            return np.empty(
                0,
                dtype=np.float64,
            )

        return np.concatenate(
            probabilities
        )

    # ------------------------------------------------------------------
    # EDGE PREDICT
    # ------------------------------------------------------------------

    def edge_predict(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return binary predictions from the distilled student.
        """

        self._check_fitted()

        probabilities = (
            self.edge_predict_proba(
                X,
                block_ids=block_ids,
            )
        )

        return (
            probabilities
            >= self.distilled_edge_model.threshold
        ).astype(np.int64)

    # ------------------------------------------------------------------
    # EDGE BENCHMARK
    # ------------------------------------------------------------------

    def benchmark_edge(
        self,
        X: np.ndarray,
        repeats: int = 10,
    ) -> DistillationMetrics:
        """
        Benchmark the distilled student.

        The benchmark uses augmented features created from a copy
        of the current EWMA baseline.
        """

        self._check_fitted()

        if self.distilled_edge_model is None:
            raise RuntimeError(
                "Distillation is disabled."
            )

        if repeats <= 0:
            raise ValueError(
                "repeats must be positive."
            )

        X = self._validate_X(
            X,
            "X",
        )

        ewma = copy.deepcopy(
            self.ewma
        )

        parts: list[np.ndarray] = []

        for start, end in self._window_ranges(
            X,
            None,
        ):
            window = X[start:end]

            if window.shape[0] < 2:
                divergence = 0.0
            else:
                divergence = ewma.divergence(
                    window
                )

            parts.append(
                self._append_feature(
                    window,
                    np.full(
                        window.shape[0],
                        divergence,
                        dtype=np.float64,
                    ),
                )
            )

        if not parts:
            raise ValueError(
                "No data available for edge benchmark."
            )

        X_augmented = np.vstack(
            parts
        )

        return (
            self.distilled_edge_model
            .benchmark_inference(
                X_augmented,
                repeats=repeats,
            )
        )

    # ------------------------------------------------------------------
    # INITIAL BENIGN WINDOW
    # ------------------------------------------------------------------

    def _first_benign_window(
        self,
        X: np.ndarray,
        y: np.ndarray,
        block_ids: np.ndarray | None,
    ) -> np.ndarray:
        """
        Find the first sufficiently benign correlation window.
        """

        for start, end in self._window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            if window.shape[0] < 2:
                continue

            benign_fraction = float(
                np.mean(
                    y[start:end] == 0
                )
            )

            if (
                benign_fraction
                >= self.benign_update_fraction
            ):
                return window

        raise ValueError(
            "Could not find a sufficiently benign "
            "initial correlation window."
        )

    # ------------------------------------------------------------------
    # CUSUM CALIBRATION
    # ------------------------------------------------------------------

    def _collect_initial_benign_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        block_ids: np.ndarray | None,
    ) -> np.ndarray:
        """
        Collect benign correlation-divergence scores for CUSUM.
        """

        scores: list[float] = []

        baseline = copy.deepcopy(
            self.ewma
        )

        for start, end in self._window_ranges(
            X,
            block_ids,
        ):
            window = X[start:end]

            if window.shape[0] < 2:
                continue

            benign_fraction = float(
                np.mean(
                    y[start:end] == 0
                )
            )

            if (
                benign_fraction
                >= self.benign_update_fraction
            ):
                score = baseline.divergence(
                    window
                )

                scores.append(
                    score
                )

                baseline.update(
                    window
                )

        if not scores:
            return np.array(
                [0.0],
                dtype=np.float64,
            )

        return np.asarray(
            scores,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # SHAP WINDOWING
    # ------------------------------------------------------------------

    def _shap_window_ranges(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None,
    ) -> Iterator[tuple[int, int]]:
        """
        Generate complete SHAP windows without crossing blocks.
        """

        n_rows = X.shape[0]

        if n_rows == 0:
            return

        if block_ids is None:
            run_starts = np.array(
                [0]
            )

            run_ends = np.array(
                [n_rows]
            )

        else:
            block_ids = np.asarray(
                block_ids
            ).reshape(-1)

            if block_ids.shape[0] != n_rows:
                raise ValueError(
                    "block_ids length must match X."
                )

            changes = (
                np.flatnonzero(
                    block_ids[1:]
                    != block_ids[:-1]
                )
                + 1
            )

            run_starts = np.concatenate(
                (
                    [0],
                    changes,
                )
            )

            run_ends = np.concatenate(
                (
                    changes,
                    [n_rows],
                )
            )

        for run_start, run_end in zip(
            run_starts,
            run_ends,
        ):
            start = int(
                run_start
            )

            end_of_run = int(
                run_end
            )

            while (
                start
                + self.shap_window_size
                <= end_of_run
            ):
                yield (
                    start,
                    start
                    + self.shap_window_size,
                )

                start += (
                    self.shap_window_size
                )

    # ------------------------------------------------------------------
    # CORRELATION WINDOWING
    # ------------------------------------------------------------------

    def _window_ranges(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None,
    ) -> Iterator[tuple[int, int]]:
        """
        Generate correlation windows.

        No window crosses a supplied block boundary.
        """

        n_rows = X.shape[0]

        if n_rows == 0:
            return

        if block_ids is None:

            run_starts = np.array(
                [0]
            )

            run_ends = np.array(
                [n_rows]
            )

        else:

            block_ids = np.asarray(
                block_ids
            ).reshape(-1)

            if block_ids.shape[0] != n_rows:
                raise ValueError(
                    "block_ids length must match X."
                )

            changes = (
                np.flatnonzero(
                    block_ids[1:]
                    != block_ids[:-1]
                )
                + 1
            )

            run_starts = np.concatenate(
                (
                    [0],
                    changes,
                )
            )

            run_ends = np.concatenate(
                (
                    changes,
                    [n_rows],
                )
            )

        for run_start, run_end in zip(
            run_starts,
            run_ends,
        ):

            start = int(
                run_start
            )

            end_of_run = int(
                run_end
            )

            while start < end_of_run:

                end = min(
                    start
                    + self.correlation_window_size,
                    end_of_run,
                )

                yield (
                    start,
                    end,
                )

                start = end

    # ------------------------------------------------------------------
    # APPEND FEATURE
    # ------------------------------------------------------------------

    @staticmethod
    def _append_feature(
        X: np.ndarray,
        divergence: np.ndarray,
    ) -> np.ndarray:
        """
        Append correlation divergence as the final feature.
        """

        divergence = np.asarray(
            divergence,
            dtype=np.float64,
        ).reshape(-1, 1)

        if X.shape[0] != divergence.shape[0]:
            raise ValueError(
                "Divergence length does not match X."
            )

        return np.column_stack(
            (
                X,
                divergence,
            )
        )

    # ------------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_X(
        X: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate a feature matrix."""

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array."
            )

        if X.shape[0] == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if X.shape[1] == 0:
            raise ValueError(
                f"{name} must contain at least one feature."
            )

        if not np.all(
            np.isfinite(X)
        ):
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return X

    @staticmethod
    def _validate_y(
        y: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate binary labels."""

        y = np.asarray(
            y,
            dtype=np.int64,
        ).reshape(-1)

        if y.size == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not np.all(
            np.isin(
                y,
                [0, 1],
            )
        ):
            raise ValueError(
                f"{name} must contain only 0 and 1."
            )

        return y

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def threshold_statistics(
        self,
    ) -> dict[str, float]:
        """Return threshold statistics."""

        self._check_fitted()

        return (
            self.constrained_threshold
            .summary()
        )

    def adaptive_statistics(
        self,
    ) -> dict[str, float | int]:
        """Return EWMA/CUSUM statistics."""

        self._check_fitted()

        return {
            "ewma_alpha": float(
                self.ewma.alpha
            ),
            "ewma_updates": int(
                self.ewma.n_updates_
            ),
            "cusum_k": float(
                self.cusum.k
            ),
            "cusum_h": float(
                self.cusum.h
            ),
            "cusum_changes": int(
                self.cusum.n_changes_
            ),
            "validation_drift_events": int(
                self.validation_drift_events_
            ),
            "validation_ewma_updates": int(
                self.validation_ewma_updates_
            ),
        }

    def shap_statistics(
        self,
    ) -> dict[str, float | int]:
        """Return lightweight SHAP statistics."""

        self._check_fitted()

        return (
            self.fast_shap.summary()
        )

    def distillation_statistics(
        self,
    ) -> dict[str, Any]:
        """Return edge-student statistics."""

        self._check_fitted()

        if self.distilled_edge_model is None:
            return {
                "enabled": False
            }

        return {
            "enabled": True,
            **self.distilled_edge_model.summary(),
        }

    def n_trees(self) -> int:
        """Return the number of XGBoost trees."""

        self._check_fitted()

        return int(
            self.backbone.n_trees()
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        """Return a complete model summary."""

        self._check_fitted()

        return {
            "method": "proposed",
            "input_features": int(
                self.n_input_features_
            ),
            "model_features": int(
                self.n_model_features_
            ),
            "ewma_alpha": float(
                self.ewma_alpha
            ),
            "cusum_k": float(
                self.cusum_k
            ),
            "cusum_h": float(
                self.cusum_h
            ),
            "benign_update_fraction": float(
                self.benign_update_fraction
            ),
            "fpr_alpha": float(
                self.fpr_alpha
            ),
            "selected_threshold": float(
                self.selected_threshold_
            ),
            "validation_threshold_metrics": (
                dict(
                    self.validation_threshold_metrics_
                )
                if self.validation_threshold_metrics_
                is not None
                else None
            ),
            "adaptive_statistics": (
                self.adaptive_statistics()
            ),
            "shap_statistics": (
                self.shap_statistics()
            ),
            "distillation_statistics": (
                self.distillation_statistics()
            ),
        }

    # ------------------------------------------------------------------
    # FITTED CHECK
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Ensure the complete model has been fitted."""

        if (
            not self.fitted_
            or self.model is None
            or self.ewma is None
            or self.cusum is None
            or self.constrained_threshold is None
            or self.fast_shap is None
            or self.n_input_features_ is None
        ):
            raise RuntimeError(
                "ProposedXGBoost is not fitted. "
                "Call fit() first."
            )