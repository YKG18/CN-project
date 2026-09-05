"""
Saurabh Methodology - Integrated XGBoost Pipeline.

Combines the three proposed methodology modules:

Module A - Correlation Behavioural Graph
    Learns a benign correlation baseline from training data and adds one
    correlation-divergence feature.

Module B - Dynamic Adaptive Threshold
    Learns F1-optimal thresholds from validation probabilities only and applies
    them to test probability windows without using test labels.

Module C - SHAP Drift Detection
    Learns a reference SHAP feature-importance ranking from training data and
    detects ranking drift on later windows.

IMPORTANT SPLIT POLICY
----------------------
This model NEVER creates its own train/test split. The caller must load the
project-standard frozen split through NetworkDataAdapter.load_split().

Expected workflow:

    adapter = NetworkDataAdapter.for_feature_set("saurabh49")
    adapter.load_split(config.SPLIT_INDEX_FILE)

    bundle = adapter.for_xgboost(balance="class_weight")

    model = SaurabhXGBoost()
    model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        train_block_ids=bundle.train_block,
        val_block_ids=bundle.val_block,
    )

    probabilities = model.predict_proba(
        bundle.X_test,
        block_ids=bundle.test_block,
    )

    predictions = model.predict(
        bundle.X_test,
        block_ids=bundle.test_block,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.saurabh.correlation_graph import CorrelationBehaviouralGraph
from src.saurabh.dynamic_threshold import DynamicAdaptiveThreshold
from src.saurabh.shap_drift import SHAPDriftDetector


# ---------------------------------------------------------------------
# XGBoost configuration
# ---------------------------------------------------------------------

SAURABH_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}


def scale_pos_weight_of(y: np.ndarray) -> float:
    """
    Calculate negative / positive class ratio.

    Used to compensate for attack-class imbalance without resampling
    validation or test data.
    """

    y = np.asarray(y)

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0:
        return 1.0

    return float(negatives / positives)


@dataclass
class SaurabhXGBoost:
    """
    Integrated Saurabh methodology classifier.

    Pipeline
    --------

    TRAINING
        X_train
           |
           v
        CorrelationBehaviouralGraph.fit()
           |
           v
        transform(X_train)
           |
           v
        XGBoost.fit()

    VALIDATION
        X_val
           |
           v
        CorrelationBehaviouralGraph.transform()
           |
           v
        XGBoost probabilities
           |
           v
        DynamicAdaptiveThreshold.fit()

    TEST
        X_test
           |
           v
        CorrelationBehaviouralGraph.transform()
           |
           v
        XGBoost probabilities
           |
           +----> DynamicAdaptiveThreshold.predict()
           |
           +----> SHAPDriftDetector.transform()

    Parameters
    ----------
    correlation_window_size:
        Window size used by Module A.

    threshold_window_size:
        Window size used by Module B.

    shap_window_size:
        Window size used by Module C.

    shap_background_size:
        Maximum background rows used by SHAP.

    shap_sample_size:
        Rows sampled when calculating window SHAP importance.

    seed:
        Random seed.

    early_stopping_rounds:
        XGBoost early stopping patience.
    """

    correlation_window_size: int = 500
    threshold_window_size: int = 500

    shap_window_size: int = 1000
    shap_background_size: int = 500
    shap_sample_size: int = 200
    shap_drift_threshold: float = 0.7

    seed: int = 42
    early_stopping_rounds: int = 50

    params: dict[str, Any] = field(
        default_factory=lambda: dict(SAURABH_XGBOOST_PARAMS)
    )

    # -----------------------------------------------------------------
    # Learned components
    # -----------------------------------------------------------------

    correlation_graph: CorrelationBehaviouralGraph = field(
        init=False,
        repr=False,
    )

    dynamic_threshold: DynamicAdaptiveThreshold = field(
        init=False,
        repr=False,
    )

    shap_drift: SHAPDriftDetector = field(
        init=False,
        repr=False,
    )

    model: Any = field(
        default=None,
        init=False,
        repr=False,
    )

    scale_pos_weight_: float = field(
        default=1.0,
        init=False,
    )

    best_iteration_: int | None = field(
        default=None,
        init=False,
    )

    validation_threshold_statistics_: dict[str, float] | None = field(
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

    fitted_: bool = field(
        default=False,
        init=False,
    )

    def __post_init__(self) -> None:
        """Initialize methodology modules."""

        self.correlation_graph = CorrelationBehaviouralGraph(
            window_size=self.correlation_window_size,
        )

        self.dynamic_threshold = DynamicAdaptiveThreshold(
            window_size=self.threshold_window_size,
        )

        self.shap_drift = SHAPDriftDetector(
            window_size=self.shap_window_size,
            background_size=self.shap_background_size,
            sample_size=self.shap_sample_size,
            drift_threshold=self.shap_drift_threshold,
            seed=self.seed,
        )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_block_ids: np.ndarray | None = None,
        val_block_ids: np.ndarray | None = None,
    ) -> "SaurabhXGBoost":
        """
        Fit the complete Saurabh methodology.

        Critical leakage policy
        -----------------------
        * Correlation baseline is fitted ONLY on training data.
        * XGBoost is trained ONLY on training data.
        * Early stopping observes validation data.
        * Dynamic thresholds are fitted ONLY on validation labels/probabilities.
        * SHAP reference ranking is fitted ONLY on training data.

        Test data is never used in fit().
        """

        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)

        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.int64).reshape(-1)

        self._validate_xy(X_train, y_train, "training")
        self._validate_xy(X_val, y_val, "validation")

        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                "Training and validation feature dimensions must match"
            )

        self._validate_block_ids(
            train_block_ids,
            len(X_train),
            "train_block_ids",
        )

        self._validate_block_ids(
            val_block_ids,
            len(X_val),
            "val_block_ids",
        )

        self.n_input_features_ = int(X_train.shape[1])

        # -------------------------------------------------------------
        # MODULE A
        # Fit benign correlation baseline ONLY on training data.
        # -------------------------------------------------------------

        self.correlation_graph.fit(
            X_train,
            y_train,
        )

        X_train_graph = self.correlation_graph.transform(
            X_train,
            block_ids=train_block_ids,
        )

        X_val_graph = self.correlation_graph.transform(
            X_val,
            block_ids=val_block_ids,
        )

        self.n_model_features_ = int(
            X_train_graph.shape[1]
        )

        # -------------------------------------------------------------
        # XGBOOST
        # -------------------------------------------------------------

        import xgboost as xgb

        self.scale_pos_weight_ = scale_pos_weight_of(
            y_train
        )

        kwargs = dict(self.params)

        kwargs["random_state"] = self.seed
        kwargs["scale_pos_weight"] = self.scale_pos_weight_
        kwargs["early_stopping_rounds"] = (
            self.early_stopping_rounds
        )

        self.model = xgb.XGBClassifier(
            **kwargs
        )

        self.model.fit(
            X_train_graph,
            y_train,
            eval_set=[
                (X_val_graph, y_val)
            ],
            verbose=False,
        )

        self.best_iteration_ = getattr(
            self.model,
            "best_iteration",
            None,
        )

        # -------------------------------------------------------------
        # MODULE B
        # Learn adaptive thresholds from VALIDATION ONLY.
        # -------------------------------------------------------------

        p_val = self.model.predict_proba(
            X_val_graph
        )[:, 1]

        self.dynamic_threshold.fit(
            y_val,
            p_val,
            block_ids=val_block_ids,
        )

        self.validation_threshold_statistics_ = (
            self.dynamic_threshold.threshold_statistics()
        )

        # -------------------------------------------------------------
        # MODULE C
        # Fit SHAP reference ranking on TRAINING ONLY.
        #
        # The SHAP detector observes the final model and transformed
        # training feature space.
        # -------------------------------------------------------------

        self.shap_drift.fit(
            self.model,
            X_train_graph,
        )

        self.fitted_ = True

        return self

    # -----------------------------------------------------------------
    # Probability prediction
    # -----------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return attack probabilities.

        Module A is applied before XGBoost prediction.

        Parameters
        ----------
        X:
            Original 49-feature Saurabh input matrix.

        block_ids:
            Optional temporal block IDs. When supplied, correlation windows
            restart at block boundaries.
        """

        self._check_fitted()

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        self._validate_x(
            X,
            "prediction",
        )

        self._validate_block_ids(
            block_ids,
            len(X),
            "block_ids",
        )

        X_graph = self.correlation_graph.transform(
            X,
            block_ids=block_ids,
        )

        probabilities = self.model.predict_proba(
            X_graph
        )[:, 1]

        return probabilities

    # -----------------------------------------------------------------
    # Final binary prediction
    # -----------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return binary attack predictions.

        Probabilities come from XGBoost and decision thresholds come from
        Module B, which was fitted on validation data only.
        """

        probabilities = self.predict_proba(
            X,
            block_ids=block_ids,
        )

        return self.dynamic_threshold.predict(
            probabilities,
            block_ids=block_ids,
        )

    # -----------------------------------------------------------------
    # SHAP drift analysis
    # -----------------------------------------------------------------

    def detect_drift(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ):
        """
        Run Module C: SHAP-based feature importance drift detection.

        Parameters
        ----------
        X : np.ndarray
            Raw saurabh49 input features.
        block_ids : np.ndarray | None
            Optional contiguous-time block identifiers. When supplied,
            correlation windows are restarted at block boundaries before
            constructing the graph feature.

        Returns
        -------
        SHAPDriftResult
            Window-level Kendall tau values and drift flags.

        Notes
        -----
        No labels are used here. This method is safe for test-time or
        deployment-time drift analysis.
        """

        self._check_fitted()

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        self._validate_x(
            X,
            "drift input",
        )

        if block_ids is not None:
            block_ids = np.asarray(block_ids).reshape(-1)

            if len(block_ids) != len(X):
                raise ValueError(
                    "block_ids must contain one value per sample"
                )

        # Module A: reconstruct the correlation-behavioural feature
        # using the fitted benign baseline.
        X_graph = self.correlation_graph.transform(
            X,
            block_ids=block_ids,
        )

        # Module C: compare SHAP importance rankings against the
        # training reference. No labels are required.
        return self.shap_drift.transform(X_graph)

    def detect_shap_drift(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ):
        """
        Alias for detect_drift().

        Provided explicitly because the project methodology refers to
        Module C as SHAP drift detection.

        Parameters
        ----------
        X : np.ndarray
            Raw saurabh49 input features.
        block_ids : np.ndarray | None
            Optional contiguous-time block identifiers.

        Returns
        -------
        SHAPDriftResult
            Window-level Kendall tau values and drift flags.
        """
        return self.detect_drift(
            X,
            block_ids=block_ids,
        )    # -----------------------------------------------------------------
    # Convenience information
    # -----------------------------------------------------------------

    @property
    def n_trees(self) -> int:
        """
        Number of trees effectively used after early stopping.
        """

        self._check_fitted()

        if self.best_iteration_ is not None:
            return int(
                self.best_iteration_
            ) + 1

        return int(
            self.params["n_estimators"]
        )

    def summary(self) -> dict[str, Any]:
        """
        Return a compact summary of the fitted methodology.
        """

        self._check_fitted()

        return {
            "input_features": self.n_input_features_,
            "model_features": self.n_model_features_,
            "graph_feature_added": (
                self.n_model_features_
                - self.n_input_features_
            ),
            "scale_pos_weight": (
                self.scale_pos_weight_
            ),
            "best_iteration": (
                self.best_iteration_
            ),
            "n_trees": self.n_trees,
            "correlation_window_size": (
                self.correlation_window_size
            ),
            "threshold_window_size": (
                self.threshold_window_size
            ),
            "shap_window_size": (
                self.shap_window_size
            ),
            "threshold_statistics": (
                self.validation_threshold_statistics_
            ),
        }

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    def _validate_xy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str,
    ) -> None:
        """Validate feature matrix and binary labels."""

        if X.ndim != 2:
            raise ValueError(
                f"{name} X must be a 2D matrix"
            )

        if len(X) == 0:
            raise ValueError(
                f"{name} data is empty"
            )

        if len(X) != len(y):
            raise ValueError(
                f"{name} X and y lengths must match"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} X contains NaN or infinite values"
            )

        if not np.isin(
            y,
            [0, 1],
        ).all():
            raise ValueError(
                f"{name} y must contain binary labels"
            )

    def _validate_x(
        self,
        X: np.ndarray,
        name: str,
    ) -> None:
        """Validate prediction feature matrix."""

        if X.ndim != 2:
            raise ValueError(
                f"{name} X must be a 2D matrix"
            )

        if len(X) == 0:
            raise ValueError(
                f"{name} data is empty"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} X contains NaN or infinite values"
            )

        if (
            self.n_input_features_ is not None
            and X.shape[1]
            != self.n_input_features_
        ):
            raise ValueError(
                f"{name} has {X.shape[1]} features, "
                f"expected {self.n_input_features_}"
            )

    @staticmethod
    def _validate_block_ids(
        block_ids: np.ndarray | None,
        n_samples: int,
        name: str,
    ) -> None:
        """Validate optional block identifiers."""

        if block_ids is None:
            return

        block_ids = np.asarray(
            block_ids
        ).reshape(-1)

        if len(block_ids) != n_samples:
            raise ValueError(
                f"{name} must contain one value "
                f"per sample"
            )

    def _check_fitted(self) -> None:
        """Raise an error when called before fit()."""

        if not self.fitted_:
            raise RuntimeError(
                "SaurabhXGBoost is not fitted. "
                "Call fit() first."
            )