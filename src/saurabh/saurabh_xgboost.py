"""
Saurabh Methodology - Integrated XGBoost Pipeline for Data4Cyber.

Combines three methodology modules with a leakage-safe evaluation policy.

Module A - Correlation Behavioural Graph
    Learns a static benign Pearson-correlation baseline from TRAINING data
    only and appends a graph_frob_div feature to each row.

Module B - Dynamic Adaptive Threshold
    Learns classification thresholds from VALIDATION probabilities and labels
    only. Test labels are never used for threshold selection.

Module C - SHAP Drift Detection
    Learns a reference SHAP feature-importance ranking from TRAINING data only
    and detects ranking drift in later windows without using their labels.

IMPORTANT SPLIT POLICY
----------------------
This model NEVER creates train/validation/test splits.

The caller must use the frozen Data4Cyber splits produced by:

    common/data/data4cyber/_prep.py

For the primary experiment:

    adapter = Data4CyberAdapter(split_mode="block")
    bundle = adapter.for_xgboost(balance="class_weight")

Block identifiers stored in bundle.meta are passed to all methodology modules
that construct temporal windows. This guarantees that windows never combine
rows from different contiguous blocks.

Typical workflow
----------------

    from common.data.data4cyber_adapter import Data4CyberAdapter
    from saurabh.saurabh_xgboost import SaurabhXGBoost

    adapter = Data4CyberAdapter(split_mode="block")
    bundle = adapter.for_xgboost(balance="class_weight")

    model = SaurabhXGBoost()

    model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        train_block_ids=bundle.meta["train_block"],
        val_block_ids=bundle.meta["val_block"],
    )

    probabilities = model.predict_proba(
        bundle.X_test,
        block_ids=bundle.meta["test_block"],
    )

    predictions = model.predict(
        bundle.X_test,
        block_ids=bundle.meta["test_block"],
    )

    drift = model.detect_drift(
        bundle.X_test,
        block_ids=bundle.meta["test_block"],
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


def scale_pos_weight_of(
    y: np.ndarray,
) -> float:
    """
    Calculate negative / positive class ratio.

    Used to compensate for attack-class imbalance without random temporal
    oversampling, preserving the original ordering of rows inside blocks.
    """

    y = np.asarray(
        y,
        dtype=np.int64,
    ).reshape(-1)

    positives = int(
        (y == 1).sum()
    )
    negatives = int(
        (y == 0).sum()
    )

    if positives == 0:
        return 1.0

    return float(
        negatives / positives
    )


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
        Module A: benign correlation baseline
           |
           v
        graph_frob_div appended
           |
           v
        XGBoost training
           |
           +----> Module C: SHAP reference ranking


    VALIDATION
        X_val
           |
           v
        Module A transform
           |
           v
        XGBoost probabilities
           |
           v
        Module B: validation threshold learning


    TEST
        X_test
           |
           v
        Module A transform
           |
           v
        XGBoost probabilities
           |
           +----> Module B: validation-learned thresholds
           |
           +----> Module C: SHAP ranking drift analysis

    Parameters
    ----------
    correlation_window_size:
        Number of contiguous rows used for each correlation window.

    threshold_window_size:
        Number of contiguous rows used for each dynamic threshold window.

    shap_window_size:
        Number of contiguous rows used for each SHAP drift window.

    shap_background_size:
        Maximum number of training rows sampled for the SHAP reference.

    shap_sample_size:
        Maximum number of rows sampled per SHAP evaluation window.

    shap_drift_threshold:
        Kendall tau below which a SHAP window is flagged as drifted.

    seed:
        Random seed.

    early_stopping_rounds:
        XGBoost validation patience.

    use_correlation_graph:
        Enable Module A.

    use_dynamic_threshold:
        Enable Module B.

    use_shap_drift:
        Enable Module C.
    """

    correlation_window_size: int = 500
    threshold_window_size: int = 500

    shap_window_size: int = 1000
    shap_background_size: int = 500
    shap_sample_size: int = 200
    shap_drift_threshold: float = 0.7

    seed: int = 42
    early_stopping_rounds: int = 50

    # -----------------------------------------------------------------
    # Ablation switches
    # -----------------------------------------------------------------

    use_correlation_graph: bool = True
    use_dynamic_threshold: bool = True
    use_shap_drift: bool = True

    params: dict[str, Any] = field(
        default_factory=lambda: dict(
            SAURABH_XGBOOST_PARAMS
        )
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

    validation_threshold_statistics_: (
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

    fitted_: bool = field(
        default=False,
        init=False,
    )

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def __post_init__(
        self,
    ) -> None:
        """Initialize methodology modules."""

        self.correlation_graph = (
            CorrelationBehaviouralGraph(
                window_size=self.correlation_window_size,
            )
        )

        self.dynamic_threshold = (
            DynamicAdaptiveThreshold(
                window_size=self.threshold_window_size,
            )
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

        Leakage policy
        --------------
        * Module A baseline uses benign TRAINING rows only.
        * XGBoost trains on TRAINING rows only.
        * Early stopping observes VALIDATION data.
        * Module B learns thresholds from VALIDATION only.
        * Module C reference ranking uses TRAINING rows only.
        * TEST data is never used during fit().
        """

        X_train = np.asarray(
            X_train,
            dtype=np.float64,
        )

        y_train = np.asarray(
            y_train,
            dtype=np.int64,
        ).reshape(-1)

        X_val = np.asarray(
            X_val,
            dtype=np.float64,
        )

        y_val = np.asarray(
            y_val,
            dtype=np.int64,
        ).reshape(-1)

        self._validate_xy(
            X_train,
            y_train,
            "training",
        )

        self._validate_xy(
            X_val,
            y_val,
            "validation",
        )

        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                "Training and validation feature dimensions "
                "must match"
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

        self.n_input_features_ = int(
            X_train.shape[1]
        )

        # -------------------------------------------------------------
        # MODULE A — Correlation Behavioural Graph
        # -------------------------------------------------------------

        if self.use_correlation_graph:

            # Static baseline fitted from benign TRAINING rows only.
            self.correlation_graph.fit(
                X_train,
                y_train,
            )

            X_train_model = (
                self.correlation_graph.transform(
                    X_train,
                    block_ids=train_block_ids,
                )
            )

            X_val_model = (
                self.correlation_graph.transform(
                    X_val,
                    block_ids=val_block_ids,
                )
            )

        else:

            # Ablation: original feature space only.
            X_train_model = X_train
            X_val_model = X_val

        self.n_model_features_ = int(
            X_train_model.shape[1]
        )

        # -------------------------------------------------------------
        # XGBOOST
        # -------------------------------------------------------------

        import xgboost as xgb

        self.scale_pos_weight_ = (
            scale_pos_weight_of(
                y_train
            )
        )

        kwargs = dict(
            self.params
        )

        kwargs["random_state"] = self.seed
        kwargs["scale_pos_weight"] = (
            self.scale_pos_weight_
        )

        # XGBoost versions differ slightly in where early stopping is
        # configured. Constructor configuration works for modern versions.
        kwargs["early_stopping_rounds"] = (
            self.early_stopping_rounds
        )

        self.model = xgb.XGBClassifier(
            **kwargs
        )

        self.model.fit(
            X_train_model,
            y_train,
            eval_set=[
                (
                    X_val_model,
                    y_val,
                )
            ],
            verbose=False,
        )

        best_iteration = getattr(
            self.model,
            "best_iteration",
            None,
        )

        self.best_iteration_ = (
            int(best_iteration)
            if best_iteration is not None
            else None
        )

        # -------------------------------------------------------------
        # MODULE B — Dynamic Adaptive Threshold
        #
        # Validation probabilities + validation labels ONLY.
        # -------------------------------------------------------------

        if self.use_dynamic_threshold:

            p_val = self.model.predict_proba(
                X_val_model
            )[:, 1]

            self.dynamic_threshold.fit(
                y_val,
                p_val,
                block_ids=val_block_ids,
            )

            self.validation_threshold_statistics_ = (
                self.dynamic_threshold.threshold_statistics()
            )

        else:

            self.validation_threshold_statistics_ = None

        # -------------------------------------------------------------
        # MODULE C — SHAP Drift Reference
        #
        # Reference feature importance is computed from TRAINING data only.
        # -------------------------------------------------------------

        if self.use_shap_drift:

            self.shap_drift.fit(
                self.model,
                X_train_model,
            )

        self.fitted_ = True

        return self

    # -----------------------------------------------------------------
    # Feature transformation
    # -----------------------------------------------------------------

    def transform_features(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform raw features into the model feature space.

        This is the single shared transformation path used by probability
        prediction and SHAP drift analysis.
        """

        self._check_fitted()

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        self._validate_x(
            X,
            "input",
        )

        self._validate_block_ids(
            block_ids,
            len(X),
            "block_ids",
        )

        if self.use_correlation_graph:

            return self.correlation_graph.transform(
                X,
                block_ids=block_ids,
            )

        return X

    # -----------------------------------------------------------------
    # Probability prediction
    # -----------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return attack-class probabilities.

        Module A is applied before XGBoost prediction.
        """

        X_model = self.transform_features(
            X,
            block_ids=block_ids,
        )

        return self.model.predict_proba(
            X_model
        )[:, 1]

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

        Module B uses thresholds learned from validation data only.
        """

        probabilities = self.predict_proba(
            X,
            block_ids=block_ids,
        )

        if self.use_dynamic_threshold:

            return self.dynamic_threshold.predict(
                probabilities,
                block_ids=block_ids,
            )

        return (
            probabilities >= 0.5
        ).astype(np.int64)

    # -----------------------------------------------------------------
    # SHAP drift analysis
    # -----------------------------------------------------------------

    def detect_drift(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> SHAPDriftResult:
        """
        Run Module C: SHAP feature-importance drift detection.

        No labels are used. This is safe for test-time or deployment-time
        analysis.

        Block IDs are passed through to SHAPDriftDetector so SHAP windows
        never cross contiguous block boundaries.
        """

        self._check_fitted()

        if not self.use_shap_drift:
            raise RuntimeError(
                "SHAP drift detection is disabled "
                "for this ablation run"
            )

        X_model = self.transform_features(
            X,
            block_ids=block_ids,
        )

        return self.shap_drift.transform(
            X_model,
            block_ids=block_ids,
        )

    def detect_shap_drift(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> SHAPDriftResult:
        """Alias for detect_drift()."""

        return self.detect_drift(
            X,
            block_ids=block_ids,
        )

    # -----------------------------------------------------------------
    # Convenience information
    # -----------------------------------------------------------------

    @property
    def n_trees(
        self,
    ) -> int:
        """
        Number of trees effectively used after early stopping.
        """

        self._check_fitted()

        if self.best_iteration_ is not None:
            return (
                self.best_iteration_ + 1
            )

        return int(
            self.params["n_estimators"]
        )

    def summary(
        self,
    ) -> dict[str, Any]:
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

            "modules": {
                "correlation_graph": (
                    self.use_correlation_graph
                ),
                "dynamic_threshold": (
                    self.use_dynamic_threshold
                ),
                "shap_drift": (
                    self.use_shap_drift
                ),
            },

            "correlation_window_size": (
                self.correlation_window_size
            ),

            "threshold_window_size": (
                self.threshold_window_size
            ),

            "shap_window_size": (
                self.shap_window_size
            ),

            "shap_background_size": (
                self.shap_background_size
            ),

            "shap_sample_size": (
                self.shap_sample_size
            ),

            "shap_drift_threshold": (
                self.shap_drift_threshold
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

    def _check_fitted(
        self,
    ) -> None:
        """Raise an error when called before fit()."""

        if not self.fitted_:
            raise RuntimeError(
                "SaurabhXGBoost is not fitted. "
                "Call fit() first."
            )