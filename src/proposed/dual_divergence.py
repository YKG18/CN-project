"""Dual-divergence configuration (P8) — Member 3, step C.

The question this answers, and nothing else:

    Does an *adapting* correlation baseline carry information the existing
    fixed-reference divergence does not already have?

Both earlier attempts to make the online baseline influence classification
failed (see docs/PROJECT_DECISIONS.md):

* replacing the frozen divergence with a streaming one -- the raw streaming
  value shifts ~0.91 sd between training and serving, and the trees split on
  absolute values, so F1 fell to 0.2002;
* routing adaptation into the decision threshold instead -- inert in its safe
  form, and recall 0.9728 -> 0.6047 in its aggressive one.

This configuration keeps the frozen feature exactly as it is and *adds* a
second one beside it:

    [ 49 base features | d_frozen | d_adaptive ]   -> 51 columns

`d_frozen` is byte-identical to what P6 uses, so nothing the current model
relies on is taken away.

MEASURED RESULT (NCSRD, project-standard split)
-----------------------------------------------
Each row differs from the control in exactly one column, so each gap is that
column's doing:

    config                          P        R       F1      FPR    gain(adp)
    P8-control (frozen only)   0.9856   0.9728   0.9792   0.0009        --
    P8-dual  (EWMA z)          0.6126   0.6623   0.6365   0.0278     0.4499
    P8-dual  (lagged)          0.9979   0.9890   0.9934   0.0001     0.0149

The control reproduces P6 exactly, so the comparison is fair.

**`lagged` works; `ewma_z` does not.** The deciding property is *statefulness*,
not adaptiveness -- both references move with recent traffic:

* `ewma_z` carries state that accumulates over the whole stream, so the
  feature's meaning depends on how far into the stream a row sits. Training and
  test sit at different points, so the trees' split points do not transfer. The
  model still spends 45% of its gain on it and is misled: F1 0.6365.
  Aligning the mean and sd (0.91 sd -> 0.15 sd) is necessary but *not*
  sufficient -- the feature's relationship to the label also shifts (train
  window-AUC 0.799 vs test 0.678).
* `lagged` compares consecutive windows, so its reference is always exactly one
  window back, everywhere. Stationary by construction, and it helps: false
  positives 76 -> 8 and false negatives 146 -> 59.

Two controls rule out the obvious alternative explanations. A perfect (oracle)
benign gate on `ewma_z` is *worse*, not better (F1 0.0000), so contamination is
not the cause. A second *fixed* divergence column takes comparable gain (0.43)
and is nearly harmless (F1 0.9714), so the cost is not "a second divergence
column" in general.

Train/serve consistency is structural, not a convention: `_stream()` builds the
training, validation and test matrices with the same code, the same gate and
one continuous piece of state. There is no path by which the two can diverge.

Leakage safety:

* the stream walk uses **no labels at any point** -- the baseline gate is
  CUSUM alone, so serving behaves exactly as training did;
* labels are used only where the project already allows it: CUSUM calibration
  on training windows, and threshold selection on validation (D5);
* prediction deep-copies the fitted state, so repeated calls agree and the
  fitted model is never mutated.

Known limitation of `lagged`: a window with no predecessor inside its own block
gets 0.0, and blocks are short relative to the window. On NCSRD ~34% of windows
(66 of 196 in test) are block-firsts; on Data4Cyber the 120-row blocks equal the
default window, so the column is constant and carries nothing unless a smaller
window is used. The feature needs at least two windows per block to exist.

Opt-in and self-contained. It shares no code path with `ProposedXGBoost`, so
P0-P7 and the 3x2 matrix cannot be affected by anything here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from proposed.constrained_threshold import ConstrainedThreshold
from proposed.cusum import CUSUMChangeDetector
from proposed.ewma import EWMACorrelationBaseline
from saurabh.saurabh_xgboost import SaurabhXGBoost


def window_ranges(
    n_rows: int,
    block_ids: np.ndarray | None,
    size: int,
) -> Iterator[tuple[int, int]]:
    """Yield `(start, end)` correlation windows that never cross a block.

    Same convention as `ProposedXGBoost._window_ranges`: fixed-size chunks
    inside each block run, with a possibly-short trailing chunk. Reimplemented
    here rather than imported so this module stays independent of the P6 class
    -- the point of the configuration is that it cannot disturb P6.
    """

    if n_rows == 0:
        return

    if block_ids is None:
        starts, ends = np.array([0]), np.array([n_rows])
    else:
        block_ids = np.asarray(block_ids).reshape(-1)

        if block_ids.shape[0] != n_rows:
            raise ValueError("block_ids length must match X.")

        changes = np.flatnonzero(block_ids[1:] != block_ids[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [n_rows]))

    for run_start, run_end in zip(starts, ends):
        position = int(run_start)

        while position < int(run_end):
            end = min(position + size, int(run_end))
            yield position, end
            position = end


@dataclass
class StreamState:
    """The adapting half of the feature builder.

    `mean`/`m2`/`count` are Welford running moments over the divergences of
    accepted windows. Standardising against them is what makes the adaptive
    column comparable between training and serving -- a fixed affine transform
    would not, because the raw distribution itself moves.
    """

    ewma: EWMACorrelationBaseline
    cusum: CUSUMChangeDetector
    mean: float = 0.0
    m2: float = 0.0
    count: int = 0
    #: Previous window's correlation matrix, for `adaptive_kind="lagged"`.
    #: Reset at every block boundary so no comparison spans a gap in time.
    prev_corr: np.ndarray | None = None

    def standardise(self, divergence: float) -> float:
        """z-score `divergence` against the windows accepted so far."""

        if self.count < 2:
            return 0.0

        variance = self.m2 / (self.count - 1)

        return float(
            (divergence - self.mean) / max(np.sqrt(variance), 1e-9)
        )

    def observe(self, divergence: float) -> None:
        """Fold an accepted window's divergence into the running moments."""

        self.count += 1
        delta = divergence - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (divergence - self.mean)


@dataclass
class DualDivergenceXGBoost:
    """XGBoost over [base features | frozen divergence | adaptive divergence].

    Parameters
    ----------
    use_adaptive_feature
        When False the adaptive column is omitted entirely, giving a 50-column
        control that is identical in every other respect. That control is the
        comparison the whole configuration exists to make -- it isolates the
        adaptive column and nothing else.
    adaptive_kind
        Which adaptive column to add.

        ``"ewma_z"``
            Streaming EWMA divergence, standardised against a running mean and
            variance. Adapts with an accumulating baseline.
        ``"lagged"``
            Frobenius distance between this window's correlation matrix and the
            **previous window's**, reset at each block boundary.

        The difference that matters is statefulness. ``ewma_z`` carries state
        that grows over the whole stream, so the feature's meaning depends on
        how far into the stream a row sits -- and training and test sit at
        different points. ``lagged`` is adaptive in the same sense (its
        reference is recent traffic, not a fixed baseline) but is **stationary
        by construction**: the reference is always exactly one window back,
        everywhere, so its distribution cannot drift between train and serve.
    """

    seed: int = 42
    correlation_window_size: int = 500
    ewma_alpha: float = 0.05
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    fpr_alpha: float = 0.05
    threshold_min: float = 0.50
    threshold_max: float = 0.99
    threshold_step: float = 0.01
    early_stopping_rounds: int = 50
    use_adaptive_feature: bool = True
    adaptive_kind: str = "ewma_z"
    params: dict[str, Any] = field(default_factory=dict)

    backbone: SaurabhXGBoost = field(default=None, init=False, repr=False)
    model: Any = field(default=None, init=False, repr=False)
    frozen_ewma_: EWMACorrelationBaseline = field(
        default=None, init=False, repr=False)
    fitted_state_: StreamState = field(default=None, init=False, repr=False)
    constrained_threshold: ConstrainedThreshold = field(
        default=None, init=False, repr=False)

    fitted_: bool = field(default=False, init=False)
    selected_threshold_: float | None = field(default=None, init=False)
    n_input_features_: int | None = field(default=None, init=False)
    n_model_features_: int | None = field(default=None, init=False)
    train_ewma_updates_: int = field(default=0, init=False)
    train_drift_events_: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.correlation_window_size < 2:
            raise ValueError("correlation_window_size must be >= 2.")

        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1].")

        if self.adaptive_kind not in ("ewma_z", "lagged"):
            raise ValueError(
                "adaptive_kind must be 'ewma_z' or 'lagged', "
                f"got {self.adaptive_kind!r}."
            )

        backbone_kwargs: dict[str, Any] = {
            "correlation_window_size": self.correlation_window_size,
            "threshold_window_size": 500,
            "shap_window_size": 1000,
            "shap_background_size": 500,
            "shap_sample_size": 200,
            "shap_drift_threshold": 0.7,
            "seed": self.seed,
            "early_stopping_rounds": self.early_stopping_rounds,
            # Saurabh's own modules stay off; this class supplies the features.
            "use_correlation_graph": False,
            "use_dynamic_threshold": False,
            "use_shap_drift": False,
        }

        if self.params:
            backbone_kwargs["params"] = dict(self.params)

        self.backbone = SaurabhXGBoost(**backbone_kwargs)

        self.constrained_threshold = ConstrainedThreshold(
            alpha=self.fpr_alpha,
            threshold_min=self.threshold_min,
            threshold_max=self.threshold_max,
            threshold_step=self.threshold_step,
        )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _stream(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None,
        state: StreamState,
    ) -> np.ndarray:
        """Build the augmented matrix, advancing `state` as the stream runs.

        The single code path used for training, validation and test. `state`
        is mutated, so callers that must not advance the fitted state pass a
        deep copy.
        """

        extra = 2 if self.use_adaptive_feature else 1
        out = np.zeros((X.shape[0], X.shape[1] + extra), dtype=np.float64)

        block_of = (None if block_ids is None
                    else np.asarray(block_ids).reshape(-1))
        current_block = None

        for start, end in window_ranges(
            X.shape[0], block_ids, self.correlation_window_size
        ):
            window = X[start:end]
            rows = end - start

            # A new block is a jump in time, so the previous window is no
            # longer "the recent past". Drop the lag reference across gaps.
            if block_of is not None and block_of[start] != current_block:
                current_block = block_of[start]
                state.prev_corr = None

            if rows < 2:
                # Too short for a correlation matrix; leave the extra columns
                # at zero and do not disturb the adaptive state.
                out[start:end, : X.shape[1]] = window
                continue

            frozen = self.frozen_ewma_.divergence(window)

            columns = [np.full(rows, frozen, dtype=np.float64)]

            if self.use_adaptive_feature and self.adaptive_kind == "lagged":
                current = EWMACorrelationBaseline._correlation_matrix(window)

                # First window of a block has no predecessor; 0.0 reads as
                # "no change observed", which is the correct neutral value.
                lagged = (
                    0.0 if state.prev_corr is None
                    else float(np.linalg.norm(current - state.prev_corr,
                                              ord="fro"))
                )

                columns.append(np.full(rows, lagged, dtype=np.float64))
                state.prev_corr = current

            elif self.use_adaptive_feature:
                adaptive = state.ewma.divergence(window)

                # Standardise BEFORE this window updates the moments, so the
                # value never depends on itself.
                columns.append(
                    np.full(rows, state.standardise(adaptive),
                            dtype=np.float64)
                )

                # CUSUM-only gate: no labels, so training and serving behave
                # identically. This is the brief's anti-contamination rule.
                if not state.cusum.update(adaptive):
                    state.ewma.update(window)
                    state.observe(adaptive)

            out[start:end] = np.column_stack((window, *columns))

        return out

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_block_ids: np.ndarray | None = None,
        val_block_ids: np.ndarray | None = None,
    ) -> "DualDivergenceXGBoost":
        """Fit on the project-standard split."""

        X_train = np.asarray(X_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_train = np.asarray(y_train).reshape(-1).astype(int)
        y_val = np.asarray(y_val).reshape(-1).astype(int)

        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("X_train and X_val feature counts differ.")

        self.n_input_features_ = int(X_train.shape[1])

        # 1. Frozen reference baseline: training benign rows only (D2).
        benign = X_train[y_train == 0]

        if benign.shape[0] < 2:
            raise ValueError(
                "Need at least two benign training rows for the baseline."
            )

        self.frozen_ewma_ = EWMACorrelationBaseline(
            alpha=self.ewma_alpha
        ).initialize(benign)

        # 2. Calibrate CUSUM on all-benign training windows. Training labels
        #    only -- never validation or test.
        calibration = [
            self.frozen_ewma_.divergence(X_train[s:e])
            for s, e in window_ranges(
                X_train.shape[0], train_block_ids, self.correlation_window_size
            )
            if e - s >= 2 and np.all(y_train[s:e] == 0)
        ]

        if not calibration:
            calibration = [0.0, 1.0]

        cusum = CUSUMChangeDetector(k=self.cusum_k, h=self.cusum_h)
        cusum.fit(np.asarray(calibration, dtype=float))
        cusum.reset()

        # 3. One continuous walk: train, then validation. Test continues from
        #    the state this leaves behind.
        state = StreamState(
            ewma=copy.deepcopy(self.frozen_ewma_),
            cusum=cusum,
        )

        X_train_model = self._stream(X_train, train_block_ids, state)
        self.train_ewma_updates_ = state.count
        self.train_drift_events_ = state.cusum.n_changes_

        X_val_model = self._stream(X_val, val_block_ids, state)

        self.n_model_features_ = int(X_train_model.shape[1])
        self.fitted_state_ = copy.deepcopy(state)

        # 4. Teacher.
        self.backbone.fit(X_train_model, y_train, X_val_model, y_val)
        self.model = self.backbone.model

        # 5. FPR-constrained threshold on validation only (D5).
        p_val = self.model.predict_proba(X_val_model)[:, 1]
        self.constrained_threshold.fit(y_val, p_val)
        self.selected_threshold_ = float(
            self.constrained_threshold.threshold_
        )

        self.fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Attack probabilities, continuing the stream from the fitted state."""

        self._check_fitted()

        X = np.asarray(X, dtype=np.float64)

        if X.shape[1] != self.n_input_features_:
            raise ValueError(
                f"Expected {self.n_input_features_} features, "
                f"received {X.shape[1]}."
            )

        # Deep copy: the call advances a private copy, so repeated calls agree
        # and the fitted state survives untouched.
        state = copy.deepcopy(self.fitted_state_)

        X_model = self._stream(X, block_ids, state)

        return self.model.predict_proba(X_model)[:, 1]

    def predict(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Binary predictions at the validation-selected threshold."""

        probabilities = self.predict_proba(X, block_ids=block_ids)

        return (probabilities >= self.selected_threshold_).astype(int)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_gain(self) -> dict[str, float]:
        """Share of total XGBoost gain taken by each divergence column.

        Read this alongside the held-out metrics, never instead of them. The
        measurements here show gain is a poor proxy for usefulness: the
        `lagged` column earns 1.5% of gain and improves every metric, while
        `ewma_z` earns 45% and collapses F1 to 0.6365. High gain means the
        trees leaned on a feature during training, which is exactly what makes
        a feature that shifts at serve time dangerous rather than valuable.
        """

        self._check_fitted()

        booster = self.model.get_booster()
        scores = booster.get_score(importance_type="total_gain")
        total = sum(scores.values()) or 1.0

        n = self.n_model_features_
        frozen_index = n - 2 if self.use_adaptive_feature else n - 1

        out = {
            "frozen_divergence": scores.get(f"f{frozen_index}", 0.0) / total,
            "n_features_used": float(len(scores)),
        }

        if self.use_adaptive_feature:
            out["adaptive_divergence"] = scores.get(f"f{n - 1}", 0.0) / total

        return out

    def _check_fitted(self) -> None:
        if not self.fitted_ or self.model is None:
            raise RuntimeError(
                "DualDivergenceXGBoost is not fitted. Call fit() first."
            )
