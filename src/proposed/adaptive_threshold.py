"""Drift-adaptive decision threshold — Member 3, step B.

The faculty brief asks for two things this module joins together:

* "replace the static benign correlation matrix with an EWMA update ...
  use CUSUM on the Frobenius divergence to reset the baseline only when
  benign traffic is confirmed";
* "use grid search **per window** to strictly bound false alarms".

The proposed teacher already carries EWMA and CUSUM, but their adapted state
never reaches a decision: `ProposedXGBoost.predict_proba` deliberately scores
against `fitted_ewma_`, the baseline frozen at the end of training. Feeding the
adapted baseline back in as a *feature* was measured and fails badly — not
because the adaptive signal is weak (it is stronger: window AUC 0.75 streaming
vs 0.67 frozen) but because its absolute scale moves ~0.9 sd between training
and serving, and the trees split on absolute values. See PROJECT_DECISIONS.md.

So this module routes adaptation somewhere a scale shift cannot hurt it: the
**decision threshold**, which is applied to probabilities, not to features.

    XGBoost (frozen features, untouched)  ->  attack probability
    EWMA + CUSUM (streaming)              ->  benign drift state
                                                  |
                                                  v
                                     per-window threshold, FPR <= alpha

Leakage safety, by construction:

* labels are never an argument to anything here;
* a window is judged benign by the *model's own predictions* under the
  threshold in force at the time, never by its labels;
* the threshold used on a window is fixed before that window's scores enter
  the pool, so no window influences its own decision;
* CUSUM freezes both the baseline and the threshold around change points, so
  an attack cannot walk the threshold upward to hide itself.

The threshold can only move **up** from the validation-fitted one. That floor
matters: the label-free rule targets FPR = alpha (0.05), while the trained
threshold already does far better than that on stable traffic (0.0009 on
NCSRD). Without the floor, "adapting" would mean degrading. With it, the
channel is inert until benign traffic actually starts producing high scores,
and then it tightens.

MEASURED RESULT: THIS DOES NOT WORK. KEEP IT OFF BY DEFAULT.
------------------------------------------------------------
Both settings were run on the NCSRD project-standard split and neither is
usable. The reason is a property of the data, not of this code.

*Default (pooled) mode is completely inert.* The threshold never moved once
across all four traffic profiles; FPR was identical to the static rule to four
decimals. The cause is measurable: per-window alarm rates are **bimodal** --
across every stream, **0.0%** of windows fall in the 5-10% band. Windows are
either near-silent (<5% alarms) or loud (>10%). The anti-contamination gate
admits only windows below 10%, and the FPR budget only bites above 5%, so the
gate filters out precisely the windows the budget would react to. The accepted
pool's 95th percentile is 0.007-0.015 against a threshold of 0.57. No setting
of the gate fixes this, because the band between them is empty.

*`window_local=True` fixes FPR and destroys detection.* Capping each window's
own alarm rate at alpha cuts FPR variance 192x under periodic URLLC
(0.021684 -> 0.000113, max 0.80 -> 0.086) -- and drops recall from 0.9728 to
0.6047, missing 2,119 attacks instead of 146.

*Why no middle ground exists.* The two failures are the same fact seen twice.
On the real test split, windows with >10% alarms are **92.3% genuine attacks**;
under the synthetic benign profiles, identical-looking windows are 100% benign
by construction. Same observable, opposite correct response. A threshold rule
sees only scores, so it cannot separate them -- and CUSUM cannot either: it
fires on 14.1% of drifted-benign windows versus 12.5% of ordinary benign
windows, which is no signal at all.

Kept, opt-in and off by default, so the negative result stays reproducible
(`experiments/ncsrd/run_nonstationary.py`). Nothing here touches
`predict_proba`, the P0-P7 ablation, or the 3x2 matrix.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from proposed.constrained_threshold import ConstrainedThreshold


@dataclass
class DriftAdaptiveThreshold:
    """Online FPR-bounded threshold driven by EWMA/CUSUM drift state.

    Wraps a **fitted** `ProposedXGBoost`. It reads the model's probabilities
    and adaptation settings; it never refits it and never mutates it, so the
    P0-P7 ablation and the 3x2 matrix are unaffected by anything here.

    Parameters
    ----------
    model
        A fitted `ProposedXGBoost` (or anything exposing `predict_proba`,
        `ewma`, `cusum`, `selected_threshold_` and the window settings).
    pool_size
        How many recent accepted-benign scores the threshold is solved on.
        Older scores fall out, which is what lets the threshold come back down
        once a drift episode passes.
    min_pool
        Minimum pool size before the threshold is allowed to move at all.
    window_local
        If True, solve the threshold on **each window's own** scores rather
        than on a pool of accepted-benign windows, and drop the
        anti-contamination gate. This is the brief's literal "grid search per
        window", and it is kept because it is the experiment that shows why
        the whole channel does not work -- see the failure note in the module
        docstring. It is not a usable configuration; the default is False.
    """

    model: object
    pool_size: int = 5000
    min_pool: int = 500
    window_local: bool = False

    # Filled in by `predict`. All per-call, so repeated calls are identical.
    threshold_trace_: list[float] = field(default_factory=list, init=False)
    n_windows_: int = field(default=0, init=False)
    n_accepted_: int = field(default=0, init=False)
    n_change_points_: int = field(default=0, init=False)
    n_threshold_changes_: int = field(default=0, init=False)
    base_threshold_: float = field(default=0.5, init=False)

    def __post_init__(self) -> None:
        if self.pool_size < 1:
            raise ValueError("pool_size must be positive.")

        if self.min_pool < 1:
            raise ValueError("min_pool must be positive.")

        if getattr(self.model, "selected_threshold_", None) is None:
            raise RuntimeError(
                "DriftAdaptiveThreshold needs a fitted model with a "
                "selected threshold."
            )

        if getattr(self.model, "ewma", None) is None:
            raise RuntimeError(
                "DriftAdaptiveThreshold needs a model carrying an EWMA "
                "baseline. Use a configuration with use_ewma=True."
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        block_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Classify a stream, adapting the threshold as benign traffic drifts.

        Probabilities come from the model's ordinary `predict_proba`, so the
        classifier -- features, trees, frozen baseline -- is bit-identical to
        the static path. Only the cut applied to those probabilities moves.
        """

        X = np.asarray(X, dtype=np.float64)

        # One frozen-baseline pass. Identical to what the static path scores,
        # which makes the static-vs-adaptive comparison exact.
        probabilities = np.asarray(
            self.model.predict_proba(X, block_ids=block_ids),
            dtype=np.float64,
        ).reshape(-1)

        # Deep copies: adaptation is per-call state, so calling twice gives
        # the same answer and the fitted model is never mutated.
        ewma = copy.deepcopy(self.model.ewma)
        cusum = copy.deepcopy(self.model.cusum)

        selector = ConstrainedThreshold(
            alpha=float(self.model.fpr_alpha),
            threshold_min=float(self.model.threshold_min),
            threshold_max=float(self.model.threshold_max),
            threshold_step=float(self.model.threshold_step),
        )

        base_threshold = float(self.model.selected_threshold_)
        threshold = base_threshold
        gate = float(self.model.benign_update_fraction)

        pool: deque[float] = deque(maxlen=self.pool_size)

        self.threshold_trace_ = []
        self.n_windows_ = 0
        self.n_accepted_ = 0
        self.n_change_points_ = 0
        self.n_threshold_changes_ = 0
        self.base_threshold_ = base_threshold

        predictions = np.zeros(X.shape[0], dtype=int)

        for start, end in self.model._window_ranges(X, block_ids):
            window = X[start:end]
            window_p = probabilities[start:end]

            if self.window_local:
                # Cap this window's own false-alarm rate at alpha. Still
                # label-free -- it reads predictions only -- but it also
                # tightens on genuine attack windows, which is exactly the
                # failure this mode exists to demonstrate.
                previous = threshold

                threshold = max(
                    base_threshold,
                    selector.fit_benign_only(window_p),
                )

                if threshold != previous:
                    self.n_threshold_changes_ += 1

            # Decide with the threshold already in force. The current window
            # cannot influence its own decision.
            predictions[start:end] = (
                window_p >= threshold
            ).astype(int)

            self.n_windows_ += 1
            self.threshold_trace_.append(threshold)

            if window.shape[0] < 2:
                # Too short for a correlation matrix; no adaptation possible.
                continue

            divergence = ewma.divergence(window)

            change_detected = cusum.update(divergence)

            if change_detected:
                self.n_change_points_ += 1

            # Predicted-benign fraction under the threshold in force.
            # Predictions, never labels.
            benign_fraction = float(
                np.mean(window_p < threshold)
            )

            if change_detected or benign_fraction < gate:
                # Suspicious or unstable: freeze the baseline AND the
                # threshold. This is the anti-contamination gate.
                continue

            self.n_accepted_ += 1

            ewma.update(window)

            if self.window_local:
                # The threshold was already set from this window alone.
                continue

            pool.extend(window_p.tolist())

            if len(pool) < self.min_pool:
                continue

            # Neyman-Pearson on the accepted-benign pool: smallest cut whose
            # empirical FPR respects alpha. Floored at the validation
            # threshold so adaptation can only tighten, never loosen.
            candidate = max(
                base_threshold,
                selector.fit_benign_only(np.fromiter(pool, dtype=float)),
            )

            if candidate != threshold:
                self.n_threshold_changes_ += 1
                threshold = candidate

        return predictions

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, float | int]:
        """Threshold behaviour over the last `predict` call."""

        trace = np.asarray(
            self.threshold_trace_,
            dtype=float,
        )

        if trace.size == 0:
            raise RuntimeError(
                "No prediction has been made yet."
            )

        return {
            "base_threshold": float(self.base_threshold_),
            "threshold_mean": float(trace.mean()),
            "threshold_min": float(trace.min()),
            "threshold_max": float(trace.max()),
            "threshold_var": float(trace.var()),
            "n_windows": int(self.n_windows_),
            "n_accepted": int(self.n_accepted_),
            "n_change_points": int(self.n_change_points_),
            "n_threshold_changes": int(self.n_threshold_changes_),
        }
