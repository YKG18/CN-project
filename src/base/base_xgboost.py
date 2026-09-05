"""Base-paper XGBoost — Member 1.

Reproduction of the XGBoost classifier in Xylouris et al., "Advancing Predictive
Security for Consumer Applications in Beyond 5G/6G Networks", IEEE TCE 2025,
Section IV.D. There is no public implementation of that paper; this is
reconstructed from the paper text, its Table II feature list, and the public
NCSRD-DS-5GDDoS dataset.

This is the `base_paper` lineage. On NCSRD it uses the **base38** feature set
and must never be mixed with `saurabh49` (docs/PROJECT_DECISIONS.md, D1). The
same methodology is reused on Data4Cyber via
`experiments/data4cyber/run_base.py`, with that dataset's own electrical
telemetry -- the model and protocol transfer, the feature schema does not (D7).

What the paper specifies (IV.D), and what we do:

| Paper | Here |
|---|---|
| 38 features | `base38` from `ncsrd_prep` (Data4Cyber: its own 140 columns) |
| imputation, label encoding, StandardScaler | done in `ncsrd_prep` |
| "custom undersampling strategy, reducing the majority class while retaining all minority class samples" | `undersample_majority()`, ratio selected on validation |
| "scale_pos_weight tuned to emphasize the minority class" | derived from the post-resampling counts |
| "trained with logloss as the evaluation metric" | `eval_metric="logloss"` |
| "early stopping rounds were configured, monitoring validation performance" | `early_stopping_rounds` on the validation split |
| "divided into training, validation, and test sets" | three-way split |

Documented deviations (the paper does not specify these):

* **Tree hyperparameters.** Depth, learning rate and subsampling are not given.
  We use standard values and let early stopping choose the tree count.
* **Cross-validation.** The paper says "we have used cross-validation to train
  the models" but also describes early stopping on a validation split. We use a
  single held-out validation split for both hyperparameter choice and early
  stopping, which is the usual simplification when early stopping is required.
* **Undersampling ratio.** "Custom sampling strategy" is not quantified, so the
  ratio is chosen on validation from a small grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

#: Tree hyperparameters. The paper fixes only `eval_metric` and the use of
#: early stopping; the rest are standard values. `n_estimators` is an upper
#: bound -- early stopping picks the actual number.
BASE_PAPER_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

#: Majority:minority ratios tried when choosing the "custom sampling strategy".
#: 1.0 is a balanced 1:1 set; 14.93 would be the untouched distribution.
UNDERSAMPLE_RATIOS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


def undersample_majority(X: np.ndarray, y: np.ndarray, ratio: float,
                         seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random undersampling that keeps **every** minority sample.

    The base paper's rule is "reduce the majority class while retaining all
    minority class samples", so which class gets thinned is decided by the data,
    not hard-coded. On NCSRD benign is the majority (6.28 % attack); on
    Data4Cyber the *attack* class is the majority (~61 %), and the rule then
    thins attacks instead.

    `ratio` is majority:minority after resampling, so `ratio=1.0` gives a
    balanced set. If the majority class is already scarcer than the request it
    is left untouched. Row order is shuffled so XGBoost's subsampling does not
    see a class-ordered set.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    # Order of the rng calls below is identical in both branches, so the
    # benign-majority case reproduces earlier NCSRD runs exactly.
    minority, majority = (pos, neg) if len(neg) >= len(pos) else (neg, pos)
    n_keep = min(len(majority), int(round(len(minority) * ratio)))
    keep = np.concatenate([minority, rng.choice(majority, size=n_keep, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]


def scale_pos_weight_of(y: np.ndarray) -> float:
    """`n_negative / n_positive` — the XGBoost imbalance knob."""
    pos = int((y == 1).sum())
    return float((y == 0).sum() / pos) if pos else 1.0


@dataclass
class BaseXGBoost:
    """The base-paper XGBoost configuration.

    Exposes `fit` / `predict_proba` / `predict` so the eventual common runner
    can treat every method interchangeably (PROJECT_DECISIONS.md conventions).
    """

    undersample_ratio: float = 1.0
    seed: int = 42
    early_stopping_rounds: int = 50
    params: dict[str, Any] = field(default_factory=lambda: dict(BASE_PAPER_PARAMS))
    #: Decision threshold. Set by `select_threshold` on VALIDATION, never test.
    threshold: float = 0.5

    model: Any = field(default=None, init=False, repr=False)
    train_counts_: dict[str, int] = field(default_factory=dict, init=False)
    scale_pos_weight_: float = field(default=1.0, init=False)
    best_iteration_: int | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "BaseXGBoost":
        """Undersample the majority class, then fit with early stopping."""
        import xgboost as xgb

        Xr, yr = undersample_majority(X, y, self.undersample_ratio, self.seed)
        self.train_counts_ = {"benign": int((yr == 0).sum()),
                              "attack": int((yr == 1).sum())}
        self.scale_pos_weight_ = scale_pos_weight_of(yr)

        kwargs = dict(self.params)
        kwargs["random_state"] = self.seed
        kwargs["scale_pos_weight"] = self.scale_pos_weight_
        if X_val is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBClassifier(**kwargs)
        fit_kw = {"verbose": False}
        if X_val is not None:
            fit_kw["eval_set"] = [(X_val, y_val)]
        self.model.fit(Xr, yr, **fit_kw)
        self.best_iteration_ = getattr(self.model, "best_iteration", None)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(attack) for each row, shape (n,)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """0/1 at the configured threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    @property
    def n_trees(self) -> int:
        if self.best_iteration_ is not None:
            return int(self.best_iteration_) + 1
        return int(self.params["n_estimators"])


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """F1-optimal threshold for the ATTACK class, chosen on VALIDATION.

    Never call this with test labels — D5 forbids selecting the operating point
    on the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))          # last point has no threshold
    return float(thresholds[best])


def evaluate(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Metrics in the D6 definitions.

    Primary metrics are for the attack class (label 1); `*_0` covers the benign
    class so the output lines up with the paper's Table III. FPR = FP/(FP+TN).
    """
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        # attack class (primary)
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        # benign class (Table III reports it too)
        "precision_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        # secondary
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, p),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
