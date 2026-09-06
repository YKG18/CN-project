"""
data4cyber_adapter.py — One Data4Cyber adapter, four model families.

Consumes the frozen splits produced by data4cyber_prep.py and exposes them in
the tensor shapes required by XGBoost, MLP, CNN and LSTM.

Unlike ncsrd_adapter.py, this adapter DOES NOT create new splits. Data4Cyber
splits are created during preprocessing because temporal/block boundaries must
be preserved to prevent leakage.

For row-level methodology modules that require temporal provenance, block,
scenario and timestamp metadata are preserved inside DataBundle.meta.

Usage
-----
    from common.data.data4cyber_adapter import Data4CyberAdapter

    ad = Data4CyberAdapter(split_mode="block")

    xgb = ad.for_xgboost()
    mlp = ad.for_mlp()
    cnn = ad.for_cnn()
    lstm = ad.for_lstm()
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DATA4CYBER_PROCESSED  # noqa: E402


SplitMode = Literal["block", "scenario"]
Balance = Literal["none", "oversample", "undersample", "class_weight"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Everything a model needs, in the shape expected by that model."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None

    model: str = ""
    feature_names: list[str] = field(default_factory=list)
    input_shape: tuple[int, ...] = ()
    class_weight: dict[int, float] = field(default_factory=dict)
    scale_pos_weight: float = 1.0

    # Dataset and split provenance.
    #
    # Row-level bundles contain:
    #   train_block, val_block, test_block
    #   train_scenario, val_scenario, test_scenario
    #   train_timestamp, val_timestamp, test_timestamp
    #
    # These remain aligned with X/y unless training oversampling or
    # undersampling is explicitly requested, in which case the corresponding
    # training metadata is sampled using the same indices.
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Human-readable summary for sanity checking."""

        lines = [
            f"DataBundle(model={self.model!r})",
            f"  input_shape      : {self.input_shape}",
            f"  X_train          : {self.X_train.shape}",
            f"  y_train          : "
            f"{np.bincount(self.y_train.astype(int), minlength=2).tolist()}",
        ]

        if self.X_val is not None:
            lines.extend([
                f"  X_val            : {self.X_val.shape}",
                f"  y_val            : "
                f"{np.bincount(self.y_val.astype(int), minlength=2).tolist()}",
            ])

        lines.extend([
            f"  X_test           : {self.X_test.shape}",
            f"  y_test           : "
            f"{np.bincount(self.y_test.astype(int), minlength=2).tolist()}",
            f"  scale_pos_weight : {self.scale_pos_weight:.4f}",
            f"  class_weight     : {self.class_weight}",
            f"  split_mode       : {self.meta.get('split_mode')}",
            f"  balance          : {self.meta.get('balance')}",
        ])

        if self.meta.get("level") == "row":
            lines.append(
                "  block_metadata   : "
                f"{all(k in self.meta for k in ('train_block', 'val_block', 'test_block'))}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Data4CyberAdapter:
    """Load frozen Data4Cyber splits and reshape them for each model family.

    Parameters
    ----------
    split_mode:
        "block" is the primary within-dataset evaluation.
        "scenario" is the secondary novel-attack robustness experiment.

    processed_dir:
        Root directory produced by data4cyber_prep.py.

    dtype:
        Output feature dtype.
    """

    def __init__(
        self,
        split_mode: SplitMode = "block",
        processed_dir: str | Path = DATA4CYBER_PROCESSED,
        dtype: type = np.float32,
        verbose: bool = True,
    ):
        if split_mode not in ("block", "scenario"):
            raise ValueError(
                f"split_mode must be 'block' or 'scenario', got {split_mode!r}"
            )

        self.split_mode = split_mode
        self.dtype = dtype
        self.verbose = verbose

        self.root = Path(processed_dir) / split_mode

        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found.\n"
                f"Build Data4Cyber first:\n"
                f"    python src/common/data/data4cyber_prep.py "
                f"--split-mode {split_mode}"
            )

        manifest_path = self.root / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing manifest: {manifest_path}"
            )

        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        self.feature_names = self.manifest.get(
            "features",
            [],
        )

        self._log(
            f"[data4cyber] split_mode={split_mode} "
            f"features={len(self.feature_names)}"
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_rows(
        self,
        split: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Load frozen row-level samples and temporal provenance metadata.

        Metadata arrays remain row-aligned with X and y.
        """

        path = self.root / f"{split}_rows.npz"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing split file: {path}"
            )

        with np.load(path, allow_pickle=False) as z:
            X = z["X"].astype(self.dtype)
            y = z["y"].astype(np.int8)

            metadata = {
                "scenario": z["scenario"].astype(str),
                "block": z["block"].astype(str),
                "timestamp": z["timestamp"].astype(str),
            }

        n_samples = len(X)

        for name, values in metadata.items():
            if len(values) != n_samples:
                raise RuntimeError(
                    f"{split} metadata field {name!r} has "
                    f"{len(values)} rows but X has {n_samples}"
                )

        return X, y, metadata

    def _load_windows(
        self,
        split: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load frozen temporal windows."""

        path = self.root / f"{split}_windows.npz"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing window file: {path}"
            )

        with np.load(path, allow_pickle=False) as z:
            X = z["X"].astype(self.dtype)
            y = z["y"].astype(np.int8)

        return X, y

    # ------------------------------------------------------------------
    # Imbalance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _class_weight(
        y: np.ndarray,
    ) -> dict[int, float]:

        classes = np.unique(y)

        if len(classes) < 2:
            return {0: 1.0, 1: 1.0}

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y,
        )

        return {
            int(c): float(w)
            for c, w in zip(classes, weights)
        }

    @staticmethod
    def _scale_pos_weight(
        y: np.ndarray,
    ) -> float:

        pos = int((y == 1).sum())
        neg = int((y == 0).sum())

        return float(neg / pos) if pos else 1.0

    def _balance_indices(
        self,
        y: np.ndarray,
        how: Balance,
        random_state: int,
    ) -> np.ndarray:
        """
        Return training indices after optional balancing.

        Keeping balancing as an index operation ensures metadata remains
        perfectly aligned with X and y.
        """

        n_samples = len(y)

        if how in ("none", "class_weight"):
            return np.arange(
                n_samples,
                dtype=np.int64,
            )

        rng = np.random.default_rng(random_state)

        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)

        if len(pos) == 0 or len(neg) == 0:
            return np.arange(
                n_samples,
                dtype=np.int64,
            )

        minority, majority = (
            (pos, neg)
            if len(pos) <= len(neg)
            else (neg, pos)
        )

        if how == "oversample":

            extra = rng.choice(
                minority,
                size=len(majority) - len(minority),
                replace=True,
            )

            keep = np.concatenate([
                majority,
                minority,
                extra,
            ])

        elif how == "undersample":

            sampled_majority = rng.choice(
                majority,
                size=len(minority),
                replace=False,
            )

            keep = np.concatenate([
                minority,
                sampled_majority,
            ])

        else:
            raise ValueError(
                f"Unknown balance mode: {how!r}"
            )

        rng.shuffle(keep)

        self._log(
            f"[balance] {how} -> "
            f"{np.bincount(y[keep].astype(int), minlength=2).tolist()}"
        )

        return keep.astype(np.int64)

    def _balance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        how: Balance,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward-compatible balancing helper."""

        keep = self._balance_indices(
            y,
            how,
            random_state,
        )

        return X[keep], y[keep]

    # ------------------------------------------------------------------
    # Row-level models
    # ------------------------------------------------------------------

    def _tabular_bundle(
        self,
        model: str,
        balance: Balance,
        random_state: int,
        reshape: str,
    ) -> DataBundle:

        X_train, y_train, train_meta = self._load_rows(
            "train"
        )
        X_val, y_val, val_meta = self._load_rows(
            "validation"
        )
        X_test, y_test, test_meta = self._load_rows(
            "test"
        )

        # --------------------------------------------------------------
        # Balance training data only.
        #
        # Apply the SAME indices to training metadata so block provenance
        # stays aligned with training rows.
        # --------------------------------------------------------------

        keep = self._balance_indices(
            y_train,
            balance,
            random_state,
        )

        X_train = X_train[keep]
        y_train = y_train[keep]

        train_meta = {
            name: values[keep]
            for name, values in train_meta.items()
        }

        n_features = X_train.shape[1]

        if reshape == "2d":
            input_shape = (n_features,)

        elif reshape == "3d":

            input_shape = (n_features, 1)

            X_train = X_train.reshape(
                -1,
                n_features,
                1,
            )
            X_val = X_val.reshape(
                -1,
                n_features,
                1,
            )
            X_test = X_test.reshape(
                -1,
                n_features,
                1,
            )

        else:
            raise ValueError(
                f"Unknown reshape: {reshape}"
            )

        return DataBundle(
            X_train=X_train,
            y_train=y_train,

            X_val=X_val,
            y_val=y_val,

            X_test=X_test,
            y_test=y_test,

            model=model,
            feature_names=list(
                self.feature_names
            ),
            input_shape=input_shape,

            class_weight=self._class_weight(
                y_train
            ),
            scale_pos_weight=self._scale_pos_weight(
                y_train
            ),

            meta={
                "split_mode": self.split_mode,
                "balance": balance,
                "level": "row",

                # Block IDs required by temporal methodology modules.
                "train_block": train_meta["block"],
                "val_block": val_meta["block"],
                "test_block": test_meta["block"],

                # Additional provenance.
                "train_scenario": train_meta["scenario"],
                "val_scenario": val_meta["scenario"],
                "test_scenario": test_meta["scenario"],

                "train_timestamp": train_meta["timestamp"],
                "val_timestamp": val_meta["timestamp"],
                "test_timestamp": test_meta["timestamp"],
            },
        )

    def for_xgboost(
        self,
        balance: Balance = "class_weight",
        random_state: int = 42,
    ) -> DataBundle:
        """2-D row-level bundle for XGBoost."""

        return self._tabular_bundle(
            model="xgboost",
            balance=balance,
            random_state=random_state,
            reshape="2d",
        )

    def for_mlp(
        self,
        balance: Balance = "class_weight",
        random_state: int = 42,
    ) -> DataBundle:
        """2-D row-level bundle for an MLP."""

        return self._tabular_bundle(
            model="mlp",
            balance=balance,
            random_state=random_state,
            reshape="2d",
        )

    def for_cnn(
        self,
        balance: Balance = "class_weight",
        random_state: int = 42,
    ) -> DataBundle:
        """3-D bundle (samples, features, channels) for 1-D CNN."""

        return self._tabular_bundle(
            model="cnn",
            balance=balance,
            random_state=random_state,
            reshape="3d",
        )

    # ------------------------------------------------------------------
    # Temporal model
    # ------------------------------------------------------------------

    def for_lstm(
        self,
        balance: Balance = "class_weight",
        random_state: int = 42,
    ) -> DataBundle:
        """
        Load pre-built temporal windows for LSTM.

        Windows were created during preprocessing strictly within their
        assigned block/scenario, so no sequence crosses a split boundary.
        """

        X_train, y_train = self._load_windows(
            "train"
        )
        X_val, y_val = self._load_windows(
            "validation"
        )
        X_test, y_test = self._load_windows(
            "test"
        )

        # Oversampling/undersampling whole windows is safe.
        if balance not in ("none", "class_weight"):

            keep = self._balance_indices(
                y_train,
                balance,
                random_state,
            )

            X_train = X_train[keep]
            y_train = y_train[keep]

        input_shape = tuple(
            X_train.shape[1:]
        )

        return DataBundle(
            X_train=np.ascontiguousarray(
                X_train,
                dtype=self.dtype,
            ),
            y_train=y_train,

            X_val=np.ascontiguousarray(
                X_val,
                dtype=self.dtype,
            ),
            y_val=y_val,

            X_test=np.ascontiguousarray(
                X_test,
                dtype=self.dtype,
            ),
            y_test=y_test,

            model="lstm",
            feature_names=list(
                self.feature_names
            ),
            input_shape=input_shape,

            class_weight=self._class_weight(
                y_train
            ),
            scale_pos_weight=self._scale_pos_weight(
                y_train
            ),

            meta={
                "split_mode": self.split_mode,
                "balance": balance,
                "level": "window",
                "window_size": self.manifest.get(
                    "window_size_seconds"
                ),
                "stride": self.manifest.get(
                    "stride_seconds"
                ),
            },
        )

    # ------------------------------------------------------------------

    def get(
        self,
        model: Literal[
            "xgboost",
            "xgb",
            "mlp",
            "cnn",
            "lstm",
        ],
        **kwargs,
    ) -> DataBundle:
        """Dispatch adapter by model name."""

        mapping = {
            "xgboost": self.for_xgboost,
            "xgb": self.for_xgboost,
            "mlp": self.for_mlp,
            "cnn": self.for_cnn,
            "lstm": self.for_lstm,
        }

        try:
            return mapping[
                model.lower()
            ](**kwargs)

        except KeyError:
            raise ValueError(
                f"Unknown model {model!r}. "
                "Use xgboost, mlp, cnn or lstm."
            ) from None


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check(
    split_mode: str,
    balance: str,
) -> int:

    ad = Data4CyberAdapter(
        split_mode=split_mode,
    )

    print("\n" + "=" * 72)

    for name in (
        "xgboost",
        "mlp",
        "cnn",
        "lstm",
    ):
        bundle = ad.get(
            name,
            balance=balance,
        )

        print(bundle.describe())

        if bundle.meta.get("level") == "row":
            print(
                "  train blocks     : "
                f"{len(bundle.meta['train_block'])}"
            )
            print(
                "  validation blocks: "
                f"{len(bundle.meta['val_block'])}"
            )
            print(
                "  test blocks      : "
                f"{len(bundle.meta['test_block'])}"
            )

        print("-" * 72)

    print(
        "OK - all four Data4Cyber bundles built."
    )

    return 0


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check Data4Cyber model bundles."
        )
    )

    parser.add_argument(
        "--split-mode",
        choices=[
            "block",
            "scenario",
        ],
        default="block",
    )

    parser.add_argument(
        "--balance",
        choices=[
            "none",
            "oversample",
            "undersample",
            "class_weight",
        ],
        default="class_weight",
    )

    args = parser.parse_args()

    raise SystemExit(
        _self_check(
            args.split_mode,
            args.balance,
        )
    )