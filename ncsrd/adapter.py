"""
adapter.py — One data adapter, four model families.

Turns the output of `prep.py` (`ue_attack_labeled_scaled.csv`) into exactly the
tensor shapes that XGBoost, MLP, CNN and LSTM expect, using an identical split
for all four so the models are actually comparable.

Quick start
-----------
    from adapter import NetworkDataAdapter

    ad = NetworkDataAdapter("data/processed/ue_attack_labeled_scaled.csv")

    xgb  = ad.for_xgboost()   # (n, 49)
    mlp  = ad.for_mlp()       # (n, 49)
    cnn  = ad.for_cnn()       # (n, 49, 1)
    lstm = ad.for_lstm()      # (n, 10, 49)

Every call returns a `DataBundle` with `.X_train/.y_train/.X_test/.y_test`
(plus `.X_val/.y_val` if you asked for a validation split), `.input_shape`,
`.class_weight` and `.scale_pos_weight`.

The default split reproduces `04_model_training_and_evaluation.ipynb`:
random stratified 80/20, `random_state=42`, SMOTE applied to the training
split only. See `split()` for leakage-free temporal / per-UE alternatives.

Hard dependencies: numpy, pandas, scikit-learn.
Optional: imbalanced-learn (only for `balance="smote"`).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight

__all__ = ["NetworkDataAdapter", "DataBundle"]

DEFAULT_DATA = "data/processed/ue_attack_labeled_scaled.csv"
DEFAULT_INDEX = "sequence_index.csv"
DEFAULT_META = "prep_metadata.json"

Strategy = Literal["random", "temporal", "group"]
Balance = Literal["none", "smote", "oversample", "undersample", "class_weight"]
SeqMode = Literal["windows", "features"]
LabelMode = Literal["last", "any"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DataBundle:
    """Everything a model needs, in the right shape.

    Attributes
    ----------
    X_train, y_train, X_test, y_test
        Ready-to-fit arrays. `X_val`/`y_val` are ``None`` unless `val_size > 0`.
    input_shape
        Shape of a single sample, excluding the batch axis. Feed straight into
        ``keras.Input(shape=bundle.input_shape)``.
    class_weight
        ``{0: w0, 1: w1}`` balanced weights from the *training* split. Pass to
        ``model.fit(..., class_weight=bundle.class_weight)``.
    scale_pos_weight
        ``n_negative / n_positive`` in the training split — the XGBoost knob
        for imbalance. Leave it at 1.0 (its value after SMOTE) if you resampled.
    """

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
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def has_val(self) -> bool:
        return self.X_val is not None

    def describe(self) -> str:
        lines = [
            f"DataBundle(model={self.model!r})",
            f"  input_shape      : {self.input_shape}",
            f"  X_train          : {self.X_train.shape}  {self.X_train.dtype}",
            f"  y_train          : {np.bincount(self.y_train.astype(int), minlength=2).tolist()}"
            f"  (benign, attack)",
        ]
        if self.has_val:
            lines += [
                f"  X_val            : {self.X_val.shape}",
                f"  y_val            : {np.bincount(self.y_val.astype(int), minlength=2).tolist()}",
            ]
        lines += [
            f"  X_test           : {self.X_test.shape}  {self.X_test.dtype}",
            f"  y_test           : {np.bincount(self.y_test.astype(int), minlength=2).tolist()}",
            f"  scale_pos_weight : {self.scale_pos_weight:.4f}",
            f"  class_weight     : {{0: {self.class_weight.get(0, 1.0):.4f}, "
            f"1: {self.class_weight.get(1, 1.0):.4f}}}",
            f"  split            : {self.meta.get('strategy')} "
            f"test_size={self.meta.get('test_size')} seed={self.meta.get('random_state')}",
            f"  balance          : {self.meta.get('balance')}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.describe()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class NetworkDataAdapter:
    """Loads `prep.py` output and reshapes it per model family.

    Parameters
    ----------
    data_path
        Path to `ue_attack_labeled_scaled.csv`.
    index_path
        Path to `sequence_index.csv` (`_time` + `imeisv`, row-aligned with the
        dataset). Auto-detected next to `data_path`. Required only for
        ``strategy="temporal"``, ``strategy="group"`` and real LSTM windows.
    target
        Label column name. Default ``"attack_label"``.
    dtype
        Feature dtype for the emitted arrays. ``float32`` halves memory versus
        ``float64`` and is what Keras/PyTorch want anyway.
    """

    def __init__(self,
                 data_path: str | Path = DEFAULT_DATA,
                 index_path: str | Path | None = None,
                 target: str = "attack_label",
                 dtype: type = np.float32,
                 verbose: bool = True):
        self.data_path = Path(data_path)
        self.target = target
        self.dtype = dtype
        self.verbose = verbose

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"{self.data_path} not found. Run prep.py first:\n"
                f"    python prep.py -i amari_ue_data_classic_tabular.csv -o data/processed"
            )

        self.df = pd.read_csv(self.data_path)
        if self.target not in self.df.columns:
            raise KeyError(
                f"target {self.target!r} not in dataset; columns = {list(self.df.columns)}"
            )

        self.feature_names: list[str] = [c for c in self.df.columns if c != self.target]
        self._X = self.df[self.feature_names].to_numpy(dtype=self.dtype)
        self._y = self.df[self.target].to_numpy(dtype=np.int8)

        n_nan = int(np.isnan(self._X).sum())
        if n_nan:
            raise ValueError(
                f"{n_nan} NaN(s) in features — prep.py should have removed all of them."
            )

        # Optional sidecars ------------------------------------------------
        self._time: np.ndarray | None = None      # int64 nanoseconds
        self._ue: np.ndarray | None = None        # integer UE codes
        self._ue_labels: np.ndarray | None = None
        self._load_index(index_path)

        self.prep_meta: dict[str, Any] = {}
        meta_file = self.data_path.parent / DEFAULT_META
        if meta_file.exists():
            self.prep_meta = json.loads(meta_file.read_text(encoding="utf-8"))

        self._split: dict[str, Any] | None = None

        self._log(f"[adapter] {self.data_path.name}: {self._X.shape[0]:,} rows x "
                  f"{len(self.feature_names)} features")
        self._log(f"[adapter] class balance: benign={int((self._y == 0).sum()):,} "
                  f"attack={int((self._y == 1).sum()):,} "
                  f"({self._y.mean() * 100:.2f}% attack)")
        if self._time is not None:
            self._log(f"[adapter] sequence index: available ({self.n_ue} UEs) — "
                      "temporal/group splits and LSTM windows enabled")
        else:
            self._log("[adapter] sequence index: NOT FOUND — temporal/group splits "
                      "and LSTM windows unavailable")

    # -- setup ------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _load_index(self, index_path: str | Path | None) -> None:
        path = Path(index_path) if index_path else self.data_path.parent / DEFAULT_INDEX
        if not path.exists():
            return

        idx = pd.read_csv(path)
        if len(idx) != len(self.df):
            warnings.warn(
                f"{path.name} has {len(idx):,} rows but the dataset has {len(self.df):,}. "
                "Ignoring it — temporal/group splits and LSTM windows are disabled. "
                "Regenerate both files with the same prep.py run.",
                stacklevel=3,
            )
            return

        self._time = pd.to_datetime(idx["_time"], utc=True).astype("int64").to_numpy()
        codes, labels = pd.factorize(idx["imeisv"], sort=True)
        self._ue = codes.astype(np.int64)
        self._ue_labels = np.asarray(labels)

    def _require_index(self, what: str) -> None:
        if self._time is None:
            raise RuntimeError(
                f"{what} needs sequence_index.csv (`_time` + `imeisv`), which was not "
                f"found next to {self.data_path.name}. Re-run prep.py without --no-index."
            )

    # -- properties -------------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        """Full feature matrix, shape (n_rows, n_features)."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Full label vector, shape (n_rows,)."""
        return self._y

    @property
    def n_ue(self) -> int:
        """Number of distinct UEs, or 0 if the sequence index is missing."""
        return 0 if self._ue_labels is None else len(self._ue_labels)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self,
              strategy: Strategy = "random",
              test_size: float = 0.2,
              val_size: float = 0.0,
              random_state: int = 42,
              stratify: bool = True) -> "NetworkDataAdapter":
        """Choose the train/val/test partition. Returns self, so it chains.

        Strategies
        ----------
        ``"random"`` (default)
            Stratified random split — reproduces notebook 04 exactly
            (``test_size=0.2``, ``random_state=42``, ``stratify=y``).
            Note: rows are 5-second samples of the same UEs, so neighbouring
            rows are near-duplicates. A random split puts near-duplicates on
            both sides and inflates scores. Fine for reproducing the notebook;
            not a clean estimate of generalization.
        ``"temporal"``
            Oldest ``1 - test_size`` of rows train, newest ``test_size`` test.
            No future information leaks backwards. Class balance is *not*
            preserved — attack windows are unevenly spread across the capture.
        ``"group"``
            ``GroupShuffleSplit`` on ``imeisv``: no UE appears in both splits.
            Answers "does this generalize to a UE we have never seen?".

        ``val_size`` is a fraction of the *whole* dataset, carved out of train.
        """
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if not 0 <= val_size < 1 - test_size:
            raise ValueError(f"val_size must be in [0, {1 - test_size}), got {val_size}")

        n = len(self._y)
        all_idx = np.arange(n)

        if strategy == "random":
            train_idx, test_idx = train_test_split(
                all_idx, test_size=test_size, random_state=random_state,
                stratify=self._y if stratify else None,
            )
        elif strategy == "temporal":
            self._require_index("strategy='temporal'")
            order = np.argsort(self._time, kind="stable")
            cut = int(round(n * (1 - test_size)))
            train_idx, test_idx = order[:cut], order[cut:]
        elif strategy == "group":
            self._require_index("strategy='group'")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=random_state)
            train_idx, test_idx = next(gss.split(all_idx, self._y, groups=self._ue))
        else:
            raise ValueError(
                f"unknown strategy {strategy!r}; use 'random', 'temporal' or 'group'"
            )

        val_idx = None
        if val_size > 0:
            frac = val_size / (1 - test_size)          # fraction *of the train part*
            if strategy == "temporal":
                cut = int(round(len(train_idx) * (1 - frac)))
                train_idx, val_idx = train_idx[:cut], train_idx[cut:]
            elif strategy == "group":
                gss = GroupShuffleSplit(n_splits=1, test_size=frac,
                                        random_state=random_state)
                sub_tr, sub_va = next(
                    gss.split(train_idx, self._y[train_idx], groups=self._ue[train_idx])
                )
                train_idx, val_idx = train_idx[sub_tr], train_idx[sub_va]
            else:
                train_idx, val_idx = train_test_split(
                    train_idx, test_size=frac, random_state=random_state,
                    stratify=self._y[train_idx] if stratify else None,
                )

        self._split = {
            "train": np.asarray(train_idx),
            "val": None if val_idx is None else np.asarray(val_idx),
            "test": np.asarray(test_idx),
            "strategy": strategy,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "stratify": stratify,
        }

        self._log(f"[split] strategy={strategy} train={len(train_idx):,} "
                  f"val={0 if val_idx is None else len(val_idx):,} test={len(test_idx):,}")
        self._log(f"[split] attack rate  train={self._y[train_idx].mean() * 100:.2f}%  "
                  f"test={self._y[test_idx].mean() * 100:.2f}%")
        if strategy == "random":
            self._log("[split] note: random split over 5s samples of the same UEs — "
                      "near-duplicate rows land on both sides. Use strategy='temporal' "
                      "or 'group' for a conservative estimate.")
        return self

    def _ensure_split(self) -> dict[str, Any]:
        if self._split is None:
            self._log("[split] no split configured — applying the notebook-04 default "
                      "(random stratified 80/20, seed 42)")
            self.split()
        assert self._split is not None
        return self._split

    def save_split(self, path: str | Path) -> Path:
        """Persist the split indices so every teammate trains on identical rows."""
        sp = self._ensure_split()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            train=sp["train"],
            val=np.array([]) if sp["val"] is None else sp["val"],
            test=sp["test"],
            config=np.array(json.dumps({
                k: sp[k] for k in
                ("strategy", "test_size", "val_size", "random_state", "stratify")
            })),
        )
        self._log(f"[split] saved -> {path}")
        return path

    def load_split(self, path: str | Path) -> "NetworkDataAdapter":
        """Restore indices written by `save_split`."""
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["config"]))
        val = z["val"]
        self._split = {
            "train": z["train"],
            "val": None if val.size == 0 else val.astype(int),
            "test": z["test"],
            **cfg,
        }
        self._log(f"[split] loaded <- {path} "
                  f"(train={len(z['train']):,} test={len(z['test']):,})")
        return self

    # ------------------------------------------------------------------
    # Imbalance handling
    # ------------------------------------------------------------------

    def _class_weight(self, y: np.ndarray) -> dict[int, float]:
        classes = np.unique(y)
        if len(classes) < 2:
            return {0: 1.0, 1: 1.0}
        w = compute_class_weight("balanced", classes=classes, y=y)
        return {int(c): float(v) for c, v in zip(classes, w)}

    @staticmethod
    def _scale_pos_weight(y: np.ndarray) -> float:
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        return float(neg / pos) if pos else 1.0

    def _balance(self, X: np.ndarray, y: np.ndarray, how: Balance,
                 random_state: int) -> tuple[np.ndarray, np.ndarray]:
        """Resample a 2-D training split. `none`/`class_weight` are pass-through."""
        if how in ("none", "class_weight"):
            return X, y

        if how == "smote":
            try:
                from imblearn.over_sampling import SMOTE
            except ImportError as exc:      # pragma: no cover - env dependent
                raise ImportError(
                    "balance='smote' needs imbalanced-learn:  pip install imbalanced-learn\n"
                    "Alternatives that need no extra package: balance='oversample', "
                    "balance='undersample', or balance='class_weight' "
                    "(then pass bundle.class_weight / bundle.scale_pos_weight to the model)."
                ) from exc
            Xr, yr = SMOTE(random_state=random_state).fit_resample(X, y)
            self._log(f"[balance] SMOTE -> {np.bincount(yr.astype(int)).tolist()}")
            return np.asarray(Xr, dtype=self.dtype), np.asarray(yr, dtype=np.int8)

        rng = np.random.default_rng(random_state)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        minority, majority = (pos, neg) if len(pos) <= len(neg) else (neg, pos)

        if how == "oversample":
            extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
            keep = np.concatenate([majority, minority, extra])
        elif how == "undersample":
            keep = np.concatenate(
                [minority, rng.choice(majority, size=len(minority), replace=False)]
            )
        else:
            raise ValueError(
                f"unknown balance {how!r}; use 'none', 'smote', 'oversample', "
                "'undersample' or 'class_weight'"
            )

        rng.shuffle(keep)
        self._log(f"[balance] {how} -> {np.bincount(y[keep].astype(int)).tolist()}")
        return X[keep], y[keep]

    # ------------------------------------------------------------------
    # Tabular bundles: XGBoost, MLP, CNN
    # ------------------------------------------------------------------

    def _tabular_bundle(self, model: str, balance: Balance,
                        random_state: int, reshape: str) -> DataBundle:
        sp = self._ensure_split()

        Xtr, ytr = self._X[sp["train"]], self._y[sp["train"]]
        Xte, yte = self._X[sp["test"]], self._y[sp["test"]]
        Xva = yva = None
        if sp["val"] is not None:
            Xva, yva = self._X[sp["val"]], self._y[sp["val"]]

        # Resample the training split only — never val/test.
        Xtr, ytr = self._balance(Xtr, ytr, balance, random_state)

        n_feat = len(self.feature_names)
        if reshape == "2d":
            input_shape: tuple[int, ...] = (n_feat,)
        elif reshape == "3d":
            input_shape = (n_feat, 1)
            Xtr = Xtr.reshape(-1, n_feat, 1)
            Xte = Xte.reshape(-1, n_feat, 1)
            if Xva is not None:
                Xva = Xva.reshape(-1, n_feat, 1)
        else:                                          # pragma: no cover
            raise ValueError(reshape)

        return DataBundle(
            X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte, y_test=yte,
            model=model,
            feature_names=list(self.feature_names),
            input_shape=input_shape,
            class_weight=self._class_weight(ytr),
            scale_pos_weight=self._scale_pos_weight(ytr),
            meta={**{k: sp[k] for k in
                     ("strategy", "test_size", "val_size", "random_state", "stratify")},
                  "balance": balance, "layout": reshape},
        )

    def for_xgboost(self, balance: Balance = "smote",
                    random_state: int = 42,
                    as_frame: bool = False) -> DataBundle:
        """2-D bundle for `xgboost.XGBClassifier`. Shape ``(n, n_features)``.

        Defaults to SMOTE to match notebook 04. If imbalanced-learn is not
        installed, use ``balance="class_weight"`` and pass
        ``scale_pos_weight=bundle.scale_pos_weight`` to the classifier instead —
        cheaper and usually just as effective.

        ``as_frame=True`` returns pandas DataFrames, which keeps feature names
        attached for SHAP plots and `plot_importance`.
        """
        b = self._tabular_bundle("xgboost", balance, random_state, "2d")
        if as_frame:
            b.X_train = pd.DataFrame(b.X_train, columns=b.feature_names)
            b.X_test = pd.DataFrame(b.X_test, columns=b.feature_names)
            if b.X_val is not None:
                b.X_val = pd.DataFrame(b.X_val, columns=b.feature_names)
        return b

    def for_mlp(self, balance: Balance = "smote",
                random_state: int = 42) -> DataBundle:
        """2-D bundle for a dense network. Shape ``(n, n_features)``.

        ``bundle.input_shape`` is ``(n_features,)`` — drop it straight into
        ``keras.Input(shape=bundle.input_shape)`` or use it as the
        ``in_features`` of the first ``nn.Linear``.
        """
        return self._tabular_bundle("mlp", balance, random_state, "2d")

    def for_cnn(self, balance: Balance = "smote",
                random_state: int = 42) -> DataBundle:
        """3-D bundle for a 1-D CNN. Shape ``(n, n_features, 1)``, channels-last.

        The convolution slides over the *feature* axis, not time — each row is
        one 5-second snapshot, so this is a 1-D CNN over the feature vector
        (the standard tabular-CNN setup, and what Keras `Conv1D` expects).

        PyTorch's `nn.Conv1d` wants channels-first: ``X.transpose(0, 2, 1)``
        gives ``(n, 1, n_features)``.

        For a CNN over *time* instead, use
        ``for_lstm(sequence_mode="windows")`` — the ``(n, timesteps, features)``
        output feeds a temporal Conv1D unchanged.
        """
        return self._tabular_bundle("cnn", balance, random_state, "3d")

    # ------------------------------------------------------------------
    # Sequence bundles: LSTM
    # ------------------------------------------------------------------

    def _build_windows(self, timesteps: int, stride: int, label_mode: LabelMode,
                       max_gib: float) -> dict[str, np.ndarray]:
        """Sliding windows of consecutive samples, built *within* each UE.

        Rows are sorted by (UE, time) and windows never cross a UE boundary, so
        each sequence is a genuine slice of one device's history.
        """
        self._require_index("LSTM sequence_mode='windows'")
        assert self._time is not None and self._ue is not None

        order = np.lexsort((self._time, self._ue))     # primary UE, secondary time
        ue_sorted = self._ue[order]

        # Contiguous per-UE blocks in the sorted ordering.
        bounds = np.flatnonzero(
            np.r_[True, ue_sorted[1:] != ue_sorted[:-1], True]
        )

        starts = [
            np.arange(s, e - timesteps + 1, stride)
            for s, e in zip(bounds[:-1], bounds[1:])
            if (e - s) >= timesteps
        ]
        if not starts:
            raise ValueError(
                f"no UE has {timesteps} consecutive samples — lower `timesteps`."
            )
        starts_arr = np.concatenate(starts)

        n_win = len(starts_arr)
        n_feat = len(self.feature_names)
        need = n_win * timesteps * n_feat * np.dtype(self.dtype).itemsize
        if need > max_gib * (1024 ** 3):
            raise MemoryError(
                f"{n_win:,} windows x {timesteps} x {n_feat} ({self.dtype.__name__}) "
                f"= {need / 1024**3:.1f} GiB > max_gib={max_gib}. "
                f"Raise `stride` (stride={timesteps} gives non-overlapping windows, "
                f"~{need / timesteps / 1024**3:.2f} GiB), shorten `timesteps`, "
                f"or raise `max_gib` if you have the RAM."
            )

        # (n_windows, timesteps) positions into the original row order
        win_idx = order[starts_arr[:, None] + np.arange(timesteps)[None, :]]

        y_win = (self._y[win_idx].max(axis=1) if label_mode == "any"
                 else self._y[win_idx[:, -1]])

        self._log(f"[seq] {n_win:,} windows  timesteps={timesteps} stride={stride} "
                  f"label_mode={label_mode}  ({need / 1024**3:.2f} GiB)")
        self._log(f"[seq] window attack rate: {y_win.mean() * 100:.2f}%")

        return {
            "win_idx": win_idx,
            "y": y_win.astype(np.int8),
            "ue": ue_sorted[starts_arr],
            "end_time": self._time[win_idx[:, -1]],
        }

    def _split_windows(self, win: dict[str, np.ndarray],
                       sp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Re-apply the configured split at the window level.

        Splitting windows (not rows) is what keeps a sequence from straddling
        the train/test boundary.
        """
        strategy = sp["strategy"]
        test_size, val_size = sp["test_size"], sp["val_size"]
        rs = sp["random_state"]
        n = len(win["y"])
        all_idx = np.arange(n)

        if strategy == "random":
            tr, te = train_test_split(
                all_idx, test_size=test_size, random_state=rs,
                stratify=win["y"] if sp["stratify"] else None,
            )
        elif strategy == "temporal":
            order = np.argsort(win["end_time"], kind="stable")
            cut = int(round(n * (1 - test_size)))
            tr, te = order[:cut], order[cut:]
        else:  # group
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
            tr, te = next(gss.split(all_idx, win["y"], groups=win["ue"]))

        va = None
        if val_size > 0:
            frac = val_size / (1 - test_size)
            if strategy == "temporal":
                cut = int(round(len(tr) * (1 - frac)))
                tr, va = tr[:cut], tr[cut:]
            elif strategy == "group":
                gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=rs)
                a, b = next(gss.split(tr, win["y"][tr], groups=win["ue"][tr]))
                tr, va = tr[a], tr[b]
            else:
                tr, va = train_test_split(
                    tr, test_size=frac, random_state=rs,
                    stratify=win["y"][tr] if sp["stratify"] else None,
                )

        self._log(f"[seq] window split train={len(tr):,} "
                  f"val={0 if va is None else len(va):,} test={len(te):,}")
        return np.asarray(tr), None if va is None else np.asarray(va), np.asarray(te)

    def for_lstm(self,
                 timesteps: int = 10,
                 stride: int = 1,
                 sequence_mode: SeqMode = "windows",
                 label_mode: LabelMode = "last",
                 balance: Balance = "class_weight",
                 random_state: int = 42,
                 max_gib: float = 4.0) -> DataBundle:
        """3-D bundle for an LSTM/GRU. Shape ``(n, timesteps, n_features)``.

        Parameters
        ----------
        timesteps
            Window length in samples. The capture is ~5 s per sample, so
            ``timesteps=10`` is roughly a 50-second view of one UE.
        stride
            Step between window starts. ``1`` = maximum overlap and the most
            windows; ``stride=timesteps`` = disjoint windows and ~`timesteps`x
            less memory.
        sequence_mode
            ``"windows"`` (default) — real time series: `timesteps` consecutive
            samples from a single UE, ordered by `_time`. Needs
            `sequence_index.csv`.
            ``"features"`` — no time axis; each row is reshaped to
            ``(n_features, 1)``. Cheap, works without the index file, but the
            LSTM then reads the feature vector as if it were a sequence, which
            is not meaningful. Use it only as a smoke test.
        label_mode
            ``"last"`` labels a window by its final sample (predict the current
            state). ``"any"`` marks a window as attack if *any* sample in it is
            — higher recall, blurrier boundaries.
        balance
            Defaults to ``"class_weight"``: SMOTE interpolates between samples
            and has no meaning across a time axis. Pass
            ``bundle.class_weight`` to ``model.fit``. ``"oversample"`` /
            ``"undersample"`` operate on whole windows and are safe;
            ``"smote"`` is accepted but flattens time first — not recommended.
        """
        if sequence_mode == "features":
            b = self._tabular_bundle("lstm", balance, random_state, "3d")
            b.meta["sequence_mode"] = "features"
            self._log("[seq] sequence_mode='features': no time axis — "
                      "shape is (n, n_features, 1). Smoke tests only.")
            return b

        if sequence_mode != "windows":
            raise ValueError(
                f"unknown sequence_mode {sequence_mode!r}; use 'windows' or 'features'"
            )

        sp = self._ensure_split()
        win = self._build_windows(timesteps, stride, label_mode, max_gib)
        tr, va, te = self._split_windows(win, sp)

        n_feat = len(self.feature_names)

        def materialize(sel: np.ndarray) -> np.ndarray:
            return self._X[win["win_idx"][sel]]        # (k, timesteps, n_features)

        Xtr, ytr = materialize(tr), win["y"][tr]
        Xte, yte = materialize(te), win["y"][te]
        Xva = yva = None
        if va is not None:
            Xva, yva = materialize(va), win["y"][va]

        if balance not in ("none", "class_weight"):
            if balance == "smote":
                warnings.warn(
                    "balance='smote' on sequences flattens the time axis and "
                    "interpolates across timesteps, which is not physically "
                    "meaningful. Prefer balance='class_weight' or 'oversample'.",
                    stacklevel=2,
                )
                flat, ytr = self._balance(
                    Xtr.reshape(len(Xtr), -1), ytr, "smote", random_state
                )
                Xtr = flat.reshape(-1, timesteps, n_feat)
            else:
                # Resample whole windows: index the window axis, keep time intact.
                dummy = np.arange(len(ytr), dtype=self.dtype).reshape(-1, 1)
                kept, ytr = self._balance(dummy, ytr, balance, random_state)
                Xtr = Xtr[kept[:, 0].astype(int)]

        return DataBundle(
            X_train=np.ascontiguousarray(Xtr, dtype=self.dtype), y_train=ytr,
            X_val=None if Xva is None else np.ascontiguousarray(Xva, dtype=self.dtype),
            y_val=yva,
            X_test=np.ascontiguousarray(Xte, dtype=self.dtype), y_test=yte,
            model="lstm",
            feature_names=list(self.feature_names),
            input_shape=(timesteps, n_feat),
            class_weight=self._class_weight(ytr),
            scale_pos_weight=self._scale_pos_weight(ytr),
            meta={**{k: sp[k] for k in
                     ("strategy", "test_size", "val_size", "random_state", "stratify")},
                  "balance": balance, "layout": "3d",
                  "sequence_mode": "windows", "timesteps": timesteps,
                  "stride": stride, "label_mode": label_mode},
        )

    # ------------------------------------------------------------------

    def get(self, model: Literal["xgboost", "mlp", "cnn", "lstm"], **kwargs) -> DataBundle:
        """Dispatch by name — handy for looping over all four models."""
        try:
            fn = {"xgboost": self.for_xgboost, "xgb": self.for_xgboost,
                  "mlp": self.for_mlp, "cnn": self.for_cnn,
                  "lstm": self.for_lstm}[model.lower()]
        except KeyError:
            raise ValueError(
                f"unknown model {model!r}; use 'xgboost', 'mlp', 'cnn' or 'lstm'"
            ) from None
        return fn(**kwargs)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check(data_path: str, index_path: str | None, timesteps: int,
                stride: int, balance: str) -> int:
    ad = NetworkDataAdapter(data_path, index_path)
    ad.split(strategy="random", test_size=0.2, val_size=0.1, random_state=42)

    print("\n" + "=" * 72)
    for name in ("xgboost", "mlp", "cnn"):
        print(ad.get(name, balance=balance).describe())
        print("-" * 72)

    kw: dict[str, Any] = {"timesteps": timesteps, "stride": stride,
                          "balance": "class_weight"}
    if ad._time is None:
        kw["sequence_mode"] = "features"
    print(ad.for_lstm(**kw).describe())
    print("=" * 72)
    print("\nOK — all four bundles built.")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Build and print all four model bundles as a sanity check."
    )
    p.add_argument("--check", action="store_true",
                   help="run the sanity check (the default action)")
    p.add_argument("-d", "--data", default=DEFAULT_DATA, help="prep.py output CSV")
    p.add_argument("--index", default=None, help="sequence_index.csv (auto-detected)")
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--balance", default="oversample",
                   choices=["none", "smote", "oversample", "undersample", "class_weight"],
                   help="training-split resampling for the tabular bundles")
    a = p.parse_args()
    raise SystemExit(_self_check(a.data, a.index, a.timesteps, a.stride, a.balance))
