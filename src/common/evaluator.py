"""Common Evaluator Module — Member 5.

Implements the project-wide D6 metric definitions and schema so no member
computes metrics manually or inconsistently.

D6 Metric Definitions (Attack Class = Label 1):
    Precision(1) = TP / (TP + FP)
    Recall(1)    = TP / (TP + FN)
    F1(1)        = 2 * P * R / (P + R)
    FPR          = FP / (FP + TN)
    Accuracy     = (TP + TN) / (TP + TN + FP + FN)
    Weighted F1  = scikit-learn f1_score(average="weighted")
    ROC-AUC      = scikit-learn roc_auc_score

D6 Results Row Schema (25 columns):
    dataset, method, config, feature_set, split_policy, seed,
    precision, recall, f1, fpr, accuracy, weighted_f1, roc_auc,
    tp, fp, fn, tn, threshold, threshold_selected_on,
    n_train, n_test, inference_latency_ms, model_size_kb, notes
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

D6_COLUMNS = [
    "dataset",
    "method",
    "config",
    "feature_set",
    "split_policy",
    "seed",
    "precision",
    "recall",
    "f1",
    "fpr",
    "accuracy",
    "weighted_f1",
    "roc_auc",
    "tp",
    "fp",
    "fn",
    "tn",
    "threshold",
    "threshold_selected_on",
    "n_train",
    "n_test",
    "inference_latency_ms",
    "model_size_kb",
    "notes",
]


class Evaluator:
    """Standardized metric evaluation and reporting for all models and datasets."""

    def __init__(self, pos_label: int = 1):
        self.pos_label = pos_label

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        p_prob: np.ndarray,
        threshold: float = 0.5,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute sample-level metrics according to the D6 specification.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels (0 = benign, 1 = attack).
        p_prob : np.ndarray
            Predicted attack probabilities, shape (n,).
        threshold : float
            Decision threshold for binary classification.
        meta : dict, optional
            Metadata fields to populate in the D6 schema (dataset, method, config, etc.).

        Returns
        -------
        dict
            Dictionary containing both D6 schema fields and formatted outputs.
        """
        y_true = np.asarray(y_true).ravel()
        p_prob = np.asarray(p_prob).ravel()

        y_pred = (p_prob >= threshold).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        # Metrics for Attack Class (1)
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        # Metrics for Benign Class (0)
        prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

        # FPR = FP / (FP + TN)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Secondary metrics
        acc = float(accuracy_score(y_true, y_pred))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Safe ROC-AUC computation
        try:
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, p_prob))
            else:
                auc = 0.5
        except Exception:
            auc = 0.5

        res = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "fpr": round(fpr, 6),
            "accuracy": round(acc, 6),
            "weighted_f1": round(weighted_f1, 6),
            "roc_auc": round(auc, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "threshold": round(float(threshold), 6),
            "precision_0": round(prec_0, 6),
            "recall_0": round(rec_0, 6),
            "f1_0": round(f1_0, 6),
        }

        # Populate metadata defaults for D6
        meta = meta or {}
        d6_row = {
            "dataset": str(meta.get("dataset", "unknown")),
            "method": str(meta.get("method", "unknown")),
            "config": str(meta.get("config", "default")),
            "feature_set": str(meta.get("feature_set", "unknown")),
            "split_policy": str(meta.get("split_policy", "project_standard")),
            "seed": int(meta.get("seed", 42)),
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "fpr": res["fpr"],
            "accuracy": res["accuracy"],
            "weighted_f1": res["weighted_f1"],
            "roc_auc": res["roc_auc"],
            "tp": res["tp"],
            "fp": res["fp"],
            "fn": res["fn"],
            "tn": res["tn"],
            "threshold": res["threshold"],
            "threshold_selected_on": str(meta.get("threshold_selected_on", "validation")),
            "n_train": int(meta.get("n_train", 0)),
            "n_test": int(meta.get("n_test", len(y_true))),
            "inference_latency_ms": float(meta.get("inference_latency_ms", 0.0)),
            "model_size_kb": float(meta.get("model_size_kb", 0.0)),
            "notes": str(meta.get("notes", "")),
        }

        res["d6_row"] = d6_row
        return res

    def evaluate_windows(
        self,
        y_true: np.ndarray,
        p_prob: np.ndarray,
        window_size: int = 500,
        block_ids: Optional[np.ndarray] = None,
        thresholds: Optional[Union[float, List[float], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compute window-level metrics (FPR mean/var/max, threshold std) inside blocks.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        p_prob : np.ndarray
            Predicted probabilities.
        window_size : int
            Size of each window.
        block_ids : np.ndarray, optional
            Block identifier per sample to keep windows inside contiguous blocks.
        thresholds : float or list of floats, optional
            Per-window threshold array or single static threshold.

        Returns
        -------
        dict
            Window-level metrics: n_windows, fpr_mean, fpr_var, fpr_max, threshold_std.
        """
        y_true = np.asarray(y_true).ravel()
        p_prob = np.asarray(p_prob).ravel()

        if block_ids is None:
            block_ids = np.zeros(len(y_true), dtype=int)

        unique_blocks = np.unique(block_ids)
        window_fprs = []
        window_threshs = []

        is_scalar_thresh = isinstance(thresholds, (int, float)) or thresholds is None
        default_t = float(thresholds) if is_scalar_thresh and thresholds is not None else 0.5

        w_idx = 0
        for blk in unique_blocks:
            mask = block_ids == blk
            sub_y = y_true[mask]
            sub_p = p_prob[mask]

            n_samples = len(sub_y)
            n_w = n_samples // window_size

            for i in range(n_w):
                start = i * window_size
                end = start + window_size
                w_y = sub_y[start:end]
                w_p = sub_p[start:end]

                t = default_t
                if not is_scalar_thresh and thresholds is not None:
                    if w_idx < len(thresholds):
                        t = float(thresholds[w_idx])
                    else:
                        t = float(thresholds[-1])

                w_pred = (w_p >= t).astype(int)
                cm = confusion_matrix(w_y, w_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                w_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                window_fprs.append(w_fpr)
                window_threshs.append(t)
                w_idx += 1

        if not window_fprs:
            return {
                "n_windows": 0,
                "fpr_mean": 0.0,
                "fpr_var": 0.0,
                "fpr_max": 0.0,
                "threshold_std": 0.0,
            }

        fprs = np.array(window_fprs)
        threshs = np.array(window_threshs)

        return {
            "n_windows": len(fprs),
            "fpr_mean": round(float(np.mean(fprs)), 6),
            "fpr_var": round(float(np.var(fprs)), 6),
            "fpr_max": round(float(np.max(fprs)), 6),
            "threshold_std": round(float(np.std(threshs)), 6),
        }

    @staticmethod
    def save_d6_row(d6_row: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Append one D6 metric row to a CSV file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        file_exists = filepath.is_file() and filepath.stat().st_size > 0

        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=D6_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({col: d6_row.get(col, "") for col in D6_COLUMNS})
