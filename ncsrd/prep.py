"""
prep.py — Attack labeling and preprocessing for the Amarisoft UE-level 5G dataset.

This is a script-form replication of `03_attack_labeling_and_preprocessing.ipynb`.
It takes the raw `amari_ue_data_classic_tabular.csv` and produces a cleaned,
labeled, scaled, ML-ready dataset.

Pipeline (in the exact order used by the notebook):
    1. Load raw CSV, parse `_time`, drop rows with unparseable timestamps.
    2. Label DDoS attack traffic using five known attack time windows (UTC).
    3. Drop non-ML columns (identifiers, IP/addressing, static config).
    4. Consolidate per-cell retransmission counters into UE-level max features.
    5. Median-impute missing numeric values.
    6. Drop the categorical `*_apn` columns.
    7. Standardize all features with StandardScaler (z-score).
    8. Drop zero-variance (constant) features.
    9. Persist dataset, fitted scaler, a row-aligned sequence index, and metadata.

Expected output on the reference dataset:
    shape (424221, 50)   ->  49 features + `attack_label`
    attack_label: 0 -> 397597, 1 -> 26624  (6.28% attack)

Usage
-----
    python prep.py --input path/to/amari_ue_data_classic_tabular.csv
    python prep.py -i raw.csv -o data/processed --tol 1e-12

Outputs written to --outdir (default: data/processed):
    ue_attack_labeled_scaled.csv   final ML dataset (features + attack_label)
    standard_scaler.pkl            fitted StandardScaler (62 pre-pruning columns)
    sequence_index.csv             row-aligned `_time` + `imeisv` (for LSTM / temporal splits)
    prep_metadata.json             feature lists, dropped columns, run summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration — mirrors the notebook exactly
# ---------------------------------------------------------------------------

TARGET = "attack_label"
TIME_COL = "_time"
UE_ID_COL = "imeisv"

#: DDoS attack windows from the dataset documentation / reference IEEE paper.
#: Naive strings here; localized to UTC at labeling time.
ATTACK_WINDOWS = [
    ("2024-08-18 07:00:00", "2024-08-18 08:00:00", "SYN Flood"),
    ("2024-08-19 07:00:00", "2024-08-19 09:41:00", "ICMP Flood"),
    ("2024-08-19 17:00:00", "2024-08-19 18:00:00", "UDP Fragmentation"),
    ("2024-08-21 12:00:00", "2024-08-21 13:00:00", "DNS Flood"),
    ("2024-08-21 17:00:00", "2024-08-21 18:00:00", "GTP-U Flood"),
]

#: Identifiers, addressing and static configuration. These describe *who* or
#: *where* rather than *how the UE behaved*, so they are removed to avoid
#: leakage and overfitting to specific UEs.
NON_ML_COLS = [
    TIME_COL,        # timestamp — used only for labeling
    "imeisv",        # UE identifier
    "5g_tmsi",
    "amf_ue_id",
    "rnti",
    "ran_id",
    "ran_plmn",
    "tac",
    "tac_plmn",
    "registered",
    # IP / addressing fields
    "bearer_0_ip",
    "bearer_0_ipv6",
    "bearer_1_ip",
    "bearer_1_ipv6",
]

OUT_DATASET = "ue_attack_labeled_scaled.csv"
OUT_SCALER = "standard_scaler.pkl"
OUT_INDEX = "sequence_index.csv"
OUT_META = "prep_metadata.json"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, parse timestamps, and drop unparseable rows."""
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL])

    print(f"[load]  raw rows={n_raw:,}  usable rows={len(df):,} "
          f"(dropped {n_raw - len(df):,} with invalid {TIME_COL})")
    print(f"[load]  time range: {df[TIME_COL].min()}  ->  {df[TIME_COL].max()}")
    return df


def add_attack_labels(df: pd.DataFrame, windows=ATTACK_WINDOWS) -> pd.DataFrame:
    """Add a binary `attack_label` column from the known attack time windows."""
    df[TARGET] = 0

    for start, end, name in windows:
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC")
        mask = (df[TIME_COL] >= lo) & (df[TIME_COL] <= hi)
        df.loc[mask, TARGET] = 1
        print(f"[label] {name:<18} {lo} -> {hi}  matched {int(mask.sum()):,} rows")

    counts = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
    print(f"[label] benign={counts.get(0, 0):,} ({pct.get(0, 0.0):.2f}%)  "
          f"attack={counts.get(1, 0):,} ({pct.get(1, 0.0):.2f}%)")
    return df


def drop_non_ml_columns(df: pd.DataFrame, cols=NON_ML_COLS) -> pd.DataFrame:
    """Remove identifier / addressing / static-config columns that exist."""
    present = [c for c in cols if c in df.columns]
    out = df.drop(columns=present)
    print(f"[drop]  removed {len(present)} non-ML columns -> shape {out.shape}")
    return out


def consolidate_cell_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse per-cell retransmission counters into UE-level maxima.

    A UE is attached to one serving cell at a time, so the per-cell columns are
    mostly NaN. Taking the row-wise max keeps the serving cell's behaviour while
    removing redundant, sparsely populated columns.
    """
    ul_cols = [c for c in df.columns if c.startswith("cell_") and "ul_retx" in c]
    dl_cols = [c for c in df.columns if c.startswith("cell_") and "dl_retx" in c]

    df["ul_retx_max"] = df[ul_cols].max(axis=1)
    df["dl_retx_max"] = df[dl_cols].max(axis=1)
    df = df.drop(columns=ul_cols + dl_cols)

    print(f"[cells] ul_retx_max <- {ul_cols}")
    print(f"[cells] dl_retx_max <- {dl_cols}")
    print(f"[cells] shape {df.shape}")
    return df, {"ul_retx_source": ul_cols, "dl_retx_source": dl_cols}


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns (robust to the heavy-tailed traffic metrics)."""
    before = int(df.isna().sum().sum())
    df = df.fillna(df.median(numeric_only=True))
    after = int(df.isna().sum().sum())
    print(f"[impute] NaNs {before:,} -> {after:,} (median imputation, numeric only)")
    return df


def drop_apn_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop `*_apn` columns: categorical config, not UE behaviour, often all-NaN."""
    apn_cols = [c for c in df.columns if c.endswith("_apn")]
    out = df.drop(columns=apn_cols)
    print(f"[apn]   dropped {apn_cols}  remaining NaNs={int(out.isna().sum().sum()):,}")
    return out, apn_cols


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """Z-score standardize every feature column; leave `attack_label` untouched."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns remain before scaling: {non_numeric}. "
            "Add them to NON_ML_COLS or handle them explicitly."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    df_scaled[TARGET] = y

    print(f"[scale] standardized {X.shape[1]} features -> shape {df_scaled.shape}")
    return df_scaled, scaler, X.columns.tolist()


def drop_zero_variance(df: pd.DataFrame, tol: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    """Remove constant features — they carry no discriminative signal.

    `tol=0.0` reproduces the notebook exactly (`std == 0`). A small tol such as
    1e-12 is more robust to float noise if you re-run on a different slice.
    """
    stds = df.drop(columns=[TARGET]).std()
    zero_var = stds[stds <= tol].index.tolist()
    out = df.drop(columns=zero_var)
    print(f"[var]   dropped {len(zero_var)} zero-variance features -> shape {out.shape}")
    for c in zero_var:
        print(f"          - {c}")
    return out, zero_var


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(input_path: str | Path,
        outdir: str | Path = "data/processed",
        tol: float = 0.0,
        write_index: bool = True) -> pd.DataFrame:
    """Execute the full pipeline and write all artifacts to `outdir`."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_raw(input_path)
    df = add_attack_labels(df)

    # Keep a row-aligned copy of time + UE id BEFORE they are dropped. This is
    # not used for training; the adapter needs it for temporal/grouped splits
    # and for building real per-UE LSTM sequences.
    seq_index = df[[TIME_COL, UE_ID_COL]].copy() if write_index else None

    df_ml = drop_non_ml_columns(df)
    df_ml, cell_meta = consolidate_cell_metrics(df_ml)
    df_ml = impute_missing(df_ml)
    df_ml, apn_cols = drop_apn_columns(df_ml)
    df_scaled, scaler, scaler_features = scale_features(df_ml)
    df_final, zero_var_cols = drop_zero_variance(df_scaled, tol=tol)

    final_features = [c for c in df_final.columns if c != TARGET]

    # --- persist -----------------------------------------------------------
    data_path = outdir / OUT_DATASET
    df_final.to_csv(data_path, index=False)
    print(f"[save]  dataset -> {data_path}")

    scaler_path = outdir / OUT_SCALER
    joblib.dump(scaler, scaler_path)
    print(f"[save]  scaler  -> {scaler_path}")

    if seq_index is not None:
        seq_index = seq_index.reset_index(drop=True)
        seq_index.index.name = "row_id"
        index_path = outdir / OUT_INDEX
        seq_index.to_csv(index_path)
        print(f"[save]  index   -> {index_path}")

    counts = df_final[TARGET].value_counts().sort_index()
    meta = {
        "source_csv": str(Path(input_path).resolve()),
        "n_rows": int(len(df_final)),
        "n_features": len(final_features),
        "target": TARGET,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "attack_rate": float(df_final[TARGET].mean()),
        "attack_windows": [
            {"name": n, "start": s, "end": e, "tz": "UTC"}
            for s, e, n in ATTACK_WINDOWS
        ],
        "dropped_non_ml_columns": [c for c in NON_ML_COLS if c in df.columns],
        "dropped_apn_columns": apn_cols,
        "dropped_zero_variance_columns": zero_var_cols,
        "zero_variance_tol": tol,
        "engineered_features": cell_meta,
        # The scaler was fit BEFORE zero-variance pruning, so it expects these
        # 62 columns in this order. Prune to `final_features` afterwards.
        "scaler_feature_names": scaler_features,
        "final_feature_names": final_features,
        "artifacts": {
            "dataset": OUT_DATASET,
            "scaler": OUT_SCALER,
            "sequence_index": OUT_INDEX if write_index else None,
        },
    }
    meta_path = outdir / OUT_META
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save]  metadata-> {meta_path}")

    print(f"\n[done]  final shape {df_final.shape}  "
          f"benign={counts.get(0, 0):,}  attack={counts.get(1, 0):,}")
    return df_final


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Label and preprocess the Amarisoft UE-level 5G dataset."
    )
    p.add_argument("-i", "--input", required=True,
                   help="path to amari_ue_data_classic_tabular.csv")
    p.add_argument("-o", "--outdir", default="data/processed",
                   help="output directory (default: data/processed)")
    p.add_argument("--tol", type=float, default=0.0,
                   help="std threshold for zero-variance pruning (default: 0.0)")
    p.add_argument("--no-index", action="store_true",
                   help="skip writing sequence_index.csv (_time + imeisv)")
    args = p.parse_args(argv)

    if not Path(args.input).exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    run(args.input, args.outdir, tol=args.tol, write_index=not args.no_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
