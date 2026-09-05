"""ncsrd_prep.py — cleaning, labeling and feature selection for NCSRD-DS-5GDDoS.

Turns the raw `amari_ue_data_classic_tabular.csv` into an ML-ready dataset.

Two feature sets are produced from the *same rows in the same order*, because
the project needs two distinct, non-interchangeable lineages (see
`docs/PROJECT_DECISIONS.md`, D1):

`base38`
    The base paper's feature set (Xylouris et al., IEEE TCE 2025, Table II).
    All 24 per-cell radio metrics are consolidated into unified UE-level
    columns -- "Each UE was associated with one of three cells, and the
    relevant cell-specific features were consolidated into unified columns
    based on the UE's associated cell" -- plus 14 UE/bearer-level columns.
    24 + 14 = 38. Constant columns are kept, because the paper kept them.

`saurabh49`
    What Saurabh's notebook 03 actually does, reproduced step for step: drop
    identifiers, consolidate *only* the retransmission counters, median-impute,
    drop `*_apn`, standardize, then drop 13 zero-variance columns. 49 features.

These are NOT subsets of one another and must not be mixed. `base38` keeps
`tac`, `cell_id`, `t3512`, the `*_sst` / `*_qos_flow_id` pairs and the
`ue_aggregate_max_bitrate_*` pair that `saurabh49` drops; `saurabh49` keeps
per-cell columns and `ran_ue_id` that `base38` does not have.

Pipeline
--------
    1. Load raw CSV, parse `_time`, drop rows with unparseable timestamps.
    2. Label DDoS traffic using five known attack time windows (UTC).
    3. Build the requested feature set (see above).
    4. Median-impute remaining numeric gaps.
    5. Standardize with StandardScaler.
    6. Persist dataset, scaler, a row-aligned sequence index, and metadata.

Step 5 fits on the whole file, which leaks test statistics into training
features. That is faithful to notebook 03 and is kept for the reference
reproduction. For project-standard runs, `ncsrd_adapter` re-standardizes on the
training rows only (`refit_scaler=True`) -- see its docstring for why that is
exactly equivalent to scaling the raw data on train only.

Usage
-----
    # with the raw CSV in data/ncsrd/raw/ , no arguments are needed:
    python src/common/data/ncsrd_prep.py                    # builds BOTH sets
    python src/common/data/ncsrd_prep.py --feature-set base38
    python src/common/data/ncsrd_prep.py -i raw.csv -o some/other/dir

Outputs (per feature set, in --outdir, default data/ncsrd/processed/):
    ncsrd_<set>.csv          features + `attack_label`
    ncsrd_<set>_scaler.pkl   fitted StandardScaler
    ncsrd_<set>_meta.json    feature lists, dropped columns, run summary
    sequence_index.csv       row-aligned `_time` + `imeisv`, shared by both sets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import NCSRD_PROCESSED, NCSRD_RAW  # noqa: E402

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

TARGET = "attack_label"
TIME_COL = "_time"
UE_ID_COL = "imeisv"

FEATURE_SETS = ("base38", "saurabh49")

#: DDoS attack windows from the dataset documentation / reference IEEE paper.
#: Naive strings here; localized to UTC at labeling time. Both endpoints are
#: inclusive, matching notebook 03.
ATTACK_WINDOWS = [
    ("2024-08-18 07:00:00", "2024-08-18 08:00:00", "SYN Flood"),
    ("2024-08-19 07:00:00", "2024-08-19 09:41:00", "ICMP Flood"),
    ("2024-08-19 17:00:00", "2024-08-19 18:00:00", "UDP Fragmentation"),
    ("2024-08-21 12:00:00", "2024-08-21 13:00:00", "DNS Flood"),
    ("2024-08-21 17:00:00", "2024-08-21 18:00:00", "GTP-U Flood"),
]

OUT_INDEX = "sequence_index.csv"


# ---------------------------------------------------------------------------
# base38 — the base paper's Table II feature set
# ---------------------------------------------------------------------------

#: Table II of Xylouris et al., verbatim and in paper order. Verified against
#: the raw schema: the first 24 are exactly the distinct per-cell metrics (none
#: left over), the last 14 are UE/bearer-level columns. 24 + 14 = 38.
BASE38_FEATURES = [
    # -- consolidated per-cell radio metrics (24) ---------------------------
    "ul_retx", "dl_retx", "ul_tx", "dl_tx", "ul_bitrate", "dl_bitrate",
    "ul_mcs", "dl_mcs", "ul_path_loss", "cell_id", "epre", "turbo_decoder_avg",
    "initial_ta", "ul_err", "ul_n_layer", "ul_phr", "ul_rank", "dl_err",
    "cqi", "p_ue", "pusch_snr", "ri", "turbo_decoder_max", "turbo_decoder_min",
    # -- UE / bearer level (14) --------------------------------------------
    "tac",
    "bearer_0_session_id", "bearer_1_session_id",
    "bearer_0_dl_total_bytes", "bearer_0_ul_total_bytes",
    "bearer_0_qos_flow_id", "bearer_0_sst",
    "bearer_1_dl_total_bytes", "bearer_1_ul_total_bytes",
    "bearer_1_qos_flow_id", "bearer_1_sst",
    "t3512",
    "ue_aggregate_max_bitrate_dl", "ue_aggregate_max_bitrate_ul",
]

#: Table II abbreviates two column names. Map paper name -> raw column name.
TABLE2_ALIASES = {
    "bearer_0_session_id": "bearer_0_pdu_session_id",
    "bearer_1_session_id": "bearer_1_pdu_session_id",
}

#: `ul_bitrate` and `dl_bitrate` exist BOTH as top-level UE columns and as
#: per-cell metrics, so Table II is ambiguous for exactly these two. We take the
#: consolidated per-cell versions, because that is the only reading under which
#: the 24 distinct cell metrics map onto Table II with none left over. Flip to
#: "ue" to use the top-level columns instead; the count stays 38 either way.
#: See docs/PROJECT_DECISIONS.md, D1, for the reasoning.
BASE38_BITRATE_SOURCE = "cell"          # "cell" | "ue"


def cell_metric_names(df: pd.DataFrame) -> list[str]:
    """Distinct metric suffixes across the `cell_<n>_<metric>` columns."""
    return sorted({
        c.split("_", 2)[2] for c in df.columns
        if c.startswith("cell_") and c.count("_") >= 2
    })


def consolidate_all_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse every `cell_<n>_<metric>` family into one UE-level `<metric>`.

    A UE is attached to one serving cell at a time, so the per-cell columns are
    mostly NaN. The row-wise max keeps the serving cell's value and drops the
    redundant, sparsely populated columns. This is the consolidation the base
    paper describes for its 38-feature table.
    """
    sources: dict[str, list[str]] = {}
    for metric in cell_metric_names(df):
        cols = [c for c in df.columns
                if c.startswith("cell_") and c.endswith("_" + metric)]
        if cols:
            df[metric] = df[cols].max(axis=1)
            sources[metric] = cols

    dropped = sorted({c for cols in sources.values() for c in cols})
    df = df.drop(columns=dropped)
    print(f"[cells] consolidated {len(sources)} metrics from {len(dropped)} "
          f"per-cell columns -> shape {df.shape}")
    return df, sources


def build_base38(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select the base paper's exact 38 features (labels already attached)."""
    df, sources = consolidate_all_cells(df)

    if BASE38_BITRATE_SOURCE == "ue":
        # Prefer the top-level UE bitrate columns over the consolidated ones.
        for m in ("ul_bitrate", "dl_bitrate"):
            if f"{m}_ue" in df.columns:                   # pragma: no cover
                df[m] = df[f"{m}_ue"]

    wanted = [TABLE2_ALIASES.get(f, f) for f in BASE38_FEATURES]
    missing = [p for p, r in zip(BASE38_FEATURES, wanted) if r not in df.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} of the 38 base-paper features are not in the data: "
            f"{missing}. The raw schema may differ from the one this was "
            f"verified against; check BASE38_FEATURES / TABLE2_ALIASES."
        )

    out = df[wanted + [TARGET]].copy()
    out.columns = BASE38_FEATURES + [TARGET]      # use the paper's names
    print(f"[base38] selected {len(BASE38_FEATURES)} features -> shape {out.shape}")
    return out, {"cell_sources": sources,
                 "bitrate_source": BASE38_BITRATE_SOURCE,
                 "aliases": TABLE2_ALIASES}


# ---------------------------------------------------------------------------
# saurabh49 — reproduction of notebook 03
# ---------------------------------------------------------------------------

#: Identifiers, addressing and static configuration dropped by notebook 03.
#: NOTE: `ran_ue_id` is NOT in this list, so it survives into the 49 features
#: even though it is a session identifier rather than a behaviour metric. That
#: is what the notebook does and we reproduce it faithfully; see
#: the known-limitations section of docs/PROJECT_DECISIONS.md.
NON_ML_COLS = [
    TIME_COL, "imeisv", "5g_tmsi", "amf_ue_id", "rnti", "ran_id", "ran_plmn",
    "tac", "tac_plmn", "registered",
    "bearer_0_ip", "bearer_0_ipv6", "bearer_1_ip", "bearer_1_ipv6",
]


def build_saurabh49(df: pd.DataFrame, tol: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Notebook-03 feature path: 49 features after zero-variance pruning.

    Zero-variance pruning happens *after* scaling here, exactly as the notebook
    does it. `tol=0.0` reproduces the notebook (`std == 0`).
    """
    present = [c for c in NON_ML_COLS if c in df.columns]
    df = df.drop(columns=present)
    print(f"[drop]  removed {len(present)} non-ML columns -> shape {df.shape}")

    # Only the retransmission counters are consolidated, unlike base38.
    ul_cols = [c for c in df.columns if c.startswith("cell_") and "ul_retx" in c]
    dl_cols = [c for c in df.columns if c.startswith("cell_") and "dl_retx" in c]
    df["ul_retx_max"] = df[ul_cols].max(axis=1)
    df["dl_retx_max"] = df[dl_cols].max(axis=1)
    df = df.drop(columns=ul_cols + dl_cols)
    print(f"[cells] ul_retx_max <- {ul_cols}")
    print(f"[cells] dl_retx_max <- {dl_cols}")

    df = impute_missing(df)

    apn_cols = [c for c in df.columns if c.endswith("_apn")]
    df = df.drop(columns=apn_cols)
    print(f"[apn]   dropped {apn_cols}  remaining NaNs={int(df.isna().sum().sum()):,}")

    return df, {"dropped_non_ml_columns": present,
                "dropped_apn_columns": apn_cols,
                "ul_retx_source": ul_cols, "dl_retx_source": dl_cols,
                "zero_variance_tol": tol}


# ---------------------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------------------


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, parse timestamps, and drop unparseable rows."""
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)
    print(f"[load]  raw rows={n_raw:,}  usable rows={len(df):,} "
          f"(dropped {n_raw - len(df):,} with invalid {TIME_COL})")
    print(f"[load]  time range: {df[TIME_COL].min()}  ->  {df[TIME_COL].max()}")
    return df


def add_attack_labels(df: pd.DataFrame, windows=ATTACK_WINDOWS) -> pd.DataFrame:
    """Add a binary `attack_label` column from the known attack time windows."""
    df[TARGET] = 0
    for start, end, name in windows:
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        mask = (df[TIME_COL] >= lo) & (df[TIME_COL] <= hi)
        df.loc[mask, TARGET] = 1
        print(f"[label] {name:<18} {lo} -> {hi}  matched {int(mask.sum()):,} rows")

    counts = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
    print(f"[label] benign={counts.get(0, 0):,} ({pct.get(0, 0.0):.2f}%)  "
          f"attack={counts.get(1, 0):,} ({pct.get(1, 0.0):.2f}%)")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns (robust to heavy-tailed traffic metrics)."""
    before = int(df.isna().sum().sum())
    df = df.fillna(df.median(numeric_only=True))
    after = int(df.isna().sum().sum())
    print(f"[impute] NaNs {before:,} -> {after:,} (median imputation, numeric only)")
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """Z-score standardize every feature column; leave `attack_label` untouched.

    Constant columns come out as all-zero (sklearn sets their scale to 1), not
    NaN, so `base38` can keep them exactly as the paper does.
    """
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns remain before scaling: {non_numeric}. "
            "The base paper label-encodes categoricals; in this dataset every "
            "selected feature is already numeric, so reaching here means the "
            "raw schema changed."
        )

    scaler = StandardScaler()
    out = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=df.index)
    out[TARGET] = y
    print(f"[scale] standardized {X.shape[1]} features -> shape {out.shape}")
    return out, scaler, X.columns.tolist()


def drop_zero_variance(df: pd.DataFrame, tol: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    """Remove constant features. Used by `saurabh49` only."""
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
        outdir: str | Path = NCSRD_PROCESSED,
        feature_set: str = "saurabh49",
        tol: float = 0.0,
        write_index: bool = True,
        df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one feature set and write its artifacts to `outdir`.

    `df_raw` lets the caller reuse an already loaded+labeled frame so that both
    feature sets are guaranteed to come from identical rows in identical order.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if df_raw is None:
        df_raw = add_attack_labels(load_raw(input_path))
    df = df_raw.copy()

    print(f"\n=== building feature set: {feature_set} ===")
    if write_index:
        # Row-aligned time + UE id, captured BEFORE they are dropped. Not
        # features; the adapter needs them for temporal/grouped splits, LSTM
        # windows, and every later window-based analysis.
        seq = df[[TIME_COL, UE_ID_COL]].copy().reset_index(drop=True)
        seq.index.name = "row_id"
        seq.to_csv(outdir / OUT_INDEX)
        print(f"[save]  index   -> {outdir / OUT_INDEX}")

    if feature_set == "base38":
        df_feat, extras = build_base38(df)
        df_feat = impute_missing(df_feat)
        df_final, scaler, scaler_features = scale_features(df_feat)
        zero_var = [c for c in df_final.columns
                    if c != TARGET and df_final[c].std() == 0]
        print(f"[var]   {len(zero_var)} constant features KEPT (the paper's 38 "
              f"include them): {zero_var}")
        extras["constant_features_kept"] = zero_var
    else:
        df_feat, extras = build_saurabh49(df, tol=tol)
        df_scaled, scaler, scaler_features = scale_features(df_feat)
        df_final, zero_var = drop_zero_variance(df_scaled, tol=tol)
        extras["dropped_zero_variance_columns"] = zero_var

    final_features = [c for c in df_final.columns if c != TARGET]

    stem = f"ncsrd_{feature_set}"
    data_path = outdir / f"{stem}.csv"
    df_final.to_csv(data_path, index=False)
    print(f"[save]  dataset -> {data_path}")

    joblib.dump(scaler, outdir / f"{stem}_scaler.pkl")
    print(f"[save]  scaler  -> {outdir / f'{stem}_scaler.pkl'}")

    counts = df_final[TARGET].value_counts().sort_index()
    meta = {
        "feature_set": feature_set,
        "source_csv": Path(input_path).name,
        "n_rows": int(len(df_final)),
        "n_features": len(final_features),
        "target": TARGET,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "attack_rate": float(df_final[TARGET].mean()),
        "attack_windows": [{"name": n, "start": s, "end": e, "tz": "UTC"}
                           for s, e, n in ATTACK_WINDOWS],
        # The scaler was fit on these columns in this order. For saurabh49 that
        # is BEFORE zero-variance pruning, so it expects more columns than the
        # dataset has; prune to `final_feature_names` after transforming.
        "scaler_feature_names": scaler_features,
        "final_feature_names": final_features,
        **extras,
    }
    (outdir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save]  metadata-> {outdir / f'{stem}_meta.json'}")
    print(f"[done]  {feature_set}: shape {df_final.shape}  "
          f"benign={counts.get(0, 0):,}  attack={counts.get(1, 0):,}")
    return df_final


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Label and preprocess the NCSRD-DS-5GDDoS UE-level dataset."
    )
    p.add_argument("-i", "--input",
                   default=NCSRD_RAW / "amari_ue_data_classic_tabular.csv",
                   help="path to amari_ue_data_classic_tabular.csv "
                        "(default: data/ncsrd/raw/)")
    p.add_argument("-o", "--outdir", default=NCSRD_PROCESSED,
                   help="output directory (default: data/ncsrd/processed/)")
    p.add_argument("--feature-set", default="both",
                   choices=[*FEATURE_SETS, "both"],
                   help="which feature set to build (default: both)")
    p.add_argument("--tol", type=float, default=0.0,
                   help="std threshold for zero-variance pruning, saurabh49 only")
    p.add_argument("--no-index", action="store_true",
                   help="skip writing sequence_index.csv (you almost never want this)")
    args = p.parse_args(argv)

    if not Path(args.input).exists():
        print(f"error: input not found: {args.input}\n"
              f"Download NCSRD-DS-5GDDoS from https://zenodo.org/records/13900057 "
              f"and place the CSV there.", file=sys.stderr)
        return 2

    # Load and label ONCE so every feature set has identical rows in identical
    # order -- that is what lets one frozen split index apply to both.
    df_raw = add_attack_labels(load_raw(args.input))

    sets = FEATURE_SETS if args.feature_set == "both" else (args.feature_set,)
    for fs in sets:
        run(args.input, args.outdir, feature_set=fs, tol=args.tol,
            write_index=not args.no_index, df_raw=df_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
