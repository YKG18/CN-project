"""data4cyber_prep.py — preprocessing for the Data4Cyber smart-grid dataset.

Data4Cyber is a cyber-physical (Modbus/MQTT power-grid) testbed, NOT a 5G
dataset. It is deliberately given its own adapter and its own feature schema
rather than being forced into NCSRD's -- only the *methodology* transfers, not
the features (see docs/PROJECT_DECISIONS.md, D7).

Two split modes, both written side by side:

`block` (primary, D3)
    Each scenario's timeline is cut into contiguous blocks of `--block-size`
    seconds; whole blocks are then assigned to train/validation/test,
    stratified by scenario and block label. Every attack family appears in
    every split, adjacent near-identical seconds never straddle a split
    boundary, and no window is ever built across a boundary. This is the
    independent within-dataset evaluation used for the 3x2 comparison.

`scenario` (secondary)
    Whole scenarios are held out, so the test set contains attack families
    never seen in training. That is a *novel-attack robustness* experiment,
    not the main comparison -- report it separately and label it as such.

Outputs land in `<output-dir>/<split-mode>/`, two views of the same split:

    <split>_rows.npz     X, y, scenario, block, timestamp  -- per-sample
    <split>_windows.npz  X, y                              -- per-window
    <split>_windows_metadata.json                          -- window provenance
    manifest.json                                          -- everything else

Row-level is the primary metric level (D4); windows exist for the metrics that
genuinely need a time axis (FPR variance, drift latency, threshold stability).

Usage
-----
    python src/common/data/data4cyber_prep.py                  # both modes
    python src/common/data/data4cyber_prep.py --split-mode block
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import DATA4CYBER_PROCESSED, DATA4CYBER_RAW, SEED  # noqa: E402

SPLIT_MODES = ("block", "scenario")

LABEL_COLUMNS = {"timestamp", "attack_active", "attack_phase", "attack_phase_all"}
OPTIONAL_ATTACK_COLUMNS = {"Attacker.event", "Attacker.old_value", "Attacker.new_value"}

#: Absolute wall-clock columns. They are monotone within a scenario and every
#: scenario occupies its own disjoint time range, so a model can split on them
#: to identify the scenario -- and therefore the label -- instead of learning
#: the attack. Verified: 8 scenarios, 8 non-overlapping ranges. Always dropped.
CLOCK_SUFFIXES = (".realtime", ".timestamp")

#: Secondary experiment only. S6 (MQTT supply-chain) is deliberately unseen in
#: training; S1_alt is a repeat run of S1 and joins training.
SCENARIO_SPLITS = {
    "train": ["S0_benign_baseline", "S1_industroyer_pv", "S1_industroyer_pv_alt",
              "S2_industroyer_bss", "S3_arp_spoof_bss_meter_half_values"],
    "validation": ["S4_arp_spoof_loads_pv_two_phase"],
    "test": ["S5_arp_spoof_loads_pv_bss_two_phase",
             "S6_mqtt_supply_chain_compromise"],
}

#: Fractions of blocks for the primary `block` mode.
BLOCK_FRACTIONS = {"train": 0.60, "validation": 0.15, "test": 0.25}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DATA4CYBER_RAW,
                   help="folder holding the S0..S6 scenario directories")
    p.add_argument("--output-dir", type=Path, default=DATA4CYBER_PROCESSED)
    p.add_argument("--split-mode", default="both", choices=[*SPLIT_MODES, "both"],
                   help="which split to build (default: both)")
    p.add_argument("--block-size", type=int, default=120,
                   help="seconds per contiguous block in `block` mode (default: 120)")
    p.add_argument("--window-size", type=int, default=60, help="seconds per window")
    p.add_argument("--stride", type=int, default=30, help="seconds between window starts")
    p.add_argument("--label", choices=["binary", "phase"], default="binary")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--include-attacker-features", action="store_true",
                   help="include explicit attacker fields (normally excluded: "
                        "they make detection trivial)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_all(root: Path) -> pd.DataFrame:
    """Load every scenario into one frame, ordered by (scenario, timestamp)."""
    folders = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and (d / "dataset.csv").exists())
    if not folders:
        raise FileNotFoundError(
            f"No scenario folders with a dataset.csv under {root}.\n"
            f"Download Data4Cyber from https://zenodo.org/records/19965384 "
            f"and unpack the S0_… – S6_… folders there."
        )
    frames, tags = [], []
    for folder in folders:
        f = pd.read_csv(root / folder / "dataset.csv")
        frames.append(f)
        tags.extend([folder] * len(f))
    # Join the scenario tag along axis=1 rather than inserting into a
    # ~160-column concat result, which pandas warns is fragmented.
    df = pd.concat([pd.concat(frames, ignore_index=True),
                    pd.Series(tags, name="scenario")], axis=1)
    df = df.sort_values(["scenario", "timestamp"], kind="stable").reset_index(drop=True)
    print(f"[load]  {len(folders)} scenarios, {len(df):,} rows, {df.shape[1]} columns")
    return df


def feature_columns(df: pd.DataFrame, include_attacker: bool) -> list[str]:
    """Numeric behaviour columns only: no labels, no attacker fields, no clocks."""
    excluded = LABEL_COLUMNS | {"scenario", "block"}
    if not include_attacker:
        excluded |= OPTIONAL_ATTACK_COLUMNS
    cand = [c for c in df.columns
            if c not in excluded and not c.endswith(CLOCK_SUFFIXES)]
    return [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]


def labels(df: pd.DataFrame, label_mode: str,
           phase_map: dict[str, int] | None = None):
    """Binary `attack_active`, or an integer-coded `attack_phase`."""
    if label_mode == "binary":
        y = df["attack_active"].astype(str).str.lower().eq("true")
        return y.astype(np.int8).to_numpy(), None
    phases = df["attack_phase"].fillna("normal").astype(str)
    if phase_map is None:
        phase_map = {n: i for i, n in enumerate(sorted(phases.unique()))}
    unknown = sorted(set(phases.unique()) - set(phase_map))
    if unknown:   # test-only phases are never silently folded into a train class
        phase_map = {**phase_map,
                     **{n: len(phase_map) + i for i, n in enumerate(unknown)}}
    return phases.map(phase_map).astype(np.int16).to_numpy(), phase_map


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def add_blocks(df: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """Label each row with a contiguous `<scenario>#<n>` block id."""
    pos = df.groupby("scenario").cumcount()
    df = df.copy()
    df["block"] = df["scenario"] + "#" + (pos // block_size).astype(str)
    print(f"[block] {df['block'].nunique()} blocks of <= {block_size}s "
          f"across {df['scenario'].nunique()} scenarios")
    return df


def assign_block_split(df: pd.DataFrame, y: np.ndarray, seed: int) -> pd.Series:
    """Assign whole blocks to train/validation/test.

    Blocks are stratified by (scenario, majority label) so every scenario and
    both classes are represented in all three splits, while keeping each block
    -- and therefore every window built inside it -- entirely on one side.
    """
    info = (pd.DataFrame({"block": df["block"], "scenario": df["scenario"], "y": y})
            .groupby("block")
            .agg(scenario=("scenario", "first"), rate=("y", "mean"))
            .reset_index())
    info["stratum"] = info["scenario"] + "|" + (info["rate"] >= 0.5).astype(int).astype(str)

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for stratum, grp in info.groupby("stratum"):
        blocks = grp["block"].to_numpy()
        rng.shuffle(blocks)
        n = len(blocks)
        if n >= 3:
            # At least one block each, and never starve val/test.
            n_tr = min(max(1, round(n * BLOCK_FRACTIONS["train"])), n - 2)
            n_va = max(1, round(n * BLOCK_FRACTIONS["validation"]))
            n_va = min(n_va, n - n_tr - 1)
        elif n == 2:
            n_tr, n_va = 1, 0            # one train, one test
        else:
            n_tr, n_va = 1, 0            # single block: train only
            print(f"[block] warning: stratum {stratum!r} has 1 block; it "
                  f"appears in train only")
        for b in blocks[:n_tr]:
            assignment[b] = "train"
        for b in blocks[n_tr:n_tr + n_va]:
            assignment[b] = "validation"
        for b in blocks[n_tr + n_va:]:
            assignment[b] = "test"
    return df["block"].map(assignment)


def assign_scenario_split(df: pd.DataFrame) -> pd.Series:
    """Assign whole scenarios, per `SCENARIO_SPLITS` (secondary experiment)."""
    lookup = {s: name for name, ss in SCENARIO_SPLITS.items() for s in ss}
    missing = sorted(set(df["scenario"]) - set(lookup))
    if missing:
        raise KeyError(f"scenarios not covered by SCENARIO_SPLITS: {missing}")
    return df["scenario"].map(lookup)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def make_windows(values: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 timestamps: np.ndarray, window_size: int, stride: int):
    """Sliding windows built strictly inside one group (scenario or block).

    Passing block ids as `groups` is what stops a window from spanning a
    train/test boundary.
    """
    windows, targets, meta = [], [], []
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        for start in range(0, len(idx) - window_size + 1, stride):
            rows = idx[start:start + window_size]
            wy = y[rows]
            windows.append(values[rows])
            # Binary: malicious if ANY second is. Phase: the final time step,
            # which is what an online detector would actually emit.
            targets.append(int(wy.max()) if y.dtype == np.int8 else int(wy[-1]))
            meta.append({"group": str(g), "start": str(timestamps[rows[0]]),
                         "end": str(timestamps[rows[-1]])})
    if not windows:
        return (np.empty((0, window_size, values.shape[1]), np.float32),
                np.empty(0, int), [])
    return np.asarray(windows, np.float32), np.asarray(targets), meta


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build(df: pd.DataFrame, assignment: pd.Series, columns: list[str],
          args: argparse.Namespace, mode: str) -> dict:
    """Fit on train only, then write both views of every split."""
    outdir = Path(args.output_dir) / mode
    outdir.mkdir(parents=True, exist_ok=True)

    y_all, label_map = labels(df, args.label)
    train_mask = (assignment == "train").to_numpy()

    raw = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    # Every statistic below comes from training rows only -- no test information
    # reaches imputation, feature filtering or scaling.
    medians = np.nan_to_num(np.nanmedian(raw[train_mask], axis=0), nan=0.0)
    raw = np.where(np.isnan(raw), medians, raw)
    keep = raw[train_mask].var(axis=0) > 1e-12
    kept_columns = np.asarray(columns)[keep].tolist()
    mean = raw[train_mask][:, keep].mean(axis=0)
    scale = raw[train_mask][:, keep].std(axis=0)
    scale[scale == 0] = 1.0
    X_all = ((raw[:, keep] - mean) / scale).astype(np.float32)

    summary = {
        "split_mode": mode,
        "source": Path(args.input_dir).name,
        "label": args.label,
        "seed": args.seed,
        "block_size_seconds": args.block_size if mode == "block" else None,
        "window_size_seconds": args.window_size,
        "stride_seconds": args.stride,
        "features_before_variance_filter": len(columns),
        "features_after_variance_filter": len(kept_columns),
        "attacker_features_included": args.include_attacker_features,
        "scenarios": sorted(df["scenario"].unique().tolist()),
    }
    if mode == "scenario":
        summary["scenario_splits"] = SCENARIO_SPLITS

    print(f"\n=== {mode} split ===")
    for name in ("train", "validation", "test"):
        sel = (assignment == name).to_numpy()
        sub = df.loc[sel]
        y = y_all[sel]
        X = X_all[sel]
        # Group windows by BLOCK in block mode so none crosses a boundary.
        groups = (sub["block"] if mode == "block" else sub["scenario"]).to_numpy()

        Xw, yw, wmeta = make_windows(X, y, groups, sub["timestamp"].to_numpy(),
                                     args.window_size, args.stride)
        np.savez_compressed(outdir / f"{name}_windows.npz", X=Xw, y=yw)
        (outdir / f"{name}_windows_metadata.json").write_text(
            json.dumps(wmeta, indent=2), encoding="utf-8")
        np.savez_compressed(
            outdir / f"{name}_rows.npz", X=X, y=y,
            scenario=sub["scenario"].to_numpy().astype(str),
            block=sub["block"].to_numpy().astype(str),
            timestamp=sub["timestamp"].to_numpy().astype(str),
        )
        summary[name] = {
            "rows": int(sel.sum()), "positive_rows": int(y.sum()),
            "positive_row_rate": round(float(y.mean()), 4) if len(y) else None,
            "windows": int(len(yw)), "positive_windows": int((yw > 0).sum()),
            "scenarios": sorted(sub["scenario"].unique().tolist()),
        }
        print(f"  {name:<11} rows={summary[name]['rows']:<6} "
              f"attack={summary[name]['positive_row_rate']}  "
              f"windows={summary[name]['windows']}")

    summary["features"] = kept_columns
    summary["label_mapping"] = label_map or {"benign": 0, "attack": 1}
    (outdir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  -> {outdir}")
    return summary


def main() -> None:
    args = parse_args()
    if min(args.window_size, args.stride, args.block_size) < 1:
        raise ValueError("window-size, stride and block-size must be positive")
    if args.block_size < args.window_size:
        raise ValueError(
            f"block-size ({args.block_size}) must be >= window-size "
            f"({args.window_size}), otherwise no window fits inside a block."
        )

    df = load_all(Path(args.input_dir))
    df = add_blocks(df, args.block_size)
    columns = feature_columns(df, args.include_attacker_features)
    dropped = [c for c in df.columns if c.endswith(CLOCK_SUFFIXES)]
    print(f"[feat]  {len(columns)} numeric feature candidates "
          f"({len(dropped)} clock columns excluded)")

    y_all, _ = labels(df, args.label)
    modes = SPLIT_MODES if args.split_mode == "both" else (args.split_mode,)
    for mode in modes:
        assignment = (assign_block_split(df, y_all, args.seed) if mode == "block"
                      else assign_scenario_split(df))
        build(df, assignment, columns, args, mode)


if __name__ == "__main__":
    main()
