"""Run the 3 methods x 2 datasets matrix over independent training seeds.

Review 3 / Member 1. Each seed re-trains and re-selects the validation
threshold on the frozen project-standard split.

Usage:
    python scripts/run_multiseed.py --seeds 1-20 --save
    python scripts/run_multiseed.py --seeds 1-3 --datasets data4cyber --methods base --save
    python scripts/run_multiseed.py --seeds 1-20 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.evaluator import D6_COLUMNS  # noqa: E402
from common.multiseed_stats import (  # noqa: E402
    HEADLINE_DATASETS,
    HEADLINE_METHODS,
)

ABLATION_CONFIGS = tuple(f"P{i}" for i in range(8))


def parse_seed_spec(spec: str) -> list[int]:
    """Parse '1-20', '1,2,3', or '42'."""
    spec = spec.strip()
    if not spec:
        raise ValueError("empty seed spec")
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid seed range {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    # preserve order, drop duplicates
    seen: set[int] = set()
    ordered: list[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            ordered.append(seed)
    return ordered


def jobs_for(
    seeds: list[int],
    datasets: list[str],
    methods: list[str],
    include_ablation: bool,
    split_policy: str,
) -> list[dict[str, str | int]]:
    jobs: list[dict[str, str | int]] = []
    method_set = set(methods)
    for seed in seeds:
        for dataset in datasets:
            for method, config_name, _label in HEADLINE_METHODS:
                if method not in method_set:
                    continue
                jobs.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "config": config_name,
                        "split_policy": split_policy,
                        "seed": seed,
                    }
                )
            if include_ablation and "proposed" in method_set:
                for cfg in ABLATION_CONFIGS:
                    if cfg == "P6":
                        continue
                    jobs.append(
                        {
                            "dataset": dataset,
                            "method": "proposed",
                            "config": cfg,
                            "split_policy": split_policy,
                            "seed": seed,
                        }
                    )
    return jobs


def json_path_for(raw_dir: Path, job: dict[str, str | int]) -> Path:
    seed = int(job["seed"])
    name = (
        f"{job['method']}_{job['dataset']}_{job['config']}_"
        f"{job['split_policy']}_seed{seed}.json"
    )
    return raw_dir / f"seed_{seed}" / name


def run_one(job: dict[str, str | int], raw_dir: Path, python_exe: str) -> int:
    seed = int(job["seed"])
    out_dir = raw_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(ROOT / "run_experiment.py"),
        "--dataset",
        str(job["dataset"]),
        "--method",
        str(job["method"]),
        "--config",
        str(job["config"]),
        "--split-policy",
        str(job["split_policy"]),
        "--seed",
        str(seed),
        "--save",
        "--output-dir",
        str(out_dir),
    ]
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return int(completed.returncode)


def append_combined_csv(raw_dir: Path, job: dict[str, str | int]) -> None:
    seed = int(job["seed"])
    src = raw_dir / f"seed_{seed}" / f"{job['method']}_{job['dataset']}_results.csv"
    dest = raw_dir / "multiseed_all_runs.csv"
    if not src.is_file():
        return
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if str(row.get("seed")) == str(seed)
                and str(row.get("config")) == str(job["config"])
                and str(row.get("method")) == str(job["method"])]
    if not rows:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not dest.is_file() or dest.stat().st_size == 0
    with dest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=D6_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in D6_COLUMNS})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="1-20", help="Seed spec, e.g. 1-20 or 1,2,3")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(HEADLINE_DATASETS),
        choices=list(HEADLINE_DATASETS),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["base", "saurabh", "proposed"],
        choices=["base", "saurabh", "proposed"],
    )
    p.add_argument("--split-policy", default="project_standard")
    p.add_argument(
        "--include-ablation",
        action="store_true",
        help="Also run proposed P0–P7 (needed for a multi-seed Table 8.1)",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if JSON already exists")
    p.add_argument("--dry-run", action="store_true", help="Print jobs without executing")
    p.add_argument("--save", action="store_true", default=True, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    seeds = parse_seed_spec(args.seeds)
    raw_dir = ROOT / "results" / "raw"
    jobs = jobs_for(
        seeds=seeds,
        datasets=list(args.datasets),
        methods=list(args.methods),
        include_ablation=args.include_ablation,
        split_policy=args.split_policy,
    )

    print(f"Planned jobs: {len(jobs)}  seeds={seeds}  datasets={args.datasets}  "
          f"methods={args.methods}", flush=True)

    failures = 0
    skipped = 0
    ran = 0
    t0 = time.perf_counter()
    for i, job in enumerate(jobs, start=1):
        dest = json_path_for(raw_dir, job)
        prefix = f"[{i}/{len(jobs)}] seed={job['seed']} {job['dataset']} {job['method']} {job['config']}"
        if dest.is_file() and not args.force:
            print(f"{prefix}  SKIP existing {dest}", flush=True)
            skipped += 1
            continue
        if args.dry_run:
            print(f"{prefix}  DRY-RUN -> {dest}", flush=True)
            continue
        print(f"{prefix}  RUN", flush=True)
        code = run_one(job, raw_dir, sys.executable)
        if code != 0:
            print(f"{prefix}  FAILED exit={code}", flush=True)
            failures += 1
            continue
        append_combined_csv(raw_dir, job)
        ran += 1

    elapsed = time.perf_counter() - t0
    print(
        f"Done. ran={ran} skipped={skipped} failed={failures} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
