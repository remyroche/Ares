#!/usr/bin/env python3
"""Synthetic benchmark for Stage-I target-cell bounded parallel execution.

This never reads production candidates or starts the target experiment.  It
compares identical deterministic scalar cells in a temporary directory and
verifies exact prediction parity before reporting elapsed time and peak-memory
admission estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage_i_base_target_ablation import (  # noqa: E402
    _run_model_cell,
    parallel_memory_admission,
    run_parallel_model_cells,
)


def _frame(rows: int, features: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if rows < 400 or features < 1:
        raise ValueError("benchmark requires >=400 rows and >=1 feature")
    rng = np.random.default_rng(20260803)
    decision = pd.date_range("2024-01-01", periods=rows // 2, freq="h", tz="UTC").repeat(2)
    rows = len(decision)
    side = np.tile(["long", "short"], rows // 2)
    matrix = rng.normal(size=(rows, features)).astype(np.float32)
    signal = matrix[:, 0] + .25 * matrix[:, min(1, features - 1)]
    metadata = pd.DataFrame({
        "candidate_id": [f"bench-{item}" for item in range(rows)],
        "__ts__": decision - pd.Timedelta(hours=1), "__symbol__": ["X"] * rows,
        "side_name": side, "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "net_bps": 80 * signal + rng.normal(0, 25, rows),
        "causal_regime": np.where(matrix[:, -1] > 0, "trend", "chop"),
        "contract_certainty": np.clip(np.abs(signal), 0, 1),
        "target": 1 / (1 + np.exp(-signal)),
    })
    names = [f"f{index}" for index in range(features)]
    frame = pd.concat(
        [metadata.reset_index(drop=True), pd.DataFrame(matrix, columns=names)], axis=1,
    )
    population = frame[[
        "candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts",
    ]].copy()
    population["label_valid"] = True
    selected = {
        "contract_sha256": "b" * 64,
        "sides": {
            side_name: {"selected_features": names, "fixed_params": {
                "num_leaves": 31, "n_estimators": 80, "learning_rate": .04,
                "min_child_samples": 40, "max_bin": 127,
            }}
            for side_name in ("long", "short")
        },
    }
    return frame, population, selected


def _job(root: Path, shared: tuple[pd.DataFrame, pd.DataFrame, dict]) -> dict:
    frame, population, selected = shared
    return {
        "root": root, "frame": frame, "arm": None, "target_column": "target",
        "family": "scalar_S", "selected_contract": selected,
        "development_seed": 11, "evaluation_fraction": .25,
        "min_train_rows": 100, "weight_mode": "uniform",
        "regime_column": "causal_regime", "resume": False,
        "experiment_input_sha256": "e" * 64, "population": population,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--workers", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument(
        "--assert-break-even", action="store_true",
        help="fail unless bounded parallel execution is faster on this host",
    )
    args = parser.parse_args()
    shared = _frame(args.rows, args.features)
    with tempfile.TemporaryDirectory(prefix="stage_i_target_benchmark_") as raw:
        root = Path(raw)
        sequential_jobs = [_job(root / "sequential" / str(i), shared) for i in range(args.jobs)]
        started = time.perf_counter()
        cache = None
        for job in sequential_jobs:
            _, cache = _run_model_cell(**job, model_cache=cache)
        sequential_seconds = time.perf_counter() - started
        parallel_jobs = [_job(root / "parallel" / str(i), shared) for i in range(args.jobs)]
        preflight_admission = parallel_memory_admission(parallel_jobs, workers=args.workers)
        started = time.perf_counter()
        _, measured_admission = run_parallel_model_cells(parallel_jobs, workers=args.workers)
        parallel_seconds = time.perf_counter() - started
        for left, right in zip(sequential_jobs, parallel_jobs, strict=True):
            left_frame = pd.read_parquet(left["root"] / "target_repair_development_predictions.parquet")
            right_frame = pd.read_parquet(right["root"] / "target_repair_development_predictions.parquet")
            pd.testing.assert_frame_equal(left_frame, right_frame, check_exact=True)
    speedup = sequential_seconds / max(parallel_seconds, 1e-12)
    result = {
        "schema": "stage_i_target_parallel_benchmark_v1",
        "synthetic_only": True, "rows": int(args.rows), "features": int(args.features),
        "jobs": int(args.jobs), "workers": int(args.workers),
        "sequential_seconds": sequential_seconds, "parallel_seconds": parallel_seconds,
        "speedup": speedup, "break_even_achieved": bool(speedup > 1.0),
        "exact_prediction_parity": True,
        "preflight_memory_admission": preflight_admission,
        "measured_memory_admission": measured_admission,
        "peak_rss_calibration": {
            "estimated_worker_peak_bytes": measured_admission["estimated_worker_peak_bytes"],
            "measured_worker_peak_rss_max_bytes": measured_admission["measured_worker_peak_rss_max_bytes"],
            "measured_to_estimated_ratio": measured_admission["measured_to_estimated_worker_peak_ratio"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.assert_break_even and speedup <= 1.0:
        raise RuntimeError(f"parallel target runner did not break even: speedup={speedup:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
