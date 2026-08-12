from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_base_target_ablation import BaseTargetAblationError, _fixed_params
from scripts.run_stage_i_base_target_ablation import (
    _execute_model_job_batch,
    _run_model_cell,
    _terminate_and_reap_workers,
    parallel_memory_admission,
    run_parallel_model_cells,
)
import scripts.run_stage_i_base_target_ablation as target_runner


def _inputs(rows: int = 240) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(8841)
    decision = pd.date_range("2025-01-01", periods=rows // 2, freq="h", tz="UTC").repeat(2)
    signal = rng.normal(size=rows)
    frame = pd.DataFrame({
        "candidate_id": [f"p-{item}" for item in range(rows)],
        "__ts__": decision - pd.Timedelta(hours=1),
        "__symbol__": ["X"] * rows,
        "side_name": np.tile(["long", "short"], rows // 2),
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "net_bps": signal * 50 + rng.normal(size=rows),
        "causal_regime": np.where(signal > 0, "up", "down"),
        "contract_certainty": np.clip(np.abs(signal), 0, 1),
        "target": 1 / (1 + np.exp(-signal)),
        "f": signal,
    })
    population = frame[[
        "candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts",
    ]].copy()
    population["label_valid"] = True
    selected = {
        "contract_sha256": "c" * 64,
        "sides": {
            side: {
                "selected_features": ["f"],
                "fixed_params": {
                    "num_leaves": 7, "n_estimators": 10, "learning_rate": .05,
                    "min_child_samples": 5, "max_bin": 31,
                },
            }
            for side in ("long", "short")
        },
    }
    return frame, population, selected


def _job(
    root: Path, *, bad: bool = False, resume: bool = False,
    family: str = "scalar_S",
) -> dict:
    frame, population, selected = _inputs()
    if family == "ordinal_O":
        frame["target"] = np.arange(len(frame), dtype=np.int8) % 5
    elif family == "R3_control":
        frame["target"] = np.arange(len(frame), dtype=np.int8) % 3
    return {
        "root": root, "frame": frame, "arm": None,
        "target_column": "absent" if bad else "target", "family": family,
        "selected_contract": selected, "development_seed": 11,
        "evaluation_fraction": .25, "min_train_rows": 30,
        "weight_mode": "uniform", "regime_column": "causal_regime",
        "resume": resume, "experiment_input_sha256": "d" * 64,
        "population": population,
    }


def test_parallel_memory_admission_falls_back_before_starting_workers() -> None:
    jobs = [_job(Path("unused-a")), _job(Path("unused-b"))]
    rejected = parallel_memory_admission(
        jobs, workers=3, available_bytes=1, budget_fraction=.65,
    )
    assert rejected["admitted"] is False
    assert rejected["reason"] == "insufficient_memory_budget"
    admitted = parallel_memory_admission(
        jobs, workers=3, available_bytes=10**12, budget_fraction=.65,
    )
    assert admitted["admitted"] is True
    assert admitted["active_workers"] == 2
    assert admitted["lightgbm_threads_per_model"] == 1


def test_high_dimensional_memory_guard_reduces_workers_dynamically() -> None:
    rng = np.random.default_rng(1)
    columns = [f"f{index}" for index in range(220)]
    frame = pd.DataFrame(rng.normal(size=(10_000, len(columns))).astype(np.float32), columns=columns)
    frame["candidate_id"] = np.arange(len(frame)).astype(str)
    population = frame[["candidate_id"]].copy()
    selected = {
        "sides": {side: {"selected_features": columns} for side in ("long", "short")},
    }
    jobs = [{
        "frame": frame, "population": population, "selected_contract": selected,
        "development_seed": 11,
    } for _ in range(4)]
    audit = parallel_memory_admission(
        jobs, workers=4, available_bytes=1_700_000_000, budget_fraction=.65,
    )
    assert audit["selected_feature_count"] == 220
    assert audit["worker_raw_multiplier"] == 8.0
    assert 2 <= audit["active_workers"] < 4
    assert audit["admitted"] is True


def test_lightgbm_thread_aliases_are_all_forced_to_one() -> None:
    params = _fixed_params(
        {"num_threads": 99, "num_thread": 98, "nthread": 97, "n_jobs": 96},
        seed=11, objective="binary",
    )
    assert params["n_jobs"] == 1
    assert not {"num_threads", "num_thread", "nthread"}.intersection(params)


@pytest.mark.parametrize("family", ["scalar_S", "ordinal_O", "R3_control"])
def test_parallel_and_sequential_cells_have_exact_prediction_parity(
    tmp_path: Path, family: str,
) -> None:
    parallel = _job(tmp_path / "parallel", family=family)
    peer = _job(tmp_path / "parallel-peer", family=family)
    _, audit = run_parallel_model_cells(
        [parallel, peer], workers=2, available_memory_bytes=10**12,
    )
    assert audit["completed_jobs"] == 2 and not audit["failed_jobs"]
    assert audit["measured_worker_peak_rss_max_bytes"] > 0
    assert audit["measured_to_estimated_worker_peak_ratio"] > 0
    sequential = _job(tmp_path / "sequential", family=family)
    _run_model_cell(**sequential, model_cache=None)
    expected = pd.read_parquet(
        sequential["root"] / "target_repair_development_predictions.parquet"
    ).sort_values("candidate_id").reset_index(drop=True)
    actual = pd.read_parquet(
        parallel["root"] / "target_repair_development_predictions.parquet"
    ).sort_values("candidate_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_sequential_memory_fallback_has_full_prediction_and_metric_parity(tmp_path: Path) -> None:
    fallback_jobs = [_job(tmp_path / "fallback-a"), _job(tmp_path / "fallback-b")]
    fallback_rows, fallback_audit = _execute_model_job_batch(
        fallback_jobs, workers=1, memory_budget_fraction=.65,
    )
    assert fallback_audit["execution_mode"] == "sequential_shared_lightgbm_bin_cache"
    parallel_jobs = [_job(tmp_path / "parallel-a"), _job(tmp_path / "parallel-b")]
    parallel_rows, parallel_audit = _execute_model_job_batch(
        parallel_jobs, workers=2, memory_budget_fraction=.90,
    )
    assert parallel_audit["execution_mode"] == "bounded_fresh_processes_read_only_arrow_mmap"
    for fallback, parallel, fallback_job, parallel_job in zip(
        fallback_rows, parallel_rows, fallback_jobs, parallel_jobs, strict=True,
    ):
        pd.testing.assert_frame_equal(fallback, parallel, check_exact=True)
        fallback_prediction = pd.read_parquet(
            fallback_job["root"] / "target_repair_development_predictions.parquet"
        )
        parallel_prediction = pd.read_parquet(
            parallel_job["root"] / "target_repair_development_predictions.parquet"
        )
        pd.testing.assert_frame_equal(fallback_prediction, parallel_prediction, check_exact=True)


def test_parallel_resume_verifies_completed_hash_bound_cell(tmp_path: Path) -> None:
    jobs = [_job(tmp_path / "a"), _job(tmp_path / "b")]
    run_parallel_model_cells(jobs, workers=2, available_memory_bytes=10**12)
    resumed = [dict(job, resume=True) for job in jobs]
    results, audit = run_parallel_model_cells(
        resumed, workers=2, available_memory_bytes=10**12,
    )
    assert audit["completed_jobs"] == 2
    assert all(result["resumed"] for result in results)


def test_resume_rejects_stale_source_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path / "source-bound")
    _run_model_cell(**job, model_cache=None)
    monkeypatch.setattr(target_runner, "_model_cell_source_fingerprint", lambda: {
        "schema": "stage_i_target_model_source_fingerprint_v1",
        "runner_sha256": "0" * 64, "target_module_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
    })
    with pytest.raises(BaseTargetAblationError, match="request drift"):
        _run_model_cell(**dict(job, resume=True), model_cache=None)


def test_interrupt_cleanup_terminates_and_reaps_every_worker() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    audit = _terminate_and_reap_workers({process.pid: (0, process)}, timeout_seconds=.05)
    assert audit["live_workers_terminated"] == 1
    assert audit["all_workers_reaped"] is True
    assert process.poll() is not None


def test_ipc_root_is_removed_when_parallel_setup_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ipc_root = tmp_path / "controlled-ipc-root"

    def controlled_mkdtemp(*args: object, **kwargs: object) -> str:
        ipc_root.mkdir()
        return str(ipc_root)

    monkeypatch.setattr(target_runner.tempfile, "mkdtemp", controlled_mkdtemp)
    monkeypatch.setattr(
        target_runner, "_write_read_only_ipc",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        run_parallel_model_cells(
            [_job(tmp_path / "a"), _job(tmp_path / "b")],
            workers=2, available_memory_bytes=10**12,
        )
    assert not ipc_root.exists()


def test_parallel_failure_does_not_discard_healthy_arm(tmp_path: Path) -> None:
    good = _job(tmp_path / "good")
    bad = _job(tmp_path / "bad", bad=True)
    with pytest.raises(BaseTargetAblationError, match="healthy cells were preserved"):
        run_parallel_model_cells(
            [good, bad], workers=2, available_memory_bytes=10**12,
        )
    assert (good["root"] / "manifest.json").is_file()
    failure_path = bad["root"].with_name("bad.failure.json")
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed"
    assert failure["experiment_input_sha256"] == "d" * 64
    assert len(failure["failure_sha256"]) == 64
