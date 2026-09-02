from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from extreme_price_movements.exact_1m_policy_contract import Exact1mExecutionContract
from scripts.run_strict_r3_exact_1m_policy_hpo import run


def test_sequential_runner_preserves_exact_exit_timestamps(tmp_path) -> None:
    contract = Exact1mExecutionContract()
    # The production contract rejects a funnel fitted on one calendar month;
    # this compact fixture still covers every compatible calibration/selection
    # month (the strict ledger begins in February 2024).
    timestamps = pd.DatetimeIndex(
        np.concatenate([
            pd.date_range(f"2024-{month:02d}-01", periods=100, freq="h", tz="UTC").to_numpy()
            for month in range(2, 13)
        ])
    )
    n = len(timestamps)
    rows = pd.DataFrame(
        {
            "candidate_id": [f"c{idx:04d}" for idx in range(n)],
            "timestamp": timestamps,
            "symbol": [f"S{idx % 10}/USD:USD" for idx in range(n)],
            "score": np.linspace(0.8, 1.0, n),
            "entry_ts": timestamps + pd.Timedelta(minutes=5),
            "entry_price": 100.0,
            "signal_atr": 1.0,
            "path_valid": True,
        }
    )
    high = np.full((n, 720), 100.0, dtype=np.float32)
    low = np.full((n, 720), 99.8, dtype=np.float32)
    close = np.full((n, 720), 100.0, dtype=np.float32)
    high[:, 0] = 103.0
    low[:, 1] = 100.5
    close[:, 1] = 100.4
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows.to_parquet(dataset / "training_rows.parquet", index=False)
    np.savez_compressed(
        dataset / "exact_paths.npz",
        entry=np.full(n, 100.0), atr=np.full(n, 1.0), high=high, low=low,
        close=close, candidate_id=rows["candidate_id"].to_numpy(dtype="U"),
    )
    (dataset / "dataset_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_exact_1m_policy_hpo_dataset_v1",
        "contract": contract.to_dict(), "contract_hash": contract.hash,
        "routing": "score-only", "path": "complete 720x1m",
        "candidate_rows": n, "valid_training_rows": n, "invalid_rows": 0,
    }))
    output = run(argparse.Namespace(
        dataset_dir=dataset, out_dir=tmp_path / "out", execution_surface="live_parent", core_trials=1,
        refine_trials=1, polish_trials=1, live_broad_parents=1, live_refine_parents=1,
        time_trials=1, geometry_trials=1, protection_trials=1,
        stage_parents=1, finalists=1, seed=17, overwrite=False,
        min_monthly_support=75, min_path_coverage=0.90,
        hpo_rows_per_month=75,
    ))
    winner = json.loads((output / "winner.json").read_text())
    assert winner["schema"] == "strict_r3_exact_1m_policy_hpo_v1"
    assert winner["contract_hash"] == contract.hash
    assert winner["execution_surface"] == "live_parent"
    assert winner["live_parent_compatible"] is True
    candidate = pd.read_parquet(output / "finalist_1_portfolio_candidates.parquet")
    assert pd.to_datetime(candidate["exit_timestamp"], utc=True).notna().all()
    assert (pd.to_numeric(candidate["holding_bars"]) <= 720).all()
