from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from extreme_price_movements.inference.p8u_c1_mc1_inference_package import FEATURES


def _source(rows_per_hour: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    hours = pd.date_range("2025-11-01T00:00:00Z", "2026-05-31T23:00:00Z", freq="h")
    timestamp = np.repeat(hours.to_numpy(), rows_per_hour)
    rows = len(timestamp)
    rng = np.random.default_rng(1729)
    score = pd.DataFrame({
        "candidate_id": [f"r{idx:06d}" for idx in range(rows)],
        "__decision_ts__": pd.to_datetime(timestamp, utc=True),
        "__symbol__": "TEST/USD:USD", "side_name": "long",
        "policy_path_valid": True, "policy_gross_bps": 180.0,
        "policy_net_bps": rng.normal(80, 120, rows), "policy_exit_bar_15m": 1,
        "policy_entry_price": 1.0, "policy_exit_price": 1.01,
        "policy_exit_reason": "timeout", "policy_label_available_ts": pd.to_datetime(timestamp, utc=True) + pd.Timedelta(hours=12),
        "policy_outcome_source": "test", "policy_cost_bps": 100.0,
    })
    for index, field in enumerate(FEATURES[:6]):
        score[field] = np.clip(rng.normal(.6 + index / 100, .1, rows), 0.0, 1.0)
    snapshot = score.loc[:, ["candidate_id", "__decision_ts__"]].rename(columns={"__decision_ts__": "snapshot_ts"}).copy()
    for index, field in enumerate(FEATURES[6:-1]):
        snapshot[field] = rng.normal(index / 10, .2, rows)
    return score, snapshot


def test_c1_prequential_producer_seals_target_free_and_replay_views(tmp_path: Path) -> None:
    score, snapshot = _source()
    bcf, current, sr = tmp_path / "bcf.parquet", tmp_path / "current.parquet", tmp_path / "sr.parquet"
    score.to_parquet(bcf, index=False)
    score.to_parquet(current, index=False)
    snapshot.to_parquet(sr, index=False)
    output = tmp_path / "out"
    command = [
        sys.executable, "scripts/build_p8u_c1_lva_dual_mc1_prequential.py",
        "--bcf", str(bcf), "--current", str(current), "--c1-snapshots", str(sr),
        "--held-start", "2026-05-01T00:00:00Z", "--held-end", "2026-06-01T00:00:00Z",
        "--output", str(output),
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    target_free = pd.read_parquet(output / "dual_target_free_predictions.parquet")
    assert not set(column for column in target_free.columns if column.startswith("policy_") or column == "policy_net_bps")
    assert len(target_free) == int(score["__decision_ts__"].ge("2026-05-01").sum())
    replay = pd.read_parquet(output / "dual_outcome_replay_panel.parquet")
    assert "policy_net_bps" in replay.columns
    config = (output / "c1_mc1_selector_config.json").read_text()
    assert '"order_submission": false' in config
