import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_pre2026_model_derived_mechanics_representation.py"
SPEC = importlib.util.spec_from_file_location("mechanics_runner", RUNNER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def test_schema_is_small_causal_intersection_and_targets_are_not_features():
    assert len(M.MECHANICS) == 15
    assert not set(M.MECHANICS).intersection(M.HEADS.values())
    assert set(M.CORE).isdisjoint(M.MECHANICS)
    assert all("target" not in field and "mfe" not in field and "mae" not in field for field in M.MECHANICS)


def test_join_vintage_fails_closed_on_identity_mismatch(tmp_path: Path):
    anchor = pd.DataFrame({
        "candidate_id": ["a"], "__ts__": ["2023-04-01T00:00:00Z"], "__symbol__": ["BTC"], "side_name": ["long"],
        "execution_label_end_utc": ["2023-04-01T12:00:00Z"], "execution_label_available_at": ["2023-04-01T12:00:00Z"],
        "execution_net_ev_12h": [0.01], "execution_gross_ev_12h": [0.02], "execution_cost_return": [0.001],
        "score_base_alpha": [0.1], "score_residual_expected_ev": [0.2],
    })
    source = pd.DataFrame({"candidate_id": ["a"], "__ts__": ["2023-04-01T00:00:00Z"], "__symbol__": ["ETH"], "side_name": ["long"], **{name: [0.0] for name in M.MECHANICS}})
    ap, sp = tmp_path / "anchor.parquet", tmp_path / "source.parquet"
    anchor.to_parquet(ap); source.to_parquet(sp)
    with pytest.raises(ValueError, match="identity mismatch"):
        M._join_vintage(M._anchor_frame(ap), sp, "early")


def test_representation_score_keeps_global_not_per_timestamp_admission():
    rows = []
    for candidate, timestamp, residual, side in [("a", "2023-04-01T00:00:00Z", .8, "long"), ("b", "2023-04-01T00:00:00Z", .2, "short"), ("c", "2023-04-01T01:00:00Z", .7, "long"), ("d", "2023-04-01T01:00:00Z", .1, "short")]:
        for arm in M.ARMS:
            for head in M.HEADS:
                rows.append({"candidate_id": candidate, "outer_month": "2023-04", "__ts__": pd.Timestamp(timestamp), "__symbol__": "BTC", "side_name": side, "execution_net_ev_12h": .01, "score_residual_expected_ev": residual, "arm": arm, "head": head, "actual_target": 1, "prediction": .5})
    scored, economics = M.score_and_economics(pd.DataFrame(rows))
    assert len(scored) == 4
    assert economics.selected_rows.eq(1).all()  # ceil(4 * 10%), once for the whole month, not twice by timestamp
