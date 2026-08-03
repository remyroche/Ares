from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.reconstruct_2022_2024_base_residual_stack_oof import (
    ALPHA_TARGET,
    _soft_alpha,
    add_anchors,
    eligible_raw_columns,
    load_pf_population,
    period_metrics,
)


def test_soft_alpha_contract() -> None:
    values = pd.Series(
        ["favorable_first", "timeout", "adverse_first_or_conflict"]
    )
    np.testing.assert_allclose(_soft_alpha(values), [1.0, 0.5, 0.0])


def test_add_anchors_are_side_timestamp_local() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2022-01-03"] * 3, utc=True),
            "side_name": ["long"] * 3,
            "base_score": [0.2, 0.9, 0.5],
        }
    )
    anchored, anchors = add_anchors(frame)
    assert {"base_prediction", "base_rank_timestamp_side"}.issubset(anchors)
    assert anchored["base_rank_timestamp_side"].tolist() == [3.0, 1.0, 2.0]


def test_period_metrics_uses_global_tail() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2022-01-03", periods=20, freq="h", tz="UTC"),
            "candidate_id": [f"c{i}" for i in range(20)],
            "side_name": ["long", "short"] * 10,
            "stack_lineage": ["x"] * 20,
            ALPHA_TARGET: np.linspace(0, 1, 20),
            "score_base_alpha": np.linspace(0, 1, 20),
            "score_residual_alpha": np.linspace(0, 1, 20),
            "score_residual_expected_ev": np.linspace(0, 1, 20),
            "execution_net_ev_12h": np.arange(20) / 10_000,
            "execution_gross_ev_12h": np.arange(20) / 10_000,
            "execution_cost_return": np.zeros(20),
        }
    )
    metrics = period_metrics(frame)
    week = metrics.loc[metrics["frequency"].eq("week")].iloc[0]
    assert week["selected_rows"] == 2
    assert week["top10_net_ev_bps"] == 18.5


def test_exec_diagnostics_are_forbidden(tmp_path) -> None:
    pd.DataFrame(
        {
            "exec_margin": [1.0],
            "safe_feature": [2.0],
            **{f"feature_{i}": [float(i)] for i in range(100)},
        }
    ).to_parquet(tmp_path / "features.parquet", index=False)
    selected = eligible_raw_columns(tmp_path / "features.parquet")
    assert "exec_margin" not in selected
    assert "safe_feature" in selected


def _pf_fixture(tmp_path):
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {f"feature_{index}": [float(index)] for index in range(101)}
    ).to_parquet(source, index=False)
    timestamp = pd.Timestamp("2024-04-01T00:00:00Z")
    stage = tmp_path / "stage.parquet"
    pd.DataFrame(
        {
            "candidate_id": ["candidate-1"],
            "signal_timestamp": [timestamp],
            "symbol": ["BTC/USD:USD"],
            "side_name": ["long"],
            "base_score": [0.8],
            "source_row_number": [0],
            "source_shard_path": [str(source)],
        }
    ).to_parquet(stage, index=False)
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "candidate_id": ["candidate-1"],
            "__ts__": [timestamp],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "__soft_tb_first_event__": ["favorable_first"],
            "execution_net_ev_12h": [0.01],
            "execution_gross_ev_12h": [0.02],
            "execution_cost_return": [0.01],
        }
    ).to_parquet(labels, index=False)
    return stage, labels


def test_pf_labels_must_be_disjoint_and_cover_stage_exactly(tmp_path) -> None:
    stage, labels = _pf_fixture(tmp_path)
    with pytest.raises(ValueError, match="overlap across supplied label"):
        load_pf_population([stage], [labels, labels])


def test_pf_stage_and_label_identity_must_match(tmp_path) -> None:
    stage, labels = _pf_fixture(tmp_path)
    changed = pd.read_parquet(labels)
    changed["__symbol__"] = "ETH/USD:USD"
    changed.to_parquet(labels, index=False)
    with pytest.raises(ValueError, match="candidate identities differ"):
        load_pf_population([stage], [labels])
