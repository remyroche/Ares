from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_market_state_head_priority_learning as priority_learning
from scripts.replay_market_state_learned_priority_cap_sweep import (
    rescale_learned_schedule,
    select_shadow_challenger,
)
from scripts.run_market_state_head_priority_learning import (
    load_train_deployable_for_static_contract,
)
from scripts.run_market_state_head_priority_learning import replay_selection_metrics


def test_rescale_learned_schedule_recomputes_bounded_adjustments() -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
                utc=True,
            ),
            "head": ["short_asset", "short_boll"],
            "centered_head_score": [-1.0, 1.0],
            "priority_scale": [1.0, 1.0],
            "portfolio_priority_adjustment": [-0.20, 0.20],
            "portfolio_priority_multiplier": [0.5, 2.0],
            "portfolio_rank_adjustment": [-0.4, 0.4],
            "priority_arm": ["old", "old"],
        }
    )

    out = rescale_learned_schedule(schedule, max_adjustment=0.05, arm="cap_005")

    assert out["priority_arm"].eq("cap_005").all()
    assert out["portfolio_priority_adjustment"].abs().max() <= 0.05 + 1e-12
    assert np.isclose(out["portfolio_priority_adjustment"].sum(), 0.0)
    assert out.loc[out["head"].eq("short_boll"), "portfolio_priority_adjustment"].iloc[0] > 0.0
    assert out["portfolio_priority_multiplier"].eq(1.0).all()
    assert out["portfolio_rank_adjustment"].eq(0.0).all()


def test_rescale_learned_schedule_can_gate_weak_separation() -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
                utc=True,
            ),
            "head": ["short_asset", "short_boll"],
            "centered_head_score": [-0.25, 0.25],
            "priority_scale": [1.0, 1.0],
            "portfolio_priority_adjustment": [-0.20, 0.20],
            "priority_arm": ["old", "old"],
        }
    )

    out = rescale_learned_schedule(
        schedule,
        max_adjustment=0.20,
        min_abs_z=0.50,
        arm="gated",
    )

    assert out["portfolio_priority_adjustment"].eq(0.0).all()
    assert out["portfolio_priority_multiplier"].eq(1.0).all()
    assert out["portfolio_rank_adjustment"].eq(0.0).all()


def test_rescale_learned_schedule_requires_raw_score_columns() -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z"], utc=True),
            "head": ["short_asset"],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        rescale_learned_schedule(schedule, max_adjustment=0.05, arm="bad")


def test_load_train_deployable_applies_static_global_rank_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.parquet"
    manifest_path = tmp_path / "static_manifest.json"
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z"], utc=True),
            "head": ["short_boll"],
            "rank_marker": [0.1],
        }
    )
    transformed = raw.assign(rank_marker=0.9)
    raw.to_parquet(train_path, index=False)
    manifest_path.write_text(
        """
        {
          "active_stack": {
            "rank_contract": "anchor_global_policy_rank_reference",
            "rank_reference_run_id": "rank_ref",
            "disabled_heads": ["long_bars", "long_dist"]
          }
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        priority_learning,
        "_load_candidates",
        lambda path: raw.copy(),
    )
    fake_materializer = types.ModuleType("scripts.materialize_t1_repaired_static_baseline")

    def fake_load_for_t1(
        path: Path,
        *,
        rank_contract: str,
        disabled_heads: set[str],
        data_root: Path,
        rank_reference_run_id: str,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        assert path == train_path
        assert rank_contract == "anchor_global_policy_rank_reference"
        assert disabled_heads == {"long_bars", "long_dist"}
        assert data_root == Path("data_perp")
        assert rank_reference_run_id == "rank_ref"
        return transformed.copy(), {"rank_source": "policy_rank_reference_percentile"}

    fake_materializer._load_for_t1 = fake_load_for_t1
    monkeypatch.setitem(
        sys.modules,
        "scripts.materialize_t1_repaired_static_baseline",
        fake_materializer,
    )

    out, diag = load_train_deployable_for_static_contract(
        train_path,
        static_baseline_manifest=manifest_path,
    )

    assert out["rank_marker"].tolist() == [0.9]
    assert diag["applied_static_rank_contract"] is True
    assert diag["rank_contract"] == "anchor_global_policy_rank_reference"
    assert diag["rank_reference_run_id"] == "rank_ref"


def test_select_shadow_challenger_prefers_smaller_near_best_cap() -> None:
    metrics = pd.DataFrame(
        {
            "arm": ["cap_0p15_zge_0p5", "cap_0p20_zge_0p5", "cap_0p15_zge_0p75"],
            "gate_passed": [True, True, True],
            "delta_net_pnl": [22.0, 22.0, 17.5],
            "max_adjustment": [0.15, 0.20, 0.15],
            "min_abs_z": [0.50, 0.50, 0.75],
            "active_schedule_share": [0.34, 0.34, 0.08],
            "accepted_jaccard": [0.966, 0.966, 0.966],
            "entrants": [2, 2, 2],
            "removed": [2, 2, 2],
        }
    )

    selected = select_shadow_challenger(metrics)

    assert selected["selected"] is True
    assert selected["arm"] == "cap_0p15_zge_0p5"
    assert selected["gate_passing_count"] == 3
    assert selected["risk_safe_gate_passing_count"] == 3
    assert selected["near_best_count"] == 2


def test_select_shadow_challenger_prefers_safe_opportunity_router() -> None:
    metrics = pd.DataFrame(
        {
            "arm": ["cap_0p3", "cap_0p6"],
            "gate_passed": [True, True],
            "delta_net_pnl": [25.8, 46.7],
            "max_adjustment": [0.30, 0.60],
            "min_abs_z": [0.0, 0.0],
            "active_schedule_share": [1.0, 1.0],
            "accepted_jaccard": [0.955, 0.938],
            "delta_full_sl_rate": [0.003, -0.007],
            "delta_timeout_rate": [-0.002, 0.006],
            "entrants": [4, 6],
            "removed": [1, 1],
            "net_replacement_pnl": [17.2, 37.5],
        }
    )

    selected = select_shadow_challenger(metrics)

    assert selected["selected"] is True
    assert selected["arm"] == "cap_0p3"
    assert selected["best_delta_net_pnl"] == pytest.approx(25.8)
    assert selected["gate_passing_count"] == 2
    assert selected["risk_safe_gate_passing_count"] == 1
    assert "risk_safe" in selected["reason"]


def test_select_shadow_challenger_falls_back_when_no_safe_arm() -> None:
    metrics = pd.DataFrame(
        {
            "arm": ["cap_0p3", "cap_0p6"],
            "gate_passed": [True, True],
            "delta_net_pnl": [25.8, 46.7],
            "max_adjustment": [0.30, 0.60],
            "min_abs_z": [0.0, 0.0],
            "accepted_jaccard": [0.90, 0.938],
            "delta_full_sl_rate": [0.020, 0.010],
            "delta_timeout_rate": [0.004, 0.006],
        }
    )

    selected = select_shadow_challenger(metrics)

    assert selected["selected"] is True
    assert selected["arm"] == "cap_0p6"
    assert selected["risk_safe_gate_passing_count"] == 0
    assert "fallback" in selected["reason"]


def test_select_shadow_challenger_reports_no_passing_arm() -> None:
    metrics = pd.DataFrame(
        {
            "arm": ["cap_0p10", "cap_0p15"],
            "gate_passed": [False, False],
            "delta_net_pnl": [3.0, 10.0],
            "max_adjustment": [0.10, 0.15],
            "min_abs_z": [0.0, 0.5],
        }
    )

    selected = select_shadow_challenger(metrics)

    assert selected["selected"] is False
    assert selected["reason"] == "no_gate_passing_positive_delta_arm"


def test_cap_sweep_can_use_opportunity_replay_gate() -> None:
    base_summary = pd.DataFrame(
        [
            {
                "arm": "P0_static_priority",
                "trade_count": 10,
                "net_pnl": 100.0,
                "full_sl_rate": 0.20,
                "timeout_rate": 0.10,
            }
        ]
    )
    candidate_summary = pd.DataFrame(
        [
            {
                "arm": "candidate",
                "trade_count": 10,
                "net_pnl": 112.0,
                "full_sl_rate": 0.20,
                "timeout_rate": 0.106,
            }
        ]
    )
    common_ts = pd.date_range("2026-06-01T00:00:00Z", periods=19, freq="h")
    base_accepted = pd.DataFrame(
        {
            "arm": ["P0_static_priority"] * 20,
            "timestamp": list(common_ts) + [pd.Timestamp("2026-06-02T00:00:00Z")],
            "symbol": [f"S{i}" for i in range(19)] + ["REMOVED"],
            "strategy_id": ["s"] * 20,
            "side": ["short"] * 20,
            "head": ["short_asset"] * 20,
            "net_return": [0.01] * 19 + [-0.01],
            "_net_return": [0.01] * 19 + [-0.01],
            "net_pnl": [1.0] * 19 + [-1.0],
        }
    )
    candidate_accepted = pd.DataFrame(
        {
            "arm": ["candidate"] * 20,
            "timestamp": list(common_ts) + [pd.Timestamp("2026-06-02T01:00:00Z")],
            "symbol": [f"S{i}" for i in range(19)] + ["ENTRANT"],
            "strategy_id": ["s"] * 20,
            "side": ["short"] * 20,
            "head": ["short_asset"] * 19 + ["short_boll"],
            "net_return": [0.01] * 19 + [0.08],
            "_net_return": [0.01] * 19 + [0.08],
            "net_pnl": [1.0] * 19 + [8.0],
        }
    )

    defensive = replay_selection_metrics(
        arm="candidate",
        candidate_summary=candidate_summary,
        candidate_accepted=candidate_accepted,
        base_summary=base_summary,
        base_accepted=base_accepted,
    )
    opportunity = replay_selection_metrics(
        arm="candidate",
        candidate_summary=candidate_summary,
        candidate_accepted=candidate_accepted,
        base_summary=base_summary,
        base_accepted=base_accepted,
        gate_mode="opportunity",
    )

    assert defensive["replay_selection_gate_passed"] is False
    assert opportunity["replay_selection_gate_passed"] is True
