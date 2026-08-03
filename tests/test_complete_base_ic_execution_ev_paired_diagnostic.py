from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "complete_base_ic_execution_ev_paired_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("paired_completion", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _panel() -> pd.DataFrame:
    rows = []
    for month_index, month in enumerate(("2025-02", "2025-03")):
        for index in range(200):
            side = "long" if index % 2 == 0 else "short"
            opportunity = index % 3 != 0
            exit_class = MODULE.EXIT_CLASSES[index % len(MODULE.EXIT_CLASSES)]
            gross = (index / 10_000) - 0.005 - month_index * 0.001
            cost = 0.01
            rows.append(
                {
                    "candidate_id": f"{month}-{index}",
                    "candidate_month": month,
                    "side_name": side,
                    "__symbol__": f"asset-{index % 5}",
                    "__ts__": pd.Timestamp(f"{month}-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                    "base_oof_score": index / 200,
                    "base_group_rows_timestamp_global": 20 + index % 120,
                    "__first_touch_target_soft__": index / 200,
                    "execution_mfe_return_12h": gross + 0.02,
                    "execution_mae_return_12h": -0.01 - index / 100_000,
                    "execution_gross_ev_12h": gross,
                    "execution_cost_return": cost,
                    "execution_net_ev_12h": gross - cost,
                    "opportunity_gross_above_cost_0bps": opportunity,
                    "execution_exit_minute": 10 + index,
                    "execution_exit_class": exit_class,
                }
            )
    return pd.DataFrame(rows)


def test_bridge_emits_all_targets_in_ic_and_deciles() -> None:
    metrics, deciles = MODULE.bridge_tables(_panel())
    assert set(metrics.target) == set(MODULE.BRIDGE_TARGETS)
    for target in MODULE.BRIDGE_TARGETS:
        assert f"{target}__mean" in deciles
        assert f"{target}__rank_ic" in deciles


def test_stable_top_is_global_and_candidate_id_deterministic() -> None:
    frame = _panel().iloc[:20].copy()
    frame["base_oof_score"] = 1.0
    selected = MODULE.stable_top(frame, 0.10)
    expected = sorted(frame.candidate_id.astype(str))[:2]
    assert selected.candidate_id.astype(str).tolist() == expected


def test_fixed_time_path_is_side_aware() -> None:
    payload = json.dumps(
        {
            "close": np.linspace(100.0, 110.0, 720).tolist(),
        }
    )
    long = MODULE._decode_fixed_returns(payload, 100.0, "long")
    short = MODULE._decode_fixed_returns(payload, 100.0, "short")
    assert long["fixed_12h_gross"] == pytest.approx(0.10)
    assert short["fixed_12h_gross"] == pytest.approx(-0.10)


def test_exit_counterfactual_preserves_frozen_ids() -> None:
    selected = _panel().groupby("candidate_month", group_keys=False).head(3)
    paths = pd.DataFrame(
        {
            "candidate_id": selected.candidate_id,
            **{
                f"fixed_{hours}h_gross": 0.02
                for hours in (1, 2, 4, 8, 12)
            },
        }
    )
    _, audit = MODULE.exit_counterfactuals(selected, paths)
    assert set(audit.candidate_id) == set(selected.candidate_id)
    assert audit["deployed_net"].equals(
        selected.set_index("candidate_id")
        .loc[audit.candidate_id, "execution_net_ev_12h"]
        .reset_index(drop=True)
    )


def test_joint_reweighting_has_zero_delta_for_identical_months() -> None:
    panel = _panel()
    first = panel.loc[panel.candidate_month.eq("2025-02")].copy()
    second = first.copy()
    second["candidate_month"] = "2025-03"
    second["candidate_id"] = "copy-" + second.candidate_id.astype(str)
    result = MODULE.joint_composition_reweight(pd.concat([first, second]))
    assert result.iloc[0]["composition_effect_bps"] == pytest.approx(0.0)
    assert result.iloc[0]["within_cell_payoff_effect_bps"] == pytest.approx(0.0)


def test_unified_attribution_reconciles_realized_delta() -> None:
    result = MODULE.unified_shapley_attribution(_panel())
    grouped = result.groupby(["from_month", "to_month"])
    for _, local in grouped:
        assert local.contribution_bps.sum() == pytest.approx(
            local.actual_delta_bps.iloc[0], abs=1e-8
        )
