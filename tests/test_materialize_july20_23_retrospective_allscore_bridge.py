from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_july20_23_retrospective_allscore_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("july_retro_bridge", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sources(rows: int = 8) -> tuple[pd.DataFrame, ...]:
    labels: list[dict[str, object]] = []
    packb: list[dict[str, object]] = []
    preentry: list[dict[str, object]] = []
    scored: list[dict[str, object]] = []
    direct: list[dict[str, object]] = []
    adapter: list[dict[str, object]] = []
    for index in range(rows):
        signal = pd.Timestamp("2026-07-20T00:00:00Z") + pd.Timedelta(hours=index)
        decision = signal + pd.Timedelta(hours=1)
        end = decision + pd.Timedelta(hours=12)
        side = "long" if index % 2 == 0 else "short"
        symbol = f"A{index}/USD:USD"
        candidate_id = f"{symbol}|{signal.strftime('%Y-%m-%dT%H:%M:%SZ')}|1h|{side}"
        identity = {
            "candidate_id": candidate_id,
            "__ts__": signal,
            "__symbol__": symbol,
            "side_name": side,
        }
        alpha = index / 100.0
        labels.append(
            {
                **identity,
                "execution_decision_utc": decision,
                "execution_label_end_utc": end,
                "execution_gross_ev_12h": alpha,
                "execution_cost_return": 0.01,
                "execution_net_ev_12h": alpha - 0.01,
                "execution_mfe_return_12h": alpha + 0.02,
                "execution_mae_return_12h": -0.02,
                "execution_exit_reason": "timeout",
                "execution_exit_hour": 12.0,
            }
        )
        packb.append(
            {
                **identity,
                "base_oof_score": alpha,
                "base_alpha_ev": alpha / 2,
                "residual_delta_ev": alpha / 3,
                "existing_alpha_ev": alpha / 4,
                "base_available_at": decision,
                "residual_available_at": decision,
            }
        )
        preentry.append(
            {
                **identity,
                "execution_decision_utc": decision,
                "existing_alpha_ev": alpha / 4,
                "pred_peak_MFE_12h_ATR": alpha,
                "oof_clean_favorable_probability": 0.5 + alpha,
                "feature_available_at": decision,
                "base_available_at": decision,
                "residual_available_at": decision,
                "peak_mfe_available_at": decision,
                "path_catboost_available_at": decision,
                "clean_probability_available_at": decision,
            }
        )
        scored.append(
            {
                **identity,
                "execution_decision_utc": decision,
                "existing_alpha_ev": alpha / 4,
                "final_direct_net_raw": alpha,
                "final_capture_probability": 0.5 + alpha,
                "frozen_margin_capture_interaction_raw": alpha / 5,
                "direct_ev_available_at": decision,
                "capture_probability_available_at": decision,
                "mapping_available_at": decision,
            }
        )
        direct.append({**identity, "q25_net_bps": index, "q50_net_bps": index + 1})
        adapter.append(
            {
                **identity,
                "q25_net_bps": index + 2,
                "q50_net_bps": index + 3,
                "base_oof_score": alpha,
                "score_parent_bps": index + 4,
                "score_adapter_bps": index + 5,
                "score_reliability_bps": index + 6,
                "score_adapter_reliability_bps": index + 7,
            }
        )
    return tuple(
        pd.DataFrame(value)
        for value in (labels, packb, preentry, scored, direct, adapter)
    )


def test_build_exact_bridge_and_stamps_evidence_status() -> None:
    frame, registry = MODULE.build_bridge(*_sources(), expected_rows=8)
    assert len(frame) == 8
    assert len(registry) == 17
    assert frame.evidence_status.eq(
        "retrospective_nonpromotable_not_oos"
    ).all()
    assert not any("mapped" in column.lower() for column in frame)
    assert set(MODULE.score_columns(frame)) == set(registry.score)


def test_build_rejects_identity_gap_or_late_availability() -> None:
    labels, packb, preentry, scored, direct, adapter = _sources()
    direct.loc[0, "__symbol__"] = "WRONG/USD:USD"
    with pytest.raises(ValueError, match="identity coverage"):
        MODULE.build_bridge(
            labels, packb, preentry, scored, direct, adapter, expected_rows=8
        )

    labels, packb, preentry, scored, direct, adapter = _sources()
    scored.loc[0, "direct_ev_available_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="not available at decision"):
        MODULE.build_bridge(
            labels, packb, preentry, scored, direct, adapter, expected_rows=8
        )


def test_build_rejects_wrong_horizon_or_score_lineage() -> None:
    labels, packb, preentry, scored, direct, adapter = _sources()
    labels.loc[0, "execution_label_end_utc"] += pd.Timedelta(hours=12)
    with pytest.raises(ValueError, match="exact 12h"):
        MODULE.build_bridge(
            labels, packb, preentry, scored, direct, adapter, expected_rows=8
        )

    labels, packb, preentry, scored, direct, adapter = _sources()
    adapter.loc[0, "base_oof_score"] += 0.1
    with pytest.raises(ValueError, match="adapter base score"):
        MODULE.build_bridge(
            labels, packb, preentry, scored, direct, adapter, expected_rows=8
        )
