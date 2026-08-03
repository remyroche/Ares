from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_stage_d_action_counterfactuals import ContractError, _load_candidates, adverse_exit_fill_proxy, build_counterfactual
from scripts.materialize_historical_exact_h12_alignment_sidecar import COST_MODEL_ID, EXECUTION_POLICY_ID
from scripts.materialize_historical_exact_h12_postcost_events import TARGET_ID as POSTCOST_TARGET_ID


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "data_perp/artifacts/stage_d_action_counterfactuals_20260731_v2"
V2_COUNTERFACTUALS = V2_ROOT / "stage_d_action_counterfactuals.parquet"
V2_EXCLUSIONS = V2_ROOT / "stage_d_action_exclusions.parquet"
EVENTS = ROOT / "data_perp/artifacts/historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
ALIGNMENT = ROOT / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"


@pytest.fixture(scope="module")
def v2_artifact() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The immutable D0 v2 pair and its explicit non-actionable tail."""
    return pd.read_parquet(V2_COUNTERFACTUALS), pd.read_parquet(V2_EXCLUSIONS)


def _repository_adverse_exit_proxy() -> Any:
    """Load the exact optimiser function body without importing its side effects."""
    source_path = Path(__file__).resolve().parents[1] / "extreme_price_movements/simple_policy_optimiser.py"
    module = ast.parse(source_path.read_text())
    node = next(item for item in module.body if isinstance(item, ast.FunctionDef) and item.name == "_adverse_exit_fill_proxy_array")
    isolated = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "np": np,
        "Optional": Optional,
        "POLICY_STOP_EXIT_BASE_GAP_BPS": 15.0,
        "POLICY_STOP_EXIT_ALPHA_THROUGH": 0.0,
        "POLICY_STOP_EXIT_MAX_GAP_BPS": 75.0,
        "Any": Any,
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace["_adverse_exit_fill_proxy_array"]


def _path(start: pd.Timestamp) -> str:
    ts = start.value + np.arange(720, dtype=np.int64) * pd.Timedelta(minutes=1).value
    price = 100.0 + np.arange(720, dtype=float) * 0.01
    return json.dumps({"timestamp": ts.tolist(), "open": price.tolist()})


def _candidate() -> dict[str, object]:
    decision = pd.Timestamp("2024-04-01T00:00:00Z")
    return {"candidate_id": "c1", "side": "long", "decision_ts": decision, "entry_ts": decision, "label_end_ts": decision + pd.Timedelta(hours=12), "label_available_ts": decision + pd.Timedelta(hours=12), "execution_policy_id": "historical_current_frozen_spread_counterfactual_h12_v1", "cost_model_id": "current_frozen_spread_counterfactual_row_cost_v1", "execution_entry_price": 100.0, "row_cost_bps": 100.0, "exact_h12_gross_bps": 150.0, "exact_h12_net_bps": 50.0, "exit_half_spread_bps": 5.0}


def test_clear_bar_high_low_are_not_observable_until_its_close() -> None:
    c = _candidate(); row = build_counterfactual(candidate=c, favorable_minute=2, raw_path=_path(c["decision_ts"]), path_source_file="paths")
    assert row["clear_event_bar_open_ts"] == c["decision_ts"] + pd.Timedelta(minutes=2)
    assert row["first_clear_ts"] == c["decision_ts"] + pd.Timedelta(minutes=3)
    assert row["action_decision_ts"] == c["decision_ts"] + pd.Timedelta(minutes=3)
    assert row["action_decision_ts"] < row["action_execution_ts"]
    assert row["action_execution_ts"] == c["decision_ts"] + pd.Timedelta(minutes=4)
    # If the clear bar's high/low were visible at bar open, this would be one
    # minute earlier.  The v2 contract intentionally forbids that look-ahead.
    assert row["clear_event_bar_open_ts"] < row["action_decision_ts"]


def test_both_arms_are_cost_once_and_delta_is_paired() -> None:
    c = _candidate(); row = build_counterfactual(candidate=c, favorable_minute=2, raw_path=_path(c["decision_ts"]), path_source_file="paths")
    assert row["net_continue_gross_bps"] - row["net_continue_cost_bps"] == pytest.approx(row["net_continue_bps"])
    assert row["net_exit_now_gross_bps"] - row["net_exit_now_cost_bps"] == pytest.approx(row["net_exit_now_bps"])
    assert row["delta_continue_bps"] == pytest.approx(row["net_continue_bps"] - row["net_exit_now_bps"])


def test_continue_is_exact_frozen_alignment_outcome() -> None:
    c = _candidate(); row = build_counterfactual(candidate=c, favorable_minute=2, raw_path=_path(c["decision_ts"]), path_source_file="paths")
    assert row["net_continue_bps"] == c["exact_h12_net_bps"]
    assert row["execution_policy_id"] == c["execution_policy_id"]


@pytest.mark.parametrize("minute", [718, 719])
def test_tail_clear_without_strictly_later_execution_open_is_not_actionable(minute: int) -> None:
    c = _candidate()
    with pytest.raises(ContractError, match="strictly later execution open"):
        build_counterfactual(candidate=c, favorable_minute=minute, raw_path=_path(c["decision_ts"]), path_source_file="paths")


def test_future_path_or_timing_drift_is_rejected() -> None:
    c = _candidate(); c["label_available_ts"] = c["label_end_ts"] + pd.Timedelta(minutes=1)
    with pytest.raises(ContractError, match="timing"):
        build_counterfactual(candidate=c, favorable_minute=2, raw_path=_path(c["decision_ts"]), path_source_file="paths")


def test_population_and_tail_exclusion_identity_are_exact(tmp_path: Path) -> None:
    candidates = []
    events = []
    for minute in (717, 718, 719):
        candidate = _candidate() | {"candidate_id": f"c{minute}", "execution_policy_id": EXECUTION_POLICY_ID, "cost_model_id": COST_MODEL_ID}
        candidates.append(candidate)
        events.append({
            "candidate_id": f"c{minute}", "side": "long", "postcost_target_id": POSTCOST_TARGET_ID,
            "postcost_h0_event": "clear_cost_first", "postcost_h0_favorable_minute": minute,
            "execution_policy_id": EXECUTION_POLICY_ID, "cost_model_id": COST_MODEL_ID,
        })
    alignment_path, events_path = tmp_path / "alignment.parquet", tmp_path / "events.parquet"
    pd.DataFrame(candidates).to_parquet(alignment_path, index=False)
    pd.DataFrame(events).to_parquet(events_path, index=False)
    eligible, excluded = _load_candidates(alignment_path, events_path)
    assert eligible.candidate_id.tolist() == ["c717"]
    assert excluded.candidate_id.tolist() == ["c718", "c719"]
    assert excluded.exclusion_reason.tolist() == [
        "no_immutable_strictly_later_open_after_completed_clear",
        "clear_bar_completion_and_strictly_later_open_not_immutable",
    ]


@pytest.mark.parametrize("side,price,spread", [
    (1.0, 100.0, 0.0),
    (1.0, 17.123456789, 5.0),
    (-1.0, 17.123456789, 5.0),
    (1.0, 0.0001234567, 250.0),
    (-1.0, 105_432.125, 1_000.0),
])
def test_adverse_exit_fill_matches_repository_float32_oracle(side: float, price: float, spread: float) -> None:
    reference = _repository_adverse_exit_proxy()
    expected = reference(
        side=np.asarray([side], dtype=np.float32),
        exit_px=np.asarray([price], dtype=np.float64),
        trigger="close",
        quote_half_spread_bps=np.asarray([spread], dtype=np.float64),
    )[0]
    actual = adverse_exit_fill_proxy(side=side, exit_price=price, quote_half_spread_bps=spread)
    assert np.float32(actual).view(np.uint32) == np.float32(expected).view(np.uint32)


# Exact Stage-D roadmap D0 names.  These are artifact-level integration
# assertions; formula and boundary behavior remains covered by the synthetic
# tests above.
def test_action_population_is_exact_clear_first_population(v2_artifact: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, excluded = v2_artifact
    events = pd.read_parquet(EVENTS, columns=["candidate_id", "postcost_h0_event", "postcost_h0_favorable_minute"])
    clear = events.loc[events.postcost_h0_event.eq("clear_cost_first")]
    assert set(rows.candidate_id).isdisjoint(set(excluded.candidate_id))
    assert set(rows.candidate_id) | set(excluded.candidate_id) == set(clear.candidate_id)
    assert set(rows.candidate_id) == set(clear.loc[clear.postcost_h0_favorable_minute.lt(718), "candidate_id"])
    assert set(excluded.candidate_id) == set(clear.loc[clear.postcost_h0_favorable_minute.ge(718), "candidate_id"])


def test_first_clear_timestamp_matches_frozen_label_pack(v2_artifact: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, _ = v2_artifact
    events = pd.read_parquet(EVENTS, columns=["candidate_id", "postcost_h0_favorable_minute"])
    alignment = pd.read_parquet(ALIGNMENT, columns=["candidate_id", "decision_ts"])
    joined = rows[["candidate_id", "clear_event_bar_open_ts", "first_clear_ts", "action_decision_ts"]].merge(events, on="candidate_id", validate="one_to_one").merge(alignment, on="candidate_id", validate="one_to_one")
    expected_open = joined.decision_ts + pd.to_timedelta(joined.postcost_h0_favorable_minute, unit="m")
    expected_observable = expected_open + pd.Timedelta(minutes=1)
    # Arrow may restore materialized timestamps at microsecond precision;
    # compare UTC instants rather than pandas' storage-resolution metadata.
    assert np.array_equal(joined.clear_event_bar_open_ts.astype(str), expected_open.astype(str))
    assert np.array_equal(joined.first_clear_ts.astype(str), expected_observable.astype(str))
    assert np.array_equal(joined.action_decision_ts.astype(str), expected_observable.astype(str))


def test_action_decision_precedes_action_execution(v2_artifact: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, _ = v2_artifact
    assert rows.action_decision_ts.lt(rows.action_execution_ts).all()
    assert (rows.action_execution_ts - rows.clear_event_bar_open_ts).eq(pd.Timedelta(minutes=2)).all()


def test_exit_now_counterfactual_cost_applied_once(v2_artifact: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, _ = v2_artifact
    assert np.allclose(rows.net_exit_now_gross_bps - rows.net_exit_now_cost_bps, rows.net_exit_now_bps, atol=1e-6)
    assert rows.net_exit_now_cost_bps.eq(rows.known_row_cost_bps).all()


def test_continue_counterfactual_matches_frozen_policy(v2_artifact: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, _ = v2_artifact
    frozen = pd.read_parquet(ALIGNMENT, columns=["candidate_id", "exact_h12_gross_bps", "exact_h12_net_bps", "row_cost_bps"])
    joined = rows.merge(frozen, on="candidate_id", validate="one_to_one", suffixes=("", "_frozen"))
    assert np.allclose(joined.net_continue_gross_bps, joined.exact_h12_gross_bps, atol=1e-6)
    assert np.allclose(joined.net_continue_bps, joined.exact_h12_net_bps, atol=1e-6)
    assert np.allclose(joined.net_continue_cost_bps, joined.row_cost_bps, atol=1e-6)


def test_delta_equals_continue_minus_exit(v2_artifact: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, _ = v2_artifact
    assert np.allclose(rows.delta_continue_bps, rows.net_continue_bps - rows.net_exit_now_bps, atol=1e-6)


def test_action_arms_use_identical_candidate_ids(v2_artifact: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, _ = v2_artifact
    assert rows.candidate_id.notna().all() and not rows.candidate_id.duplicated().any()
    assert rows[["net_exit_now_bps", "net_continue_bps", "delta_continue_bps"]].notna().all().all()
    assert np.allclose(rows.delta_continue_bps, rows.net_continue_bps - rows.net_exit_now_bps, atol=1e-6)
