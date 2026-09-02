from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import scripts.materialize_strict_r3_exact_1m_policy_hpo_dataset as exact_materializer
from extreme_price_movements.exact_1m_policy_contract import (
    Exact1mExecutionContract,
    Exact1mPolicyParams,
    simulate_exact_1m_parent_policy,
)
from extreme_price_movements.data_store import _execution_1m_part_bounds_seconds
from extreme_price_movements.strict_r3_shadow_portfolio import (
    ShadowOpenPosition,
    ShadowPortfolioState,
    advance_shadow_state,
)
from scripts.materialize_strict_r3_exact_1m_policy_hpo_dataset import (
    _explicit_candidate_population,
    _sha256,
    _verify_download_receipts,
)


def _paths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.full((1, 720), 100.0)
    low = np.full((1, 720), 100.0)
    close = np.full((1, 720), 100.0)
    return high, low, close


def test_trailing_is_armed_from_prior_completed_mfe_and_fills_at_close() -> None:
    high, low, close = _paths()
    # Bar 0 reaches activation and also has a low through the future trail.
    # It must not exit: that MFE only becomes actionable on bar 1.
    high[0, 0] = 102.0
    low[0, 0] = 100.0
    close[0, 0] = 101.5
    low[0, 1] = 101.0
    close[0, 1] = 100.8
    outcome = simulate_exact_1m_parent_policy(
        entry=np.array([100.0]), atr=np.array([1.0]), highs=high, lows=low, closes=close,
        entry_timestamps=pd.DatetimeIndex(["2026-01-01T00:05:00Z"]),
        params=Exact1mPolicyParams(sl_mult=10.0, trailing_activation_mult=1.0, fixed_trailing_gap_mult=0.5),
        contract=Exact1mExecutionContract(), median_atr_fraction=0.01,
    )
    assert outcome["exit_reason"][0] == "trailing"
    assert outcome["exit_bar"][0] == 1
    assert outcome["exit_price"][0] == 100.8
    assert pd.Timestamp(outcome["exit_timestamp"][0], tz="UTC") == pd.Timestamp("2026-01-01T00:07:00Z")


def test_cost_is_applied_once_and_invalid_paths_stay_invalid() -> None:
    high, low, close = _paths()
    high[0, -1] = 101.0
    close[0, -1] = 101.0
    outcome = simulate_exact_1m_parent_policy(
        entry=np.array([100.0]), atr=np.array([1.0]), highs=high, lows=low, closes=close,
        entry_timestamps=pd.DatetimeIndex(["2026-01-01T00:05:00Z"]),
        params=Exact1mPolicyParams(), contract=Exact1mExecutionContract(), median_atr_fraction=0.01,
    )
    assert outcome["path_valid"][0]
    assert np.isclose(outcome["gross_bps"][0] - outcome["net_bps"][0], 100.0)
    low[0, 10] = np.nan
    invalid = simulate_exact_1m_parent_policy(
        entry=np.array([100.0]), atr=np.array([1.0]), highs=high, lows=low, closes=close,
        entry_timestamps=pd.DatetimeIndex(["2026-01-01T00:05:00Z"]),
        params=Exact1mPolicyParams(), contract=Exact1mExecutionContract(), median_atr_fraction=0.01,
    )
    assert not invalid["path_valid"][0]
    assert invalid["exit_reason"][0] == "invalid_exact_1m_path"
    assert np.isnat(invalid["exit_timestamp"][0])


def test_exact_contract_forbids_legacy_bar_unit_parameters() -> None:
    try:
        Exact1mPolicyParams.from_mapping({"trailing_activation_decay_half_life_bars": 4})
    except ValueError as exc:
        assert "time units" in str(exc)
    else:
        raise AssertionError("legacy bar unit must be rejected")


def test_optional_policy_parameters_round_trip_from_dataframe_nan() -> None:
    params = Exact1mPolicyParams.from_mapping({
        "sl_atr_power": np.nan,
        "tp_atr_multiplier": np.nan,
        "capital_protect_lock_frac": np.nan,
        "adverse_exit_theta": np.nan,
    })
    assert params.sl_atr_power is None
    assert params.tp_atr_multiplier is None
    assert params.capital_protect_lock_frac is None
    assert params.adverse_exit_theta is None


def test_wilder14_source_lookback_covers_delayed_entry_hourly_phase() -> None:
    """A xx:05 entry needs a 101-hour source window for 100 whole bins."""
    entry = pd.Timestamp("2024-02-10T00:05:00Z")

    def bars(hours: int) -> pd.DataFrame:
        index = pd.date_range(
            entry - pd.Timedelta(hours=hours),
            periods=hours * 60,
            freq="min",
            tz="UTC",
        )
        return pd.DataFrame(
            {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
            index=index,
        )

    too_short = exact_materializer._causal_atr(bars(100))
    enough = exact_materializer._causal_atr(bars(101))
    assert pd.isna(too_short.reindex([entry], method="ffill").iloc[0])
    assert np.isfinite(enough.reindex([entry], method="ffill").iloc[0])
    assert exact_materializer.ATR_SOURCE_LOOKBACK_HOURS == 101


def test_exact_1m_trailing_trigger_matches_live_state_machine() -> None:
    """HPO uses the same trigger/bar timestamp as the live minute monitor.

    The live monitor submits a reduce-only market order *after* the threshold
    bar has completed.  The offline contract therefore uses that completed
    close as the historical fill proxy, while this test proves the underlying
    trigger itself is identical to the proven state machine.
    """
    high, low, close = _paths()
    high[0, 0] = 102.0
    low[0, 1] = 101.0
    close[0, 1] = 100.8
    entry_ts = pd.Timestamp("2026-01-01T00:05:00Z")
    params = Exact1mPolicyParams(
        sl_mult=10.0,
        trailing_activation_mult=1.0,
        fixed_trailing_gap_mult=0.5,
    )
    exact = simulate_exact_1m_parent_policy(
        entry=np.array([100.0]), atr=np.array([1.0]), highs=high, lows=low,
        closes=close, entry_timestamps=pd.DatetimeIndex([entry_ts]), params=params,
        contract=Exact1mExecutionContract(), median_atr_fraction=0.01,
    )
    bars = pd.DataFrame(
        {
            "high": high[0, :2], "low": low[0, :2], "close": close[0, :2],
        },
        index=pd.date_range(entry_ts, periods=2, freq="min"),
    )
    position = ShadowOpenPosition(
        symbol="TEST/USD:USD", side="long", gross_notional=100.0,
        effective_leverage=1.0, candidate_id="candidate", entry_ts=entry_ts,
        entry_price=100.0, atr=1.0, next_bar_ts=entry_ts,
        timeout_ts=entry_ts + pd.Timedelta(hours=12),
    )
    _, live = advance_shadow_state(
        ShadowPortfolioState(as_of_ts=entry_ts, wallet=1_000.0, open_positions=(position,)),
        decision_ts=entry_ts + pd.Timedelta(minutes=2),
        bars_by_symbol={position.symbol: bars}, stop_loss_atr=params.sl_mult,
        trailing_activation_atr=params.trailing_activation_mult,
        trailing_giveback_atr=params.fixed_trailing_gap_mult,
        bar_minutes=1,
    )
    assert live.iloc[0]["exit_reason"] == exact["exit_reason"][0] == "trailing"
    assert pd.Timestamp(live.iloc[0]["exit_ts"]) == pd.Timestamp(exact["exit_timestamp"][0], tz="UTC")
    # Shadow bookkeeping uses the threshold; the HPO uses the completed bar
    # close because that is when the actual reduce-only market order is sent.
    assert live.iloc[0]["exit_price"] == 101.5
    assert exact["exit_price"][0] == 100.8


def test_minute_source_receipts_must_bind_to_exact_candidate_request(tmp_path) -> None:
    contract = Exact1mExecutionContract()
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    population = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "timestamp": pd.to_datetime(["2024-02-01T00:00:00Z", "2024-02-01T01:00:00Z"]),
        "symbol": ["A/USD:USD", "B/USD:USD"],
    })
    population["entry_ts"] = population["timestamp"] + pd.Timedelta(minutes=5)
    request_path = request_dir / "candidate_download_request.parquet"
    population.assign(score=[0.9, 0.8]).to_parquet(request_path, index=False)
    request_manifest = request_dir / "candidate_download_request.json"
    request_manifest.write_text(json.dumps({"contract_hash": contract.hash}))
    request_sha, stage_sha = _sha256(request_path), _sha256(request_manifest)
    for partition_id in range(16):
        (request_dir / f"download_partition_{partition_id}.json").write_text(json.dumps({
            "partition_count": 16, "partition_id": partition_id,
            "candidate_sha256": request_sha,
            "stage_manifest": {"sha256": stage_sha},
            "product_mapping_contract": "frozen-symbol-contract",
            "summary": {
                "failed_symbols": 0, "incomplete_symbols": 0,
                "required_minutes": 10, "covered_minutes": 10, "fetched_rows": 10,
            },
            "results": [{"symbol": "A/USD:USD", "status": "ok", "coverage_after": 1.0}],
        }))
    audit = _verify_download_receipts(request_dir, population, contract)
    assert audit["summary"]["covered_minutes"] == 160
    assert len(audit["receipts"]) == 16


def test_terminal_incomplete_symbol_receipts_can_be_audited_without_routing_by_path(tmp_path) -> None:
    """A source gap invalidates only its later joined paths, never the route."""
    contract = Exact1mExecutionContract()
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    population = pd.DataFrame({
        "candidate_id": ["a"],
        "timestamp": pd.to_datetime(["2024-02-01T00:00:00Z"]),
        "symbol": ["A/USD:USD"],
    })
    population["entry_ts"] = population["timestamp"] + pd.Timedelta(minutes=5)
    request_path = request_dir / "candidate_download_request.parquet"
    population.assign(score=[0.9]).to_parquet(request_path, index=False)
    manifest_path = request_dir / "candidate_download_request.json"
    manifest_path.write_text(json.dumps({"contract_hash": contract.hash}))
    request_sha, stage_sha = _sha256(request_path), _sha256(manifest_path)
    for partition_id in range(16):
        incomplete = partition_id == 3
        (request_dir / f"download_partition_{partition_id}.json").write_text(json.dumps({
            "partition_count": 16,
            "partition_id": partition_id,
            "candidate_sha256": request_sha,
            "stage_manifest": {"sha256": stage_sha},
            "product_mapping_contract": "frozen-symbol-contract",
            "summary": {
                "failed_symbols": 0,
                "incomplete_symbols": int(incomplete),
                "required_minutes": 10,
                "covered_minutes": 8 if incomplete else 10,
                "fetched_rows": 10,
            },
            "results": [{
                "symbol": "A/USD:USD",
                "status": "incomplete" if incomplete else "ok",
                "coverage_after": 0.8 if incomplete else 1.0,
                "required_minutes": 10,
                "covered_after": 8 if incomplete else 10,
            }],
        }))
    try:
        _verify_download_receipts(request_dir, population, contract)
    except AssertionError as exc:
        assert "not complete" in str(exc)
    else:
        raise AssertionError("strict receipt validation accepted an incomplete source")
    audit = _verify_download_receipts(
        request_dir, population, contract, allow_incomplete_symbols=True,
    )
    assert audit["allows_incomplete_symbols"] is True
    assert audit["summary"]["incomplete_symbols"] == 1
    assert audit["incomplete_symbol_paths"] == [{
        "receipt": "download_partition_3.json",
        "symbol": "A/USD:USD",
        "coverage_after": 0.8,
        "required_minutes": 10,
        "covered_after": 8,
    }]


def test_explicit_target_free_candidate_input_is_identity_bound_and_excludes_outcomes(tmp_path, monkeypatch) -> None:
    """The dual-MC1 request can bypass the legacy ledger without label routing."""
    contract = Exact1mExecutionContract(entry_delay_minutes=0)
    request_dir = tmp_path / "dual_request"
    request_dir.mkdir()
    timestamp = pd.Timestamp("2024-02-10T00:00:00Z")
    request = pd.DataFrame({
        "candidate_id": ["dual-a"],
        "timestamp": [timestamp],
        "symbol": ["A/USD:USD"],
        "side_name": ["long"],
        "entry_ts": [timestamp],
        "priority_bps": [87.5],
    })
    request_path = request_dir / "candidate_download_request.parquet"
    request.to_parquet(request_path, index=False)
    manifest_path = request_dir / "candidate_download_request.json"
    manifest = {
        "schema": "strict_r3_exact_1m_dual30_bcf_priority_candidate_request_v1",
        "target_free": True,
        "selection_inputs": ["bcf_mc1_expected_bps", "current_v5_mc1_expected_bps"],
        "forbidden_selection_inputs": ["policy_path_valid", "policy_net_bps", "outcome", "label"],
        "candidate_sha256": _sha256(request_path),
        "contract_hash": contract.hash,
        "rows": 1,
    }
    manifest_path.write_text(json.dumps(manifest))
    population, source = _explicit_candidate_population(request_path, manifest_path, contract)
    assert population.loc[0, "score"] == 87.5
    assert source["target_free"] is True
    assert source["score_column"] == "priority_bps"

    # A labelled/policy-result column is rejected from the source schema before
    # any rows are read.  This protects the candidate route from a convenient
    # but invalid outcome-qualified panel.
    contaminated = request.assign(policy_net_bps=999.0)
    contaminated_path = request_dir / "contaminated.parquet"
    contaminated.to_parquet(contaminated_path, index=False)
    contaminated_manifest = dict(manifest, candidate_sha256=_sha256(contaminated_path))
    contaminated_manifest_path = request_dir / "contaminated.json"
    contaminated_manifest_path.write_text(json.dumps(contaminated_manifest))
    try:
        _explicit_candidate_population(contaminated_path, contaminated_manifest_path, contract)
    except AssertionError as exc:
        assert "outcome-derived" in str(exc)
    else:
        raise AssertionError("outcome-derived candidate source was accepted")

    # The full materializer must consume the explicit source without touching
    # its legacy ledger selection.  Use complete synthetic 1m source bars only
    # after the source identity has been sealed above.
    request_sha, manifest_sha = _sha256(request_path), _sha256(manifest_path)
    for partition_id in range(16):
        (request_dir / f"download_partition_{partition_id}.json").write_text(json.dumps({
            "partition_count": 16, "partition_id": partition_id,
            "candidate_sha256": request_sha,
            "stage_manifest": {"sha256": manifest_sha},
            "product_mapping_contract": "test-frozen-symbol-contract",
            "summary": {
                "failed_symbols": 0, "incomplete_symbols": 0,
                "required_minutes": 1, "covered_minutes": 1, "fetched_rows": 0,
            },
            "results": [{"symbol": "A/USD:USD", "status": "ok", "coverage_after": 1.0}],
        }))
    bars = pd.DataFrame(
        {
            "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
        },
        index=pd.date_range(timestamp - pd.Timedelta(hours=101), periods=6_780, freq="min", tz="UTC"),
    )

    class _FakeStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def load(self, _symbol, *, start_ts, end_ts, **_kwargs):
            start = pd.Timestamp(start_ts)
            end = pd.Timestamp(end_ts)
            return bars.loc[(bars.index >= start) & (bars.index <= end)].copy()

    monkeypatch.setattr(exact_materializer, "PartitionedOHLCVStore", _FakeStore)
    output = exact_materializer.materialize(
        argparse.Namespace(
            out_dir=tmp_path / "dataset",
            overwrite=False,
            entry_delay_minutes=0,
            start="2024-02-01T00:00:00Z",
            end="2024-03-01T00:00:00Z",
            candidate_input=request_path,
            candidate_manifest=manifest_path,
            request_only=False,
            download_request_dir=request_dir,
            ledger=tmp_path / "must_not_be_read.parquet",
            retained_fraction=0.05,
            cap_per_month=1,
            minute_root=tmp_path / "unused",
        )
    )
    dataset_manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert dataset_manifest["candidate_source"]["mode"] == "explicit_target_free_candidate_input"
    assert dataset_manifest["candidate_source"]["target_free"] is True
    training = pd.read_parquet(output / "training_rows.parquet")
    assert training.loc[0, "score"] == 87.5


def test_legacy_timestamp_only_execution_parts_are_range_prunable(tmp_path) -> None:
    path = tmp_path / "part-1686506400-1686509940.parquet"
    assert _execution_1m_part_bounds_seconds(path) == (1686506400, 1686509940)


def test_legacy_compact_execution_parts_are_range_prunable(tmp_path) -> None:
    path = tmp_path / "compact-1678525200-1683385140.parquet"
    assert _execution_1m_part_bounds_seconds(path) == (1678525200, 1683385140)
