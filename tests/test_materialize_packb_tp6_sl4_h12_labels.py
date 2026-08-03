from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.materialize_packb_tp6_sl4_h12_labels import (
    COST_BPS,
    _causal_hourly_atr_from_minute,
    _label_candidates_with_minute,
    _load_stage_i_common30_candidates,
    _materialise_month_sides,
    _materialise_symbol_checkpoint_batch,
    _assemble_symbol_checkpoints,
    _minute_path_pruned,
    _overlapping_minute_fragments,
    _packb_to_kraken_symbol,
    _validate_candidate_frame,
    _write_status,
)
from scripts.materialize_stage_i_2024_2026_surface import (
    LABEL_COLUMNS,
    R3_PRIMITIVES,
    _packb_exact_labels_for_month,
    _join_packb_exact_labels,
)


def _minute_fixture(signal: pd.Timestamp) -> pd.DataFrame:
    start = signal - pd.Timedelta(hours=14)
    end = signal + pd.Timedelta(hours=13)
    index = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)


def _candidate(signal: pd.Timestamp, side: str = "long") -> pd.DataFrame:
    symbol = "TEST/USD:USD"
    return pd.DataFrame({
        "candidate_id": [f"{symbol}|{signal.isoformat()}|1h|{side}"],
        "__ts__": [signal],
        "__symbol__": [symbol],
        "side_name": [side],
    })


def test_packb_symbol_normalisation_matches_kraken_execution_paths() -> None:
    assert _packb_to_kraken_symbol("BTC/USD:USD") == "BTC_USD:USD"


def test_stage_i_common30_loader_preserves_frozen_identity_and_month_side(tmp_path) -> None:
    signal = pd.Timestamp("2026-02-10 00:00", tz="UTC")
    source = tmp_path / "request"
    source.mkdir()
    pd.DataFrame({
        "candidate_id": ["frozen-long", "frozen-short", "other-month"],
        "signal_timestamp": [signal, signal, pd.Timestamp("2026-03-01", tz="UTC")],
        "symbol": ["BTC/USD:USD"] * 3,
        "side_name": ["long", "short", "long"],
    }).to_parquet(source / "staged_candidates.parquet", index=False)
    out = _load_stage_i_common30_candidates(
        source, pd.Timestamp("2026-02-01", tz="UTC"), "long"
    )
    assert out["candidate_id"].tolist() == ["frozen-long"]
    assert out["__ts__"].tolist() == [signal]
    assert out["__symbol__"].tolist() == ["BTC/USD:USD"]


def test_atr14_is_available_at_signal_close_and_does_not_use_decision_path() -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    minute = _minute_fixture(signal)
    baseline = _causal_hourly_atr_from_minute(minute).loc[signal]
    decision = signal + pd.Timedelta(hours=1)
    minute.loc[decision:decision + pd.Timedelta(minutes=59), ["high", "low"]] = [200.0, 1.0]
    after = _causal_hourly_atr_from_minute(minute).loc[signal]
    assert np.isfinite(baseline) and baseline > 0.0
    assert after == baseline


def test_atr14_refuses_a_partial_hour_in_its_completed_input_history() -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    minute = _minute_fixture(signal)
    # This falls in the first of the 14 hourly candles feeding ATR at signal.
    minute.loc[signal - pd.Timedelta(hours=13, minutes=31), ["open", "high", "low", "close"]] = np.nan
    assert np.isnan(_causal_hourly_atr_from_minute(minute).loc[signal])


def test_same_minute_double_touch_is_adverse_and_cost_is_applied_once() -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    minute = _minute_fixture(signal)
    decision = signal + pd.Timedelta(hours=1)
    # ATR is 2.0 in the fixture: high reaches +6 ATR and low reaches -4 ATR.
    minute.loc[decision, ["high", "low"]] = [112.0, 92.0]
    out = _label_candidates_with_minute(_candidate(signal), minute)
    assert bool(out.loc[0, "label_valid"])
    assert out.loc[0, "t2_tp6_sl4_event"] == 1.0
    assert out.loc[0, "t4_tp6_sl4_exit_pnl_atr"] == -4.0
    assert np.isclose(out.loc[0, "t4_tp6_sl4_net_bps"], out.loc[0, "t4_tp6_sl4_gross_bps"] - COST_BPS)
    assert out.loc[0, "lower_touch_minute"] == 1.0
    assert out.loc[0, "robust_clear_event_b25"] == 0.0


def test_incomplete_h12_path_is_invalid_not_a_timeout_or_economic_failure() -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    minute = _minute_fixture(signal)
    missing = signal + pd.Timedelta(hours=4, minutes=7)
    minute.loc[missing, ["open", "high", "low", "close"]] = np.nan
    out = _label_candidates_with_minute(_candidate(signal), minute)
    assert not bool(out.loc[0, "label_valid"])
    assert bool(out.loc[0, "target_invalid"])
    assert out.loc[0, "invalid_reason"] == "incomplete_h12_ohlc_path"
    targets = [
        "t2_tp6_sl4_event", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
        "pre_adverse_mfe_atr", "lower_touch_minute", "robust_clear_event_b25",
        "robust_clear_soft_b25_t50",
    ]
    assert out.loc[0, targets].isna().all()


def test_partial_month_side_checkpoint_manifest_never_claims_full_completion(tmp_path) -> None:
    months = [
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-02-01", tz="UTC"),
    ]
    _write_status(
        tmp_path,
        [{"month": "2025-01", "side": "long", "status": "materialised", "rows": 1}],
        source=tmp_path / "source",
        minute_root=tmp_path / "minute",
        start=months[0],
        end=pd.Timestamp("2025-03-01", tz="UTC"),
        required_months=months,
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    consumer_manifest = json.loads((tmp_path / "manifest.json").read_text())
    coverage = pd.read_parquet(tmp_path / "coverage.parquet")
    assert manifest["status"] == "partial"
    assert manifest == consumer_manifest
    assert manifest["complete"] is False
    assert manifest["cells"]["expected"] == 4
    assert len(coverage) == 1


def test_month_sides_loads_one_shared_symbol_path_and_atr_substrate() -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    calls: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []

    def loader(root, symbol, start, end):
        calls.append((str(root), symbol, start, end))
        return _minute_fixture(signal)

    outputs = _materialise_month_sides(
        {"long": _candidate(signal, "long"), "short": _candidate(signal, "short")},
        Path("/irrelevant"),
        minute_loader=loader,
    )
    assert len(calls) == 1
    assert set(outputs) == {"long", "short"}
    assert all(len(frame) == 1 for frame in outputs.values())


def test_completed_relabel_checkpoint_is_readable_by_stage_i_surface(tmp_path) -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    minute = _minute_fixture(signal)
    records = []
    for side in ("long", "short"):
        out = _label_candidates_with_minute(_candidate(signal, side), minute)
        destination = tmp_path / "parts" / "month=2025-01" / f"side={side}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(destination, index=False)
        records.append({"month": "2025-01", "side": side, "status": "materialised", "rows": 1})
    month = pd.Timestamp("2025-01-01", tz="UTC")
    _write_status(
        tmp_path, records, source=tmp_path / "source", minute_root=tmp_path / "minute",
        start=month, end=pd.Timestamp("2025-02-01", tz="UTC"), required_months=[month],
    )
    labels, manifest = _packb_exact_labels_for_month("2025-01", sidecar=tmp_path)
    assert manifest["complete"] is True
    assert set(LABEL_COLUMNS).issubset(labels.columns)
    assert set(R3_PRIMITIVES).issubset(labels.columns)
    assert len(labels) == 2
    candidates = labels.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__"]].copy()
    joined = _join_packb_exact_labels(candidates, labels)
    assert len(joined) == len(candidates)
    assert joined["label_available_ts"].eq(joined["__decision_ts__"] + pd.Timedelta(hours=12)).all()


def test_historical_hashed_candidate_identity_is_validated_without_packb_spelling() -> None:
    signal = pd.Timestamp("2023-10-10 00:00", tz="UTC")
    historical = _candidate(signal).assign(candidate_id="a" * 64)
    accepted = _validate_candidate_frame(
        historical, month=pd.Timestamp("2023-10-01", tz="UTC"), side="long",
        source_kind="historical", source_name=Path("historical_candidates.parquet"),
    )
    assert accepted.loc[0, "candidate_id"] == "a" * 64
    try:
        _validate_candidate_frame(
            historical, month=pd.Timestamp("2023-10-01", tz="UTC"), side="long",
            source_kind="packb", source_name=Path("not_packb.parquet"),
        )
    except ValueError as exc:
        assert "Pack-B candidate identity" in str(exc)
    else:
        raise AssertionError("Pack-B mode must retain its transparent candidate-id contract")


def test_packb_candidate_identity_accepts_equivalent_zulu_timestamp_spelling() -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    frame = _candidate(signal).assign(
        candidate_id="TEST/USD:USD|2025-01-10T00:00:00Z|1h|long"
    )
    accepted = _validate_candidate_frame(
        frame,
        month=pd.Timestamp("2025-01-01", tz="UTC"),
        side="long",
        source_kind="packb",
        source_name=Path("packb.parquet"),
    )
    assert accepted.loc[0, "candidate_id"].endswith("Z|1h|long")


def test_minute_fragment_pruning_reads_only_overlapping_epoch_ranges(tmp_path: Path) -> None:
    symbol = "TEST_USD:USD"
    location = tmp_path / f"symbol={symbol}" / "year=2025"
    location.mkdir(parents=True)
    wanted_ts = pd.date_range("2025-01-02T00:00:00Z", periods=2, freq="min")
    old_ts = pd.date_range("2025-01-01T00:00:00Z", periods=2, freq="min")
    for name, ts, value in (
        ("old", old_ts, 1.0),
        ("wanted", wanted_ts, 2.0),
    ):
        start, end = int(ts[0].timestamp()), int(ts[-1].timestamp())
        pd.DataFrame({"ts": ts, "open": value, "high": value, "low": value, "close": value}).to_parquet(
            location / f"part-{name}-{start}-{end}.parquet", index=False
        )
    start, end = wanted_ts[0], wanted_ts[-1] + pd.Timedelta(minutes=1)
    selected = _overlapping_minute_fragments(tmp_path, symbol, start, end)
    assert [path.name for path in selected] == [f"part-wanted-{int(wanted_ts[0].timestamp())}-{int(wanted_ts[-1].timestamp())}.parquet"]
    frame = _minute_path_pruned(tmp_path, symbol, start, end)
    assert frame.open.tolist() == [2.0, 2.0]


def test_symbol_batches_resume_and_assemble_exact_month(tmp_path: Path) -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    one = _candidate(signal)
    two = _candidate(signal).assign(
        __symbol__="OTHER/USD:USD",
        candidate_id=f"OTHER/USD:USD|{signal.isoformat()}|1h|long",
    )
    candidates = {"long": pd.concat([one, two], ignore_index=True)}

    def loader(_root, _symbol, _start, _end):
        return _minute_fixture(signal)

    first = _materialise_symbol_checkpoint_batch(
        candidates, tmp_path, month=pd.Timestamp("2025-01-01", tz="UTC"),
        output_root=tmp_path / "out", batch_size=1, minute_loader=loader,
    )
    assert first == {"total_symbols": 2, "processed_this_batch": 1, "remaining_symbols": 1}
    assert _assemble_symbol_checkpoints(
        candidates, month=pd.Timestamp("2025-01-01", tz="UTC"), output_root=tmp_path / "out"
    ) is None
    second = _materialise_symbol_checkpoint_batch(
        candidates, tmp_path, month=pd.Timestamp("2025-01-01", tz="UTC"),
        output_root=tmp_path / "out", batch_size=1, minute_loader=loader,
    )
    assert second["remaining_symbols"] == 0
    assembled = _assemble_symbol_checkpoints(
        candidates, month=pd.Timestamp("2025-01-01", tz="UTC"), output_root=tmp_path / "out"
    )
    assert assembled is not None
    assert set(assembled["long"].candidate_id) == set(candidates["long"].candidate_id)


def test_absent_minute_symbol_is_explicit_invalid_coverage_not_a_crash(tmp_path: Path) -> None:
    signal = pd.Timestamp("2025-01-10 00:00", tz="UTC")
    output = _materialise_month_sides({"long": _candidate(signal)}, tmp_path)["long"]
    assert not output.label_valid.any()
    assert output.target_invalid.all()
    assert output.invalid_reason.eq("symbol_minute_source_unavailable").all()
