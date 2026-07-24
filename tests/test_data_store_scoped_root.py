import pandas as pd
import pytest

from extreme_price_movements.data_store import (
    _feature_delta_compact_rows,
    _kraken_oi_to_quote_notional,
    append_symbol_features,
    read_symbol_features,
    scoped_data_root,
)


def test_scoped_data_root_is_idempotent_for_exchange_scoped_perp_root():
    cfg = {
        "data_root": "data_perp/exchanges/krakenfutures",
        "exchange_id": "krakenfutures",
        "market_mode": "perps",
    }

    assert scoped_data_root(cfg) == "data_perp/exchanges/krakenfutures"


def test_scoped_data_root_scopes_plain_perp_root():
    cfg = {
        "data_root": "data_perp",
        "exchange_id": "krakenfutures",
        "market_mode": "perps",
    }

    assert scoped_data_root(cfg) == "data_perp/exchanges/krakenfutures"


def test_kraken_analytics_open_interest_is_converted_to_quote_notional():
    index = pd.date_range("2026-07-15 14:00", periods=2, freq="h", tz="UTC")
    native = pd.Series([2_000.0, 2_100.0], index=index)
    prices = pd.DataFrame({"close": [65_000.0, 66_000.0]}, index=index)

    quote = _kraken_oi_to_quote_notional(native, prices)

    assert quote.tolist() == [130_000_000.0, 138_600_000.0]


def test_kraken_analytics_open_interest_falls_back_to_perp_close_per_row():
    index = pd.date_range("2026-07-15 14:00", periods=3, freq="h", tz="UTC")
    native = pd.Series([2_000.0, 2_100.0, 2_200.0], index=index)
    prices = pd.DataFrame(
        {
            "mark_close": [65_100.0, float("nan"), 67_100.0],
            "close": [65_000.0, 66_000.0, 67_000.0],
        },
        index=index,
    )

    quote = _kraken_oi_to_quote_notional(native, prices)

    assert quote.tolist() == [130_200_000.0, 138_600_000.0, 147_620_000.0]


def test_feature_duckdb_delta_compacts_at_configured_row_budget(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    monkeypatch.setenv("EPM_FEATURE_DELTA_DUCKDB", "1")
    monkeypatch.setenv("EPM_FEATURE_DELTA_APPEND", "1")
    monkeypatch.setenv("EPM_FEATURE_DELTA_COMPACT_ROWS", "2")

    path = tmp_path / "symbol=BTC_USD.parquet"
    index = pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC")
    base = pd.DataFrame({"feature_a": [1.0]}, index=index[:1])
    append_symbol_features(str(path), "BTC/USD:USD", base)

    append_symbol_features(
        str(path),
        "BTC/USD:USD",
        pd.DataFrame({"feature_a": [2.0]}, index=index[1:2]),
    )
    assert (tmp_path / "symbol=BTC_USD.parquet.deltas.duckdb").exists()

    append_symbol_features(
        str(path),
        "BTC/USD:USD",
        pd.DataFrame({"feature_a": [3.0]}, index=index[2:]),
    )

    # The buffer is allowed to reach the configured row budget.  The next
    # appended row makes it exceed that budget and triggers compaction.
    assert (tmp_path / "symbol=BTC_USD.parquet.deltas.duckdb").exists()
    next_index = pd.date_range("2026-07-01 03:00", periods=1, freq="h", tz="UTC")
    append_symbol_features(
        str(path),
        "BTC/USD:USD",
        pd.DataFrame({"feature_a": [4.0]}, index=next_index),
    )

    # The third buffered update exceeds the budget.  The merged parquet view
    # remains complete and the transient DuckDB buffer is removed.
    assert not (tmp_path / "symbol=BTC_USD.parquet.deltas.duckdb").exists()
    result = read_symbol_features(str(path), columns=["feature_a"])
    assert result["feature_a"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_feature_delta_default_compaction_budget_is_ten_thousand(monkeypatch):
    monkeypatch.delenv("EPM_FEATURE_DELTA_COMPACT_ROWS", raising=False)
    assert _feature_delta_compact_rows() == 10_000


def test_selected_feature_read_skips_nonoverlapping_base_part(tmp_path, monkeypatch):
    import extreme_price_movements.data_store as data_store

    monkeypatch.setenv("EPM_FEATURE_DELTA_DUCKDB", "0")
    monkeypatch.setenv("EPM_FEATURE_DELTA_APPEND", "1")
    path = tmp_path / "symbol=BTC_USD.parquet"
    index = pd.date_range("2026-07-01", periods=2, freq="24h", tz="UTC")
    append_symbol_features(
        str(path),
        "BTC/USD:USD",
        pd.DataFrame({"feature_a": [1.0]}, index=index[:1]),
    )
    append_symbol_features(
        str(path),
        "BTC/USD:USD",
        pd.DataFrame({"feature_a": [2.0]}, index=index[1:]),
    )

    original_read = data_store.pd.read_parquet
    read_paths: list[str] = []

    def _tracked_read(path_arg, *args, **kwargs):
        read_paths.append(str(path_arg))
        return original_read(path_arg, *args, **kwargs)

    monkeypatch.setattr(data_store.pd, "read_parquet", _tracked_read)
    result = read_symbol_features(
        str(path),
        columns=["feature_a"],
        start_ts=index[-1],
        end_ts=index[-1],
    )

    assert result["feature_a"].tolist() == [2.0]
    assert read_paths
    assert all(".deltas" in read_path for read_path in read_paths)
