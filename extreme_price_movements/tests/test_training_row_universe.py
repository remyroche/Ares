import pandas as pd
import pytest

from extreme_price_movements.pipeline_steps import _row_universe_filter_datasets


def test_row_universe_filter_keeps_exact_timestamp_symbol_rows(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
                utc=True,
            ),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "x": [1, 2, 3],
        }
    )
    universe = df.iloc[[0, 2]][["__ts__", "__symbol__"]].rename(
        columns={"__ts__": "timestamp", "__symbol__": "symbol"}
    )
    path = tmp_path / "universe.parquet"
    universe.to_parquet(path, index=False)
    monkeypatch.setenv("EPM_TRAIN_ROW_UNIVERSE_PATH", str(path))

    out = _row_universe_filter_datasets({"train_x": df}, {})

    assert len(out["train_x"]) == 2
    assert out["train_x"]["__symbol__"].tolist() == ["BTC/USD:USD", "SOL/USD:USD"]


def test_row_universe_strict_rejects_non_identical_rows(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            "__symbol__": ["BTC/USD:USD"],
        }
    )
    universe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                utc=True,
            ),
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
        }
    )
    path = tmp_path / "universe.parquet"
    universe.to_parquet(path, index=False)
    monkeypatch.setenv("EPM_TRAIN_ROW_UNIVERSE_PATH", str(path))
    monkeypatch.setenv("EPM_TRAIN_ROW_UNIVERSE_STRICT", "1")

    with pytest.raises(ValueError, match="not row-identical"):
        _row_universe_filter_datasets({"train_x": df}, {})
