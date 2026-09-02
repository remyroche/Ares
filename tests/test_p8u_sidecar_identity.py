from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_tp6_sl4_exact170_canonical_consensus as source


def _write(path, field: str, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-08-14T12:00:00Z", periods=4, freq="h")
    pd.DataFrame({field: np.repeat(value, len(index))}, index=index).to_parquet(path)


def test_oi_funding_prefers_exact_linear_usd_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "ROOT", tmp_path)
    funding = tmp_path / "data_perp/exchanges/krakenfutures/funding_hourly"
    oi = tmp_path / "data_perp/exchanges/krakenfutures/open_interest_hourly"
    # The broad prefix sorts this stale alias first; the source panel must not
    # consume it when the exact linear USD sidecar is available.
    _write(funding / "BTC_USD_BTC.parquet", "funding_rate", -9.0)
    _write(funding / "BTC_USD_USD.parquet", "funding_rate", 3.0)
    _write(oi / "BTC_USD_BTC.parquet", "open_interest", 9.0)
    _write(oi / "BTC_USD_USD.parquet", "open_interest", 3.0)
    index = pd.date_range("2026-08-14T13:00:00Z", periods=2, freq="h")
    panel = {}
    source._add_oi_funding_panels(
        panel, ["BTC/USD:USD"], index, index[0], index[-1] + pd.Timedelta(hours=1)
    )
    assert panel["funding_rate"].iloc[0, 0] == 3.0
    assert panel["open_interest"].iloc[0, 0] == 3.0
