from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/backfill_historical_execution_ev_12h_oof.py"
SPEC = importlib.util.spec_from_file_location("historical_backfill", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_historical_symbol_normalization_and_stable_identity() -> None:
    assert MODULE.normalize_historical_symbol("BTC/USD:USD") == "BTC_USD:USD"
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    assert MODULE.stable_candidate_id(ts, "BTC_USD:USD", "long") == MODULE.stable_candidate_id(ts, "BTC_USD:USD", "long")
    assert MODULE.stable_candidate_id(ts, "BTC_USD:USD", "long") != MODULE.stable_candidate_id(ts, "BTC_USD:USD", "short")


def test_monthly_oof_never_trains_on_open_label_interval() -> None:
    rows = []
    for side in ("long", "short"):
        for day in pd.date_range("2025-01-01", "2025-04-30", freq="h", tz="UTC"):
            rows.append({"__ts__": day, "__symbol__": "BTC_USD:USD", "side_name": side, "candidate_id": f"{side}-{day}", "candidate_month": str(day.to_period("M")), "execution_label_end_utc": day + pd.Timedelta(hours=13), "execution_net_ev_12h": 0.01 if day.hour % 2 else -0.01, **{f: float(day.hour) for f in MODULE.FEATURES}})
    out, audit = MODULE.strict_monthly_oof(pd.DataFrame(rows))
    assert not out.empty
    for row in audit:
        if row["status"] == "trained":
            assert pd.Timestamp(row["train_cutoff_utc"]) < pd.Timestamp(f"{row['month']}-01", tz="UTC")
    assert (out["oof_train_cutoff_utc"] < out["__ts__"]).all()
