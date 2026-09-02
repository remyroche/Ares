import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "backfill_kraken_oi_funding_sidecars.py"
SPEC = importlib.util.spec_from_file_location("oi_funding_sidecars_test", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_corrupt_sidecar_is_quarantined_not_partially_replaced(tmp_path: Path) -> None:
    root = tmp_path / "krakenfutures"
    path = root / "funding_hourly" / "TON_USD_USD.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PAR1not-a-valid-parquet-file")
    product = MODULE.Product("TON_USD:USD", "PF_TONUSD", "TON_USD_USD", "READY")

    result = MODULE._merge_observations(
        product,
        "funding_hourly",
        pd.Timestamp("2026-08-23T13:00:00Z"),
        pd.Timestamp("2026-08-23T16:00:00Z"),
        root,
        pd.Series([0.1], index=pd.DatetimeIndex(["2026-08-23T14:00:00Z"])),
        {"status": "OK"},
        root / "corrupt_sidecars",
    )

    assert result["status"] == "UNAVAILABLE_LOCAL_SIDECAR"
    assert result["partial_api_replacement_forbidden"] is True
    assert not path.exists()
    marker = path.with_name(path.name + ".unavailable.json")
    assert marker.exists()
    assert Path(result["quarantine"]["quarantine_path"]).is_file()

    repeat = MODULE._merge_observations(
        product,
        "funding_hourly",
        pd.Timestamp("2026-08-23T13:00:00Z"),
        pd.Timestamp("2026-08-23T16:00:00Z"),
        root,
        pd.Series([0.2], index=pd.DatetimeIndex(["2026-08-23T15:00:00Z"])),
        {"status": "OK"},
        root / "corrupt_sidecars",
    )
    assert repeat["status"] == "UNAVAILABLE_LOCAL_SIDECAR"
    assert repeat["merge_mode"] == "prior_quarantine_marker"
    assert not path.exists()
