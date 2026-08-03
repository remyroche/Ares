import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_production_data_adapter import (
    MonthlyReferencePartition,
    StageIProductionDataError,
    iter_frozen_oof_chunks,
    load_selector_sample,
    load_reference_ledgers,
    _canonical_symbol,
)


def _partition(path, month, population, offset=0):
    n = 12
    ts = pd.date_range(month, periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"{population}-{offset + i}" for i in range(n)],
        "__ts__": ts,
        "__symbol__": ["BTC" if i % 2 == 0 else "ETH" for i in range(n)],
        "side_name": ["long" if i % 2 == 0 else "short" for i in range(n)],
        "label_valid": [True] * (n - 1) + [False],
        "exact_net_bps": np.resize(np.array([-250.0, -20.0, 20.0, 120.0]), n),
        "exact_gross_bps": np.resize(np.array([-150.0, 80.0, 120.0, 220.0]), n),
        "label_available_ts": ts + pd.Timedelta(hours=13),
        "t2_tp6_sl4_event": np.resize(np.array([1, 2, 2, 0]), n),
        "robust_clear_event_b25": np.resize(np.array([0, 0, 0, 1]), n),
        "robust_clear_soft_b25_t50": np.resize(np.array([0.0, 0.1, 0.4, 0.9]), n),
    })
    frame.to_parquet(path, index=False)
    return MonthlyReferencePartition(path=path, source_month=month[:7], population=population)


def _loader(ledger, fields):
    out = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
    for i, field in enumerate(fields):
        out[str(field)] = np.arange(len(out), dtype=np.float32) + i
    return out


def test_reference_surface_symbol_alias_is_normalized_to_store_payload() -> None:
    assert _canonical_symbol("BTC_USD:USD") == "BTC/USD:USD"
    assert _canonical_symbol("BTC/USD:USD") == "BTC/USD:USD"


def test_selector_sample_filters_before_exact_pit_load_and_keeps_population_lineage(tmp_path) -> None:
    p1 = _partition(tmp_path / "2023.parquet", "2023-05-01", "historical_2022_2023")
    p2 = _partition(tmp_path / "2025.parquet", "2025-05-01", "common30_2025_2026", offset=100)
    seen = []

    def loader(ledger, fields):
        seen.append(len(ledger))
        return _loader(ledger, fields)

    chunk = load_selector_sample(
        [p1, p2], declared_features=["f1", "f2"], pit_feature_loader=loader,
        selector_max_rows=8, frozen_candidate_universe=None,
    )
    assert seen == [len(chunk.ledger)]
    assert len(chunk.ledger) <= 8
    assert set(chunk.ledger["population_segment"]) == {"historical_2022_2023", "common30_2025_2026"}
    assert chunk.features[["f1", "f2"]].isna().sum().sum() == 0


def test_adapter_rejects_low_coverage_or_non_exact_loader_and_streams_selected_only(tmp_path) -> None:
    p = _partition(tmp_path / "2024.parquet", "2024-05-01", "surface_2024")

    def low_coverage(ledger, fields):
        out = _loader(ledger, fields)
        out["f1"] = np.nan
        return out

    with pytest.raises(StageIProductionDataError, match="coverage below 90%"):
        load_selector_sample([p], declared_features=["f1", "f2"], pit_feature_loader=low_coverage, selector_max_rows=10)

    chunks = list(iter_frozen_oof_chunks(
        [p], selected_features=["f1", "f2"], pit_feature_loader=_loader, batch_rows=4
    ))
    assert sum(len(chunk.ledger) for chunk in chunks) == 11  # label_valid only
    assert all(list(chunk.features.columns[-2:]) == ["f1", "f2"] for chunk in chunks)


def test_adapter_reads_real_reference_directory_aliases(tmp_path) -> None:
    directory = tmp_path / "identity_labels"
    directory.mkdir()
    ts = pd.date_range("2024-06-01", periods=4, freq="h", tz="UTC")
    for symbol, positions in (("BTC", [0, 2]), ("ETH", [1, 3])):
        pd.DataFrame(
            {
                "candidate_id": [f"{symbol}-{i}" for i in positions],
                "__ts__": ts[positions],
                "__symbol__": symbol,
                "side_name": ["long" if i % 2 == 0 else "short" for i in positions],
                "label_valid": True,
                "t4_tp6_sl4_net_bps": [10.0 + i for i in positions],
                "t4_tp6_sl4_gross_bps": [110.0 + i for i in positions],
                "__label_available_at__": ts[positions] + pd.Timedelta(hours=13),
                "t2_tp6_sl4_event": [2 for _ in positions],
                "robust_clear_event_b25": [0 for _ in positions],
                "robust_clear_soft_b25_t50": [0.25 for _ in positions],
            }
        ).to_parquet(directory / f"symbol={symbol}.parquet", index=False)
    ledger = load_reference_ledgers(
        [MonthlyReferencePartition(directory, "2024-06", "surface_2024")]
    )
    assert len(ledger) == 4
    assert {"exact_gross_bps", "exact_net_bps", "label_available_ts"}.issubset(ledger.columns)
    assert ledger["exact_net_bps"].notna().all()
