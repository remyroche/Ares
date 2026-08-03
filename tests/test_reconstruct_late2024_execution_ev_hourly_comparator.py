from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.reconstruct_late2024_execution_ev_hourly_comparator import (
    BASE_SCORE,
    BASE_TARGET,
    DIRECT_SCORE,
    TARGET,
    comparator_manifest_contract,
    generate_base_oof,
    generate_execution_ev_oof,
    eligible_raw_features,
    load_pit_candidate_universe,
    top10_global_metrics,
)


def _write_feature_file(path, symbol: str, timestamps: pd.DatetimeIndex) -> None:
    pd.DataFrame(
        {
            "ts": timestamps,
            "__symbol__": [symbol] * len(timestamps),
            "raw_a": np.arange(len(timestamps), dtype=np.float32),
            "raw_b": np.sin(np.arange(len(timestamps), dtype=np.float32)),
        }
    ).to_parquet(path, index=False)


def test_pit_universe_uses_only_rows_physically_available_in_requested_window(tmp_path) -> None:
    timestamps = pd.to_datetime(
        ["2024-09-30T23:00Z", "2024-10-01T00:00Z", "2024-10-01T01:00Z", "2025-01-01T00:00Z"],
        utc=True,
    )
    _write_feature_file(tmp_path / "symbol=BTC_USD:USD.parquet", "BTC_USD:USD", timestamps)
    universe, features, report = load_pit_candidate_universe(
        tmp_path,
        start=pd.Timestamp("2024-10-01T00:00Z"),
        end=pd.Timestamp("2024-10-02T00:00Z"),
        configured_features=("raw_a", "raw_b", "raw_missing"),
        minimum_features=2,
    )
    assert features == ["raw_a", "raw_b"]
    assert universe["__ts__"].tolist() == list(timestamps[1:3])
    assert universe["__symbol__"].eq("BTC_USD:USD").all()
    assert report["rows_in_requested_interval"] == 2
    assert report["timestamp_physical_columns"] == {"ts": 1, "__index_level_0__": 0, "missing": 0}


def test_pit_universe_accepts_legacy_unnamed_datetime_index_safely(tmp_path) -> None:
    index = pd.date_range("2024-10-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"__symbol__": ["ETH_USD:USD"] * 3, "raw_a": [1.0, 2.0, 3.0], "raw_b": [4.0, 5.0, 6.0]},
        index=index,
    )
    frame.to_parquet(tmp_path / "symbol=ETH_USD:USD.parquet")
    universe, _, report = load_pit_candidate_universe(
        tmp_path,
        start=pd.Timestamp("2024-10-01T01:00Z"),
        end=pd.Timestamp("2024-10-01T03:00Z"),
        configured_features=("raw_a", "raw_b"),
        minimum_features=2,
    )
    assert universe["__ts__"].tolist() == list(index[1:])
    assert report["timestamp_physical_columns"]["__index_level_0__"] == 1


def test_raw_pool_rejects_untrusted_historical_funding_fields(tmp_path) -> None:
    _write_feature_file(
        tmp_path / "symbol=SOL_USD:USD.parquet",
        "SOL_USD:USD",
        pd.date_range("2024-10-01", periods=3, freq="h", tz="UTC"),
    )
    path = tmp_path / "symbol=SOL_USD:USD.parquet"
    frame = pd.read_parquet(path)
    frame["funding_per_hour"] = 0.001
    frame["xasset_btc_fund_z"] = 0.002
    frame.to_parquet(path, index=False)
    features, _ = eligible_raw_features(
        [path],
        configured_features=("raw_a", "raw_b", "funding_per_hour", "xasset_btc_fund_z"),
        minimum_features=2,
    )
    assert features == ["raw_a", "raw_b"]


def test_metrics_slice_one_global_book_instead_of_reranking_sides() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2024-10-01", periods=20, freq="h", tz="UTC"),
            "candidate_month": ["2024-10"] * 10 + ["2024-11"] * 10,
            "side_name": ["long"] * 10 + ["short"] * 10,
            DIRECT_SCORE: np.arange(20, dtype=float),
            TARGET: np.arange(20, dtype=float) / 100.0,
        }
    )
    metrics = top10_global_metrics(frame)
    assert metrics["global"]["global_top10_rows"] == 2
    assert set(metrics["global_book_by_month"]) == {"2024-11"}
    assert set(metrics["global_book_by_side"]) == {"short"}


def _synthetic_labels() -> pd.DataFrame:
    timestamps = pd.date_range("2024-07-01", "2024-12-31 18:00", freq="6h", tz="UTC")
    rows = []
    for side_number, side in enumerate(("long", "short")):
        for symbol_number, symbol in enumerate(("A_USD:USD", "B_USD:USD", "C_USD:USD")):
            for number, timestamp in enumerate(timestamps):
                raw_a = np.sin(number / 11.0 + side_number + symbol_number / 3.0)
                raw_b = float((number + symbol_number) % 13)
                target = 0.012 * np.tanh(raw_a - 0.03 * raw_b + 0.1 * side_number)
                rows.append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": symbol,
                        "side_name": side,
                        "candidate_id": f"{side}-{symbol}-{number}",
                        "execution_decision_utc": timestamp + pd.Timedelta(hours=1),
                        "execution_label_end_utc": timestamp + pd.Timedelta(hours=13),
                        "candidate_month": timestamp.strftime("%Y-%m"),
                        TARGET: target,
                        BASE_TARGET: 1.0 / (1.0 + np.exp(-target / 0.01)),
                        "raw_a": raw_a,
                        "raw_b": raw_b,
                    }
                )
    return pd.DataFrame(rows)


def test_two_layer_hourly_comparator_is_side_local_forward_and_inner_oof_only(monkeypatch) -> None:
    import scripts.reconstruct_late2024_execution_ev_hourly_comparator as module

    monkeypatch.setattr(module, "MIN_BASE_TRAIN_ROWS", 100)
    monkeypatch.setattr(module, "MIN_META_TRAIN_ROWS", 100)
    monkeypatch.setattr(module, "MAX_FIT_ROWS", 500)
    monkeypatch.setattr(module, "FOLD_DAYS", 28)
    labels = _synthetic_labels()
    base, base_audit = generate_base_oof(
        labels,
        ["raw_a", "raw_b"],
        start=pd.Timestamp("2024-07-01T00:00Z"),
        end=pd.Timestamp("2025-01-01T00:00Z"),
    )
    direct, direct_audit = generate_execution_ev_oof(
        base,
        evaluation_start=pd.Timestamp("2024-10-01T00:00Z"),
        end=pd.Timestamp("2025-01-01T00:00Z"),
    )
    assert direct["__ts__"].min() == pd.Timestamp("2024-10-01T00:00Z")
    assert set(direct["candidate_month"]) == {"2024-10", "2024-11", "2024-12"}
    assert np.isfinite(direct[DIRECT_SCORE]).all()
    assert direct[BASE_SCORE].notna().all()
    for audit in [*base_audit, *direct_audit]:
        if audit["status"] == "trained":
            assert pd.Timestamp(audit["max_train_label_end_utc"]) <= pd.Timestamp(audit["fold_start_utc"])
    assert base["base_oof_train_cutoff_utc"].le(base["__ts__"]).all()
    assert direct["execution_ev_oof_train_cutoff_utc"].le(direct["__ts__"]).all()


def test_meta_layer_fails_closed_when_base_score_was_not_prior_oof() -> None:
    base = _synthetic_labels().iloc[:200].copy()
    base[BASE_SCORE] = 0.5
    base["base_oof_fold_start_utc"] = base["__ts__"]
    base["base_oof_train_cutoff_utc"] = base["__ts__"] + pd.Timedelta(hours=1)
    try:
        generate_execution_ev_oof(
            base,
            evaluation_start=pd.Timestamp("2024-10-01T00:00Z"),
            end=pd.Timestamp("2024-11-01T00:00Z"),
        )
    except ValueError as error:
        assert "cutoff is after" in str(error)
    else:
        raise AssertionError("meta layer accepted a future-fitted base score")


def test_manifest_is_explicitly_hourly_only_and_non_poolable() -> None:
    contract = comparator_manifest_contract()
    assert contract["evidence_tier"] == "hourly_bar_approximation"
    text = " ".join(contract["forbidden_claims"])
    assert "exact 1m" in text
    assert "entry-timing" in text
    assert "L2/spread" in text
    assert "pooling metrics with exact-1m tier" in text
