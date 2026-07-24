from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import _load_feature_store_columns
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _merge_authoritative_store_features,
    _missing_feature_store_columns,
    _resolve_base_model_features,
)


def test_feature_loader_uses_unified_base_plus_delta_store(
    tmp_path: Path, monkeypatch
) -> None:
    feature_dir = tmp_path / "data_perp" / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    path = feature_dir / "symbol=BTC_USD:USD.parquet"
    pd.DataFrame(
        {"base_only": [1.0]},
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="ts"),
    ).to_parquet(path)
    ts = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")

    from extreme_price_movements import static_feature_store

    def fake_read(*, feature_store_ts, data_root, feature_keys, symbols, start_ts, end_ts):
        assert feature_store_ts == pd.Timestamp("2026-01-01T00:00:00Z")
        assert data_root == tmp_path / "data_perp"
        assert feature_keys == ["delta_only"]
        assert symbols == ["BTC/USD:USD"]
        assert start_ts == ts.min()
        assert end_ts == ts.max()
        return {
            "delta_only": pd.DataFrame(
                {"BTC/USD:USD": [2.0, 3.0]}, index=ts
            )
        }

    monkeypatch.setattr(static_feature_store, "read_static_features", fake_read)
    frame = pd.DataFrame(
        {"__ts__": ts, "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"]}
    )

    matrix, report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=["delta_only"],
    )

    assert matrix["delta_only"].tolist() == [2.0, 3.0]
    assert report["retained_features"] == 1
    assert report["loaded_symbols"] == 1


def test_missing_feature_store_columns_preserves_order_and_deduplicates() -> None:
    assert _missing_feature_store_columns(
        ["embedded_a", "embedded_b"],
        ["embedded_a", "store_c", "store_c", "embedded_b", "store_d"],
    ) == ["store_c", "store_d"]


def test_static_store_values_replace_embedded_label_features() -> None:
    labels = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "target_soft": [0.1, 0.9],
            "observable": [99.0, 99.0],
            "label_only_context": [1, 2],
        },
        index=[10, 11],
    )
    static = pd.DataFrame(
        {"observable": [1.25, 2.5], "new_feature": [3.0, 4.0]},
        index=[20, 21],
    )

    merged = _merge_authoritative_store_features(labels, static)

    assert merged.index.tolist() == [0, 1]
    assert merged["observable"].tolist() == [1.25, 2.5]
    assert merged["new_feature"].tolist() == [3.0, 4.0]
    assert merged["target_soft"].tolist() == [0.1, 0.9]
    assert merged["label_only_context"].tolist() == [1, 2]


def test_base_contract_excludes_embedded_observable_fallbacks() -> None:
    rows = 120
    frame = pd.DataFrame(
        {
            "store_feature": range(rows),
            "stale_embedded_feature": range(rows, 2 * rows),
            "side": [1, -1] * (rows // 2),
            "__archetype_label_family__": ["trend"] * rows,
        }
    )

    selected = _resolve_base_model_features(
        frame,
        None,
        authoritative_store_features=["store_feature"],
    )

    assert selected == ["store_feature", "side"]


def test_fixed_base_contract_rejects_embedded_observable_fallback() -> None:
    rows = 120
    frame = pd.DataFrame(
        {
            "store_feature": range(rows),
            "stale_embedded_feature": range(rows, 2 * rows),
            "side": [1, -1] * (rows // 2),
        }
    )

    try:
        _resolve_base_model_features(
            frame,
            ["store_feature", "stale_embedded_feature"],
            authoritative_store_features=["store_feature"],
        )
    except RuntimeError as exc:
        assert "not supplied by the authoritative static store" in str(exc)
    else:
        raise AssertionError("stale embedded feature should be rejected")
