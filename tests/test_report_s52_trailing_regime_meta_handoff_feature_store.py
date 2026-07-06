from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_trailing_regime_meta_handoff import (  # noqa: E402
    _join_feature_store_context,
    _train_meta_handoff,
    _feature_store_context_columns,
    run_handoff_only,
)


def test_feature_store_context_join_exports_safe_cross_market_columns(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    ts = pd.date_range("2026-06-01", periods=3, freq="h", tz="UTC", name="ts")
    features = pd.DataFrame(
        {
            "pct_assets_up_1h": [0.4, 0.5, 0.6],
            "btc_ret_24h_pct": [1.0, 2.0, 3.0],
            "xasset_ob_liquidity_peer_resid": [0.1, 0.2, 0.3],
            "spectral_entropy_ret_48": [0.8, 0.7, 0.6],
            "future_return_label": [99.0, 99.0, 99.0],
            "ordinary_feature": [7.0, 8.0, 9.0],
            "__symbol__": ["BTC/USD:USD"] * 3,
        },
        index=ts,
    )
    features.to_parquet(feature_dir / "symbol=BTC_USD:USD.parquet")
    ledger = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-06-01 00:00:00"), pd.Timestamp("2026-06-01 01:00:00")],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
            "side_name": ["long", "short"],
            "month": ["2026-06", "2026-06"],
            "score": [0.1, 0.2],
            "selected_top10": [True, True],
            "source_tag": ["source", "source"],
            "source_semantic_family": ["run_entry", "run_entry"],
            "gmm_cluster_posterior_0": [0.7, 0.6],
        }
    )

    joined, contract = _join_feature_store_context(ledger, feature_dir)
    assert contract["feature_store_context_status"] == "joined"
    assert contract["matched_rows"] == 2
    assert "pct_assets_up_1h" in contract["loaded_columns"]
    assert "btc_ret_24h_pct" in contract["loaded_columns"]
    assert "xasset_ob_liquidity_peer_resid" in contract["loaded_columns"]
    assert "spectral_entropy_ret_48" in contract["loaded_columns"]
    assert "future_return_label" not in joined.columns
    assert "ordinary_feature" not in joined.columns

    handoff, handoff_contract = _train_meta_handoff(
        joined,
        pd.DataFrame(),
        [],
        selected_col="selected_top10",
    )
    for col in ("pct_assets_up_1h", "btc_ret_24h_pct", "xasset_ob_liquidity_peer_resid", "spectral_entropy_ret_48"):
        assert col in handoff.columns
        assert col in handoff_contract["cross_market_context_columns"]


def test_handoff_only_materializes_feature_store_context(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    ts = pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC", name="ts")
    pd.DataFrame(
        {
            "pct_assets_up_1h": [0.4, 0.5, 0.6, 0.7],
            "btc_ret_24h_pct": [1.0, 2.0, 3.0, 4.0],
            "xasset_ob_liquidity_peer_resid": [0.1, 0.2, 0.3, 0.4],
            "__symbol__": ["BTC/USD:USD"] * 4,
        },
        index=ts,
    ).to_parquet(feature_dir / "symbol=BTC_USD:USD.parquet")
    ledger = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-04-01 00:00:00"), pd.Timestamp("2026-06-01 00:00:00")],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
            "side_name": ["long", "short"],
            "month": ["2026-04", "2026-06"],
            "score": [0.1, 0.2],
            "selected_top10": [True, True],
            "source_tag": ["source", "source"],
            "source_semantic_family": ["run_entry", "run_entry"],
            "u_policy_net": [0.01, 0.02],
            "ret_net": [0.01, 0.02],
            "mae_norm": [0.2, 0.3],
            "mfe_norm": [0.8, 0.9],
            "first_touch_net": [0.01, 0.02],
            "first_touch_full_path_mae_norm": [0.2, 0.3],
            "timeout": [0.0, 0.0],
            "clean_exec": [1.0, 1.0],
            "dirty_positive": [0.0, 0.0],
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    out = tmp_path / "out"
    manifest = run_handoff_only(
        ledger_path=ledger_path,
        output_dir=out,
        label_context_dir=None,
        feature_dir=feature_dir,
        feature_store_scope="cross_market",
        fit_months=["2026-04"],
        holdout_month="2026-06",
        selected_col="selected_top10",
        embedded_round_trip_cost=0.003,
        executable_cost_floor=0.010,
    )
    assert manifest["mode"] == "handoff_only"
    assert manifest["feature_store_context_contract"]["loaded_column_count"] >= 3
    handoff = pd.read_parquet(out / "train_meta_regime_handoff.parquet")
    assert "pct_assets_up_1h" in handoff.columns
    assert "btc_ret_24h_pct" in handoff.columns
    assert "xasset_ob_liquidity_peer_resid" in handoff.columns


def test_config_meta_full_scope_includes_normal_meta_keys_and_blocks_targets() -> None:
    columns = [
        "__symbol__",
        "rv_24h",
        "funding_abs_z",
        "ob_spread_bps",
        "xasset_ob_liquidity_peer_resid",
        "pct_assets_up_1h",
        "future_return_label",
        "target_soft",
        "ordinary_feature",
    ]
    selected = _feature_store_context_columns(columns, scope="config_meta_full")
    assert "rv_24h" in selected
    assert "funding_abs_z" in selected
    assert "ob_spread_bps" in selected
    assert "xasset_ob_liquidity_peer_resid" in selected
    assert "pct_assets_up_1h" in selected
    assert "future_return_label" not in selected
    assert "target_soft" not in selected
    assert "ordinary_feature" not in selected
