from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_trailing_regime_meta_handoff import (  # noqa: E402
    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
    BASE_TARGET_CONTRACT_HASH_COLUMN,
    HANDOFF_RANK_SCOPE_COLUMN,
    _append_frozen_ae_gmm_context,
    _contract_hash,
    _join_feature_store_context,
    _materialize_label_path_end,
    _materialize_promoted_base_contract,
    _train_meta_handoff,
    _feature_store_context_columns,
    run_handoff_only,
)
from scripts import report_s52_trailing_regime_meta_handoff as handoff_module  # noqa: E402
from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS  # noqa: E402


def test_materialize_label_path_end_uses_full_96_bar_15m_path(tmp_path: Path) -> None:
    label_context_dir = tmp_path / "labels"
    label_context_dir.mkdir()
    (label_context_dir / "side_archetype_trailing_materialization_summary.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"path_fetch": {"path_len": 96, "path_timeframe": "15m"}}
                ]
            }
        ),
        encoding="utf-8",
    )
    first_path = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")

    out, contract = _materialize_label_path_end(
        pd.DataFrame({"__first_path_ts__": [first_path]}),
        label_context_dir,
    )

    assert out["__label_path_end_ts__"].iloc[0] == first_path + pd.Timedelta(hours=24)
    assert contract["label_horizon_seconds"] == pytest.approx(24 * 60 * 60)
    assert contract["resolution_rule"] == "__first_path_ts__ + path_len * path_timeframe"


def test_materialize_label_path_end_rejects_inconsistent_path_contracts(
    tmp_path: Path,
) -> None:
    label_context_dir = tmp_path / "labels"
    label_context_dir.mkdir()
    (label_context_dir / "side_archetype_trailing_materialization_summary.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"path_fetch": {"path_len": 96, "path_timeframe": "15m"}},
                    {"path_fetch": {"path_len": 128, "path_timeframe": "15m"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="share one path resolution contract"):
        _materialize_label_path_end(
            pd.DataFrame(
                {
                    "__first_path_ts__": [
                        pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
                    ]
                }
            ),
            label_context_dir,
        )


def test_train_meta_handoff_preserves_complete_ae_gmm_registry() -> None:
    ledger = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-06-01", tz="UTC")],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "month": ["2026-06"],
            "score": [0.5],
            "selected_top30": [True],
            **{column: [0.0] for column in AE_GMM_FEATURE_COLUMNS},
        }
    )

    handoff, contract = _train_meta_handoff(
        ledger,
        pd.DataFrame(),
        [],
        selected_col="selected_top30",
    )

    assert set(AE_GMM_FEATURE_COLUMNS).issubset(handoff.columns)
    assert contract["ae_gmm_context_column_count"] == len(AE_GMM_FEATURE_COLUMNS)


def test_train_meta_handoff_records_explicit_base_target_weight_contract() -> None:
    target_contract = {
        "schema": "base_soft_label_contract_v1",
        "target_column": "__first_touch_target_soft__",
    }
    weight_spec = {
        "schema": "target_strength_weight_v1",
        "spec": {"exponent": 1.5, "weight_range_ratio": 4.0},
    }
    ledger = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-06-01", tz="UTC")],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "month": ["2026-06"],
            "score": [0.5],
            "selected_top30": [True],
            HANDOFF_RANK_SCOPE_COLUMN: ["timestamp_side"],
            "base_target_contract_json": [json.dumps(target_contract)],
            "base_sample_weight_spec_json": [json.dumps(weight_spec)],
            BASE_TARGET_CONTRACT_HASH_COLUMN: [_contract_hash(target_contract)],
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: [_contract_hash(weight_spec)],
        }
    )

    handoff, contract = _train_meta_handoff(
        ledger,
        pd.DataFrame(),
        [],
        selected_col="selected_top30",
        strict_base_contract=True,
    )

    inherited = contract["inherited_base_contract"]
    assert inherited["explicit_base_contract"] is True
    assert handoff[HANDOFF_RANK_SCOPE_COLUMN].eq("timestamp_side").all()
    assert handoff[BASE_TARGET_CONTRACT_HASH_COLUMN].iloc[0] == _contract_hash(
        target_contract
    )
    assert handoff[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN].iloc[0] == _contract_hash(
        weight_spec
    )


def test_materialize_incumbent_soft_label_weight_contract() -> None:
    ledger = pd.DataFrame(
        {
            "base_model_target_mode": ["target_soft", "target_soft"],
            "base_model_weight_arm": [
                "W7_timestamp_balanced",
                "W7_timestamp_balanced",
            ],
        }
    )

    out = _materialize_promoted_base_contract(ledger)

    target = json.loads(out["base_target_contract_json"].iloc[0])
    weight = json.loads(out["base_sample_weight_spec_json"].iloc[0])
    assert target["target_mode"] == "target_soft"
    assert weight["schema"] == "base_weight_arm_v1"
    assert weight["weight_arm"] == "W7_timestamp_balanced"
    assert out[BASE_TARGET_CONTRACT_HASH_COLUMN].eq(_contract_hash(target)).all()
    assert out[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN].eq(_contract_hash(weight)).all()


def test_feature_store_context_join_exports_safe_cross_market_columns(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features" / "20260601_000000"
    feature_dir.mkdir(parents=True)
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
    feature_dir = tmp_path / "features" / "20260601_000000"
    feature_dir.mkdir(parents=True)
    label_context_dir = tmp_path / "labels"
    label_context_dir.mkdir()
    (label_context_dir / "side_archetype_trailing_materialization_summary.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"path_fetch": {"path_len": 96, "path_timeframe": "15m"}}
                ]
            }
        ),
        encoding="utf-8",
    )
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
            "__first_path_ts__": [
                pd.Timestamp("2026-04-01 01:00:00", tz="UTC"),
                pd.Timestamp("2026-06-01 01:00:00", tz="UTC"),
            ],
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
        label_context_dir=label_context_dir,
        feature_dir=feature_dir,
        feature_store_scope="cross_market",
        fixed_ae_gmm_state_pkl=None,
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


def test_feature_store_join_reads_logical_delta_and_union_schema(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features" / "20260601_000000"
    feature_dir.mkdir(parents=True)
    ts = pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC", name="ts")
    symbol_a = feature_dir / "symbol=AAA_USD:USD.parquet"
    symbol_b = feature_dir / "symbol=BBB_USD:USD.parquet"
    pd.DataFrame(
        {"ordinary_feature": [1.0, 2.0], "__symbol__": ["AAA/USD:USD"] * 2},
        index=ts,
    ).to_parquet(symbol_a)
    pd.DataFrame(
        {"pct_assets_up_1h": [0.6, 0.7], "__symbol__": ["BBB/USD:USD"] * 2},
        index=ts,
    ).to_parquet(symbol_b)

    delta = pd.DataFrame(
        {
            "ts": ts,
            "pct_assets_up_1h": [0.4, 0.5],
            "__symbol__": ["AAA/USD:USD"] * 2,
        }
    )
    con = duckdb.connect(str(symbol_a) + ".deltas.duckdb")
    con.register("incoming", delta)
    con.execute("CREATE TABLE feature_deltas AS SELECT * FROM incoming")
    con.close()

    ledger = pd.DataFrame(
        {
            "__ts__": [ts[0].tz_localize(None), ts[1].tz_localize(None)] * 2,
            "__symbol__": ["AAA/USD:USD"] * 2 + ["BBB/USD:USD"] * 2,
            "side_name": ["long", "short", "long", "short"],
        }
    )
    joined, contract = _join_feature_store_context(ledger, feature_dir)

    assert contract["logical_store_reader"] == "static_feature_store.read_static_features"
    assert contract["store_access"] == "read_only"
    assert contract["match_rate"] == 1.0
    assert contract["source_signature"]["file_count"] == 3
    np.testing.assert_allclose(
        joined["pct_assets_up_1h"].to_numpy(),
        [0.4, 0.5, 0.6, 0.7],
        rtol=1e-6,
        atol=1e-7,
    )
    assert "ts" not in contract["loaded_columns"]


def test_feature_store_join_preserves_ledger_native_columns(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features" / "20260601_000000"
    feature_dir.mkdir(parents=True)
    ts = pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC", name="ts")
    pd.DataFrame(
        {
            "pct_assets_up_1h": [0.4, 0.5],
            "__symbol__": ["BTC/USD:USD"] * 2,
        },
        index=ts,
    ).to_parquet(feature_dir / "symbol=BTC_USD:USD.parquet")
    ledger = pd.DataFrame(
        {
            "__ts__": ts.tz_localize(None),
            "__symbol__": ["BTC/USD:USD"] * 2,
            "pct_assets_up_1h": [0.9, 0.8],
        }
    )

    joined, contract = _join_feature_store_context(ledger, feature_dir)

    assert joined["pct_assets_up_1h"].tolist() == [0.9, 0.8]
    assert "pct_assets_up_1h_x" not in joined.columns
    assert "pct_assets_up_1h_y" not in joined.columns
    assert "pct_assets_up_1h" in contract["skipped_existing_columns"]


def test_frozen_context_reuses_existing_outputs_without_missing_input_fill(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "state.pkl"
    joblib.dump(
        {
            "feature_columns": ["raw_a", "raw_b"],
            "cycle_input_fill_values": {"raw_a": 1.0, "raw_b": 2.0},
            "cycle_state_hash": "cycle-hash",
        },
        state_path,
    )

    def fake_transform(x, state, index):
        return pd.DataFrame({"gmm_entropy": 0.25}, index=index)

    monkeypatch.setattr(handoff_module, "transform_ae_gmm_features", fake_transform)
    frame = pd.DataFrame({"raw_a": [3.0, 4.0], "gmm_entropy": [0.1, 0.2]})

    out, contract = _append_frozen_ae_gmm_context(frame, state_path)

    assert out["gmm_entropy"].tolist() == [0.1, 0.2]
    assert contract["status"] == "existing_frozen_outputs_reused"
    assert contract["missing_state_input_features_not_required_for_reuse"] == ["raw_b"]
    assert contract["cycle_state_hash"] == "cycle-hash"


def test_frozen_context_rejects_incomplete_outputs_when_inputs_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "state.pkl"
    joblib.dump(
        {
            "feature_columns": ["raw_a", "raw_b"],
            "cycle_state_hash": "cycle-hash",
        },
        state_path,
    )

    def fake_transform(x, state, index):
        return pd.DataFrame({"gmm_entropy": 0.25}, index=index)

    monkeypatch.setattr(handoff_module, "transform_ae_gmm_features", fake_transform)
    frame = pd.DataFrame({"raw_a": [3.0, 4.0], "gmm_entropy": [0.1, None]})

    with pytest.raises(ValueError, match="incomplete frozen outputs"):
        _append_frozen_ae_gmm_context(frame, state_path)


def test_frozen_context_completes_only_missing_rows_with_cycle_fill(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "state.pkl"
    joblib.dump(
        {
            "feature_columns": ["raw_a", "raw_b"],
            "cycle_input_fill_values": {"raw_a": 1.0, "raw_b": 2.0},
            "cycle_state_hash": "cycle-hash",
        },
        state_path,
    )
    observed: list[pd.DataFrame] = []

    def fake_transform(x, state, index):
        observed.append(x.copy())
        return pd.DataFrame({"gmm_entropy": x["raw_b"] / 10.0}, index=index)

    monkeypatch.setattr(handoff_module, "transform_ae_gmm_features", fake_transform)
    frame = pd.DataFrame(
        {
            "raw_a": [3.0, 4.0],
            "raw_b": [5.0, None],
            "gmm_entropy": [0.1, None],
        }
    )

    out, contract = _append_frozen_ae_gmm_context(frame, state_path)

    assert out["gmm_entropy"].tolist() == pytest.approx([0.1, 0.2])
    assert observed[-1]["raw_b"].tolist() == [2.0]
    assert contract["status"] == "existing_frozen_outputs_completed"
    assert contract["existing_complete_rows"] == 1
    assert contract["recomputed_rows"] == 1
