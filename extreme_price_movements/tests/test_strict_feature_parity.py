import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_transform_contract import FeatureTransformContract
from extreme_price_movements.inference.feature_parity import (
    FeatureParityError,
    required_source_groups_for_features,
    validate_feature_parity_before_prediction,
    validate_final_model_matrix,
    validate_model_bundle_transform_contract,
    validate_required_source_panels,
    validate_raw_history_sufficiency,
    validate_transformed_feature_panels,
    validate_transform_stat_completeness,
)


def _contract(required_warmup_hours: int = 2) -> FeatureTransformContract:
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    panels = {
        "ret24h": pd.DataFrame(
            {"BTC/USDC": [0.0, 0.1, 0.2, 0.3], "SOL/USDC": [0.2, 0.3, 0.4, 0.5]},
            index=idx,
        ),
        "rv_24h": pd.DataFrame(
            {"BTC/USDC": [0.1, 0.2, 0.3, 0.4], "SOL/USDC": [0.2, 0.3, 0.4, 0.5]},
            index=idx,
        ),
    }
    contract = FeatureTransformContract.fit_from_panels(
        panels,
        {
            "market_mode": "spot",
            "feature_transform_kind": "robust",
            "feature_transform_clip_quantiles": [0.0, 1.0],
            "feature_transform_required_warmup_hours": required_warmup_hours,
        },
        "run_a",
        {
            "stage_name": "train_base",
            "symbols": ["BTC/USDC", "SOL/USDC"],
            "allowed_start_ts": idx.min().isoformat(),
            "allowed_end_ts": idx.max().isoformat(),
        },
    )
    return contract


def _cfg(scope: str = "symbol") -> dict:
    return {
        "strict_feature_parity": True,
        "strict_feature_parity_scope": scope,
        "market_mode": "spot",
        "feature_parity_require_current_timestamp": True,
    }


def test_missing_contract_refuses_inference():
    with pytest.raises(FeatureParityError):
        validate_model_bundle_transform_contract({}, _cfg(), "run_a")


def test_contract_hash_mismatch_refuses_inference():
    contract = _contract()
    with pytest.raises(FeatureParityError):
        validate_model_bundle_transform_contract(
            {
                "feature_transform_contract": contract,
                "feature_transform_contract_hash": "sha256:wrong",
            },
            _cfg(),
            "run_a",
        )


def test_missing_required_feature_refuses_inference():
    contract = _contract()
    feats = {"ret24h": pd.DataFrame({"BTC/USDC": [0.1]}, index=[pd.Timestamp("2026-01-01 03:00", tz="UTC")])}
    with pytest.raises(FeatureParityError):
        validate_feature_parity_before_prediction(
            feats=feats,
            contract=contract,
            symbols=["BTC/USDC"],
            end_ts=pd.Timestamp("2026-01-01 03:00", tz="UTC"),
            required_feature_keys={"ret24h", "rv_24h"},
            cfg=_cfg(),
        )


def test_missing_transform_stats_refuses_inference():
    contract = _contract()
    contract.per_column_stats.pop("rv_24h")
    with pytest.raises(FeatureParityError):
        validate_transform_stat_completeness(contract, {"rv_24h"}, strict=True)


def test_insufficient_raw_history_refuses_symbol():
    contract = _contract(required_warmup_hours=500)
    idx = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    panel = {"close": pd.DataFrame({"BTC/USDC": np.arange(100.0)}, index=idx)}
    with pytest.raises(FeatureParityError):
        validate_raw_history_sufficiency(
            panel,
            contract,
            ["BTC/USDC"],
            idx.max(),
            {"ret24h"},
            cfg=_cfg("symbol"),
        )


def test_one_bad_symbol_does_not_block_good_symbol_in_symbol_scope():
    contract = _contract(required_warmup_hours=2)
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    feats = {
        "ret24h": pd.DataFrame({"BTC/USDC": [0.1, 0.2, 0.3, 0.4]}, index=idx),
        "rv_24h": pd.DataFrame({"BTC/USDC": [0.1, 0.2, 0.3, 0.4]}, index=idx),
    }
    report = validate_transformed_feature_panels(
        feats,
        contract,
        ["BTC/USDC", "SOL/USDC"],
        idx.max(),
        {"ret24h", "rv_24h"},
        cfg=_cfg("symbol"),
    )
    assert report["accepted_symbols"] == ["BTC/USDC"]
    assert report["rejected_symbols"] == ["SOL/USDC"]


def test_global_scope_blocks_all_on_one_bad_symbol():
    contract = _contract(required_warmup_hours=2)
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    feats = {
        "ret24h": pd.DataFrame({"BTC/USDC": [0.1, 0.2, 0.3, 0.4]}, index=idx),
        "rv_24h": pd.DataFrame({"BTC/USDC": [0.1, 0.2, 0.3, 0.4]}, index=idx),
    }
    with pytest.raises(FeatureParityError):
        validate_transformed_feature_panels(
            feats,
            contract,
            ["BTC/USDC", "SOL/USDC"],
            idx.max(),
            {"ret24h", "rv_24h"},
            cfg=_cfg("global"),
        )


def test_mark_index_feature_requires_real_mark_and_reference_source():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    panel = {
        "close": pd.DataFrame({"BTC/USD:USD": [100.0, 101.0, 102.0, 103.0]}, index=idx),
        "mark_price": pd.DataFrame({"BTC/USD:USD": [100.1, 101.1, 102.1, 103.1]}, index=idx),
    }
    with pytest.raises(FeatureParityError):
        validate_required_source_panels(
            panel,
            ["BTC/USD:USD"],
            idx.max(),
            {"mark_index_basis"},
            cfg=_cfg(),
        )

    panel["spot_close"] = pd.DataFrame(
        {"BTC/USD:USD": [99.9, 100.9, 101.9, 102.9]}, index=idx
    )
    report = validate_required_source_panels(
        panel,
        ["BTC/USD:USD"],
        idx.max(),
        {"mark_index_basis"},
        cfg=_cfg(),
    )
    assert report["accepted_symbols"] == ["BTC/USD:USD"]
    assert report["required_source_groups"].keys() >= {"mark", "spot_or_index"}


def test_index_source_can_satisfy_cross_reference_without_spot():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    panel = {
        "close": pd.DataFrame({"ETH/USD:USD": [100.0, 101.0, 102.0, 103.0]}, index=idx),
        "index_price": pd.DataFrame({"ETH/USD:USD": [99.8, 100.8, 101.8, 102.8]}, index=idx),
    }
    report = validate_required_source_panels(
        panel,
        ["ETH/USD:USD"],
        idx.max(),
        {"basis_pct"},
        cfg=_cfg(),
    )
    assert report["accepted_symbols"] == ["ETH/USD:USD"]


def test_spot_specific_feature_requires_spot_not_index():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    panel = {
        "close": pd.DataFrame({"SOL/USD:USD": [100.0, 101.0, 102.0, 103.0]}, index=idx),
        "index_price": pd.DataFrame({"SOL/USD:USD": [99.8, 100.8, 101.8, 102.8]}, index=idx),
    }
    with pytest.raises(FeatureParityError):
        validate_required_source_panels(
            panel,
            ["SOL/USD:USD"],
            idx.max(),
            {"spot_ret_24h"},
            cfg=_cfg(),
        )


def test_stale_source_is_rejected():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    panel = {
        "mark_price": pd.DataFrame(
            {"BTC/USD:USD": [100.1, np.nan, np.nan, np.nan]}, index=idx
        )
    }
    with pytest.raises(FeatureParityError):
        validate_required_source_panels(
            panel,
            ["BTC/USD:USD"],
            idx.max(),
            {"mark_price"},
            cfg={**_cfg(), "feature_source_max_staleness_hours": {"mark": 1.0}},
        )


def test_source_group_mapping_covers_basis_mark_funding_oi():
    groups = required_source_groups_for_features(
        {"mark_index_basis", "basis_funding_div_2h", "oi_chg_2h"}
    )
    assert groups.keys() >= {"mark", "spot_or_index", "funding", "open_interest"}


def test_vwap_and_oi_rel_vol_require_perp_volume_source():
    groups = required_source_groups_for_features(
        {"dist_vwap_norm", "dist_vwap_24_atr", "trapped_longs_96", "vwap_zone_1d_atr", "dist_stack", "oi_rel_vol_2h"}
    )
    assert groups.keys() >= {"perp_volume"}
    assert "open_interest" in groups
    assert "oi_rel_vol_2h" in groups["open_interest"]


def test_nonfinite_transformed_value_refuses_inference():
    contract = _contract()
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    feats = {
        "ret24h": pd.DataFrame({"BTC/USDC": [0.1, 0.2, 0.3, np.nan]}, index=idx),
        "rv_24h": pd.DataFrame({"BTC/USDC": [0.1, 0.2, 0.3, 0.4]}, index=idx),
    }
    with pytest.raises(FeatureParityError):
        validate_transformed_feature_panels(
            feats,
            contract,
            ["BTC/USDC"],
            idx.max(),
            {"ret24h", "rv_24h"},
            cfg=_cfg(),
        )


def test_final_model_matrix_column_order_enforced():
    X = pd.DataFrame({"b": [1.0], "a": [2.0]})
    with pytest.raises(FeatureParityError):
        validate_final_model_matrix(X, ["a", "b"], "model", strict=True)


def test_final_model_matrix_accepts_exact_finite_float32():
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = validate_final_model_matrix(X, ["a", "b"], "model", strict=True)
    assert out.dtypes.tolist() == [np.dtype("float32"), np.dtype("float32")]
