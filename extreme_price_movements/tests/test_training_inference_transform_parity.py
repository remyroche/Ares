import json
import pickle

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_transform_contract import FeatureTransformContract
from extreme_price_movements.inference import feature_generator
from extreme_price_movements.model_loader import load_full_state
from extreme_price_movements.pipeline_steps import _drop_known_unusable_raw_feature_keys


def _contract() -> FeatureTransformContract:
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    panels = {
        "ret24h": pd.DataFrame({"AAA/USDC": [0.0, 0.1, 0.2, 0.3]}, index=idx),
        "barrier_pct": pd.DataFrame({"AAA/USDC": [0.02, 0.02, 0.02, 0.02]}, index=idx),
    }
    return FeatureTransformContract.fit_from_panels(
        panels,
        {
            "market_mode": "spot",
            "feature_transform_kind": "standard",
            "feature_transform_clip_quantiles": [0.0, 1.0],
        },
        "run_a",
        {
            "stage_name": "train_base",
            "symbols": ["AAA/USDC"],
            "allowed_start_ts": idx.min().isoformat(),
            "allowed_end_ts": idx.max().isoformat(),
        },
    )


def test_inference_transform_path_does_not_fit(monkeypatch):
    contract = _contract()
    raw = {
        "ret24h": pd.DataFrame(
            {"AAA/USDC": [0.15]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 04:00", tz="UTC")]),
        ),
        "barrier_pct": pd.DataFrame(
            {"AAA/USDC": [0.02]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 04:00", tz="UTC")]),
        ),
    }

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("inference must not fit transforms")

    monkeypatch.setattr(FeatureTransformContract, "fit_from_panels", forbidden_fit)
    transformed = feature_generator._transform_feature_panels_for_inference(
        raw,
        {"feature_transform_contract": contract},
        strict=True,
        label="test",
    )

    assert "ret24h" in transformed
    assert transformed["ret24h"].iloc[0, 0] != raw["ret24h"].iloc[0, 0]
    assert np.isclose(transformed["barrier_pct"].iloc[0, 0], raw["barrier_pct"].iloc[0, 0])


def test_live_snapshot_cache_hash_mismatch_is_cache_miss(tmp_path):
    run_id = "run_a"
    cache_key = "cache-key"
    cfg = {
        "live_feature_snapshot_cache_enabled": True,
        "live_feature_snapshot_cache_dir": str(tmp_path),
        "feature_transform_contract_hash": "sha256:expected",
    }
    symbols = ["AAA/USDC"]
    required = {"ret24h"}
    end_ts = pd.Timestamp("2026-01-01 04:00", tz="UTC")
    cache_dir = feature_generator._feature_snapshot_dir(cfg, run_id, cache_key)
    cache_dir.mkdir(parents=True)
    pd.DataFrame({"ret24h": [0.1]}, index=symbols).to_parquet(cache_dir / "latest.parquet")
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "version": feature_generator.LIVE_FEATURE_CACHE_VERSION,
                "cache_key": cache_key,
                "symbols_hash": feature_generator._hash_values(symbols),
                "required_hash": feature_generator._hash_values(required),
                "contract_hash": "sha256:wrong",
                "end_ts": end_ts.isoformat(),
            }
        )
    )

    loaded = feature_generator._load_live_feature_snapshot(
        cfg=cfg,
        run_id=run_id,
        cache_key=cache_key,
        symbols=symbols,
        end_ts=end_ts,
        required_feature_keys=required,
    )

    assert loaded == {}


def test_trained_state_embeds_contract_for_model_loader(tmp_path):
    contract = _contract()
    manifest = {"contract_hash": contract.contract_hash}
    run_id = "run_a"
    model_dir = tmp_path / "artifacts" / run_id / "models"
    model_dir.mkdir(parents=True)
    state = {
        "ts_trained": run_id,
        "bundle": {"alpha_models": {}, "meta_models": {}},
        "risk_params": {},
        "feature_transform_contract": contract,
        "feature_transform_manifest": manifest,
        "feature_transform_contract_hash": contract.contract_hash,
    }
    with (model_dir / "trained_state.pkl").open("wb") as f:
        pickle.dump(state, f)

    loaded = load_full_state(run_id, str(tmp_path))

    assert loaded["feature_transform_contract_hash"] == contract.contract_hash
    assert loaded["bundle"]["feature_transform_contract"].contract_hash == contract.contract_hash
    assert loaded["bundle"]["feature_transform_manifest"] == manifest


def test_raw_compute_cfg_marks_contract_raw_mode():
    contract = _contract()
    cfg = feature_generator._raw_feature_compute_cfg({"feature_transform_contract": contract})

    assert cfg["feature_transform_contract_raw_mode"] is True


def test_transform_matrix_values_are_replayable_after_cache_rejection():
    contract = _contract()
    matrix = pd.DataFrame(
        {"barrier_pct": [0.02], "ret24h": [0.15]},
        index=["AAA/USDC"],
    )
    shuffled = matrix[["ret24h", "barrier_pct"]]

    a = contract.transform_matrix(matrix, strict=True)
    b = contract.transform_matrix(shuffled, strict=True)

    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_raw_feature_snapshot_drops_unnormalized_orderbook_duplicates():
    idx = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    features = {
        "ob_spread_bps": pd.DataFrame({"AAA/USDC": [1.0, 2.0]}, index=idx),
        "ob_spread_bps_z_24h": pd.DataFrame({"AAA/USDC": [0.1, 0.2]}, index=idx),
        "ob_depth_usd_l20": pd.DataFrame({"AAA/USDC": [1000.0, 1200.0]}, index=idx),
        "ob_depth_usd_l20_z": pd.DataFrame({"AAA/USDC": [0.0, 0.1]}, index=idx),
    }

    cleaned = _drop_known_unusable_raw_feature_keys(features, label="test")

    assert "ob_spread_bps" not in cleaned
    assert "ob_depth_usd_l20" not in cleaned
    assert "ob_spread_bps_z_24h" in cleaned
    assert "ob_depth_usd_l20_z" in cleaned
