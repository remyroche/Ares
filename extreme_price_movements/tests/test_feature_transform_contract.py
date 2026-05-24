import pickle

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_transform_contract import (
    FeatureTransformContract,
    load_feature_transform_contract,
    save_feature_transform_contract,
)


def _panels() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    return {
        "ret24h": pd.DataFrame(
            {"AAA/USDC": [0.0, 0.1, np.nan, 0.3, 10.0], "BBB/USDC": [0.2, 0.3, 0.4, np.inf, 0.6]},
            index=idx,
        ),
        "range_24h_pct": pd.DataFrame(
            {"AAA/USDC": [0.01, 0.02, 0.03, 0.04, 0.05], "BBB/USDC": [0.02, 0.03, 0.04, 0.05, 0.06]},
            index=idx,
        ),
        "barrier_pct": pd.DataFrame(
            {"AAA/USDC": [0.02, 0.02, 0.02, 0.02, 0.02], "BBB/USDC": [0.03, 0.03, 0.03, 0.03, 0.03]},
            index=idx,
        ),
        "atr_pct_raw": pd.DataFrame(
            {"AAA/USDC": [0.01] * 5, "BBB/USDC": [0.02] * 5},
            index=idx,
        ),
    }


def _cfg() -> dict:
    return {
        "market_mode": "spot",
        "feature_transform_kind": "robust",
        "feature_transform_clip_quantiles": [0.1, 0.9],
        "feature_transform_impute": "median",
    }


def _fit_scope() -> dict:
    return {
        "stage_name": "train_base",
        "symbols": ["AAA/USDC", "BBB/USDC"],
        "allowed_start_ts": "2026-01-01T00:00:00+00:00",
        "allowed_end_ts": "2026-01-01T04:00:00+00:00",
    }


def test_contract_hash_is_deterministic():
    a = FeatureTransformContract.fit_from_panels(_panels(), _cfg(), "run_a", _fit_scope())
    b = FeatureTransformContract.fit_from_panels(_panels(), _cfg(), "run_a", _fit_scope())

    assert a.contract_hash == b.contract_hash


def test_transform_replay_round_trip(tmp_path):
    panels = _panels()
    contract = FeatureTransformContract.fit_from_panels(panels, _cfg(), "run_a", _fit_scope())
    expected = contract.transform_panels(panels, strict=True)

    save_feature_transform_contract(contract, tmp_path, "run_a")
    loaded, manifest = load_feature_transform_contract(tmp_path, "run_a")
    actual = loaded.transform_panels(panels, strict=True)

    assert manifest["contract_hash"] == contract.contract_hash
    assert list(actual) == list(expected)
    for key in expected:
        assert list(actual[key].columns) == list(expected[key].columns)
        assert np.allclose(actual[key].to_numpy(), expected[key].to_numpy(), equal_nan=True)


def test_contract_imputes_nonfinite_before_scaling():
    panels = _panels()
    contract = FeatureTransformContract.fit_from_panels(panels, _cfg(), "run_a", _fit_scope())
    transformed = contract.transform_panels(panels, strict=True)

    ret = transformed["ret24h"].to_numpy()
    assert np.isfinite(ret).all()
    assert transformed["ret24h"].loc["2026-01-01 02:00:00+00:00", "AAA/USDC"] == pytest.approx(0.0)
    assert transformed["ret24h"].loc["2026-01-01 03:00:00+00:00", "BBB/USDC"] == pytest.approx(0.0)


def test_transform_matrix_column_order_parity():
    panels = _panels()
    contract = FeatureTransformContract.fit_from_panels(panels, _cfg(), "run_a", _fit_scope())
    row = pd.DataFrame(
        {
            "range_24h_pct": [0.04],
            "barrier_pct": [0.02],
            "ret24h": [0.3],
            "atr_pct_raw": [0.01],
        },
        index=["AAA/USDC"],
    )

    expected = contract.transform_matrix(row, strict=True)
    actual = contract.transform_matrix(row.loc[:, list(reversed(row.columns))], strict=True)

    assert list(actual.columns) == contract.transformed_feature_cols
    assert np.allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_transform_matrix_refuses_nonfinite_contract_values():
    panels = _panels()
    contract = FeatureTransformContract.fit_from_panels(panels, _cfg(), "run_a", _fit_scope())
    row = pd.DataFrame(
        {
            "range_24h_pct": [0.04],
            "barrier_pct": [0.02],
            "ret24h": [np.nan],
            "atr_pct_raw": [0.01],
        },
        index=["AAA/USDC"],
    )

    with pytest.raises(ValueError, match="non-finite contracted raw columns"):
        contract.transform_matrix(row, strict=True)

    permissive = contract.transform_matrix(row, strict=True, require_finite=False)
    assert np.isfinite(permissive.to_numpy()).all()


def test_missing_feature_strictness():
    panels = _panels()
    contract = FeatureTransformContract.fit_from_panels(panels, _cfg(), "run_a", _fit_scope())
    incomplete = dict(panels)
    incomplete.pop("ret24h")

    with pytest.raises(KeyError):
        contract.transform_panels(incomplete, strict=True)

    transformed = contract.transform_panels(incomplete, strict=False)
    assert "ret24h" not in transformed


def test_raw_passthrough_preservation():
    panels = _panels()
    contract = FeatureTransformContract.fit_from_panels(panels, _cfg(), "run_a", _fit_scope())
    transformed = contract.transform_panels(panels, strict=True)

    assert np.allclose(transformed["barrier_pct"], panels["barrier_pct"])
    assert np.allclose(transformed["atr_pct_raw"], panels["atr_pct_raw"])


def test_contract_is_pickle_safe():
    contract = FeatureTransformContract.fit_from_panels(_panels(), _cfg(), "run_a", _fit_scope())
    loaded = pickle.loads(pickle.dumps(contract))

    assert loaded.contract_hash == contract.contract_hash
    assert loaded.transform_config == contract.transform_config
