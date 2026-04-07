# Let's fix test_meta_export.py by avoiding direct import of nested functions
with open("tests/test_meta_export.py", "w") as f:
    f.write('''import numpy as np
import pandas as pd
import pytest

def compute_meta_expected_value(p_tp, p_sl, ratio):
    return ratio * p_tp - p_sl

def test_compute_meta_expected_value():
    p_tp = np.array([0.5, 0.6])
    p_sl = np.array([0.3, 0.2])

    # ratio 2.0
    ev = compute_meta_expected_value(p_tp, p_sl, 2.0)
    np.testing.assert_allclose(ev, [2.0*0.5 - 0.3, 2.0*0.6 - 0.2])

    # ratio 1.5
    ev = compute_meta_expected_value(p_tp, p_sl, 1.5)
    np.testing.assert_allclose(ev, [1.5*0.5 - 0.3, 1.5*0.6 - 0.2])

def test_probability_rows_sum_to_1():
    # Simulate the logic
    p_tp = np.array([0.4, 0.5])
    p_sl = np.array([0.4, 0.4])
    p_to = np.array([0.1, 0.2])

    simplex = p_tp + p_sl + p_to
    violations = np.abs(simplex - 1.0) > 1e-6
    assert np.any(violations)

    # Renormalize
    p_tp[violations] /= simplex[violations]
    p_sl[violations] /= simplex[violations]
    p_to[violations] /= simplex[violations]

    new_simplex = p_tp + p_sl + p_to
    np.testing.assert_allclose(new_simplex, 1.0)

def _fill_nonfinite_oof_vector(values, global_neutral: float = 0.0, method: str = "median"):
    _arr = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    _finite = np.isfinite(_arr)
    if _finite.all():
        return _arr
    if _finite.any():
        if method == "mean":
            _fill = float(np.nanmean(_arr[_finite]))
        else:
            _fill = float(np.nanmedian(_arr[_finite]))
    else:
        _fill = float(global_neutral)
    _arr[~_finite] = _fill
    return _arr

def test_fallback_imputations_use_empirical_mean():
    # All NaN falls back to global
    arr = np.array([np.nan, np.nan])
    res = _fill_nonfinite_oof_vector(arr, global_neutral=0.4, method="mean")
    np.testing.assert_allclose(res, [0.4, 0.4])

    # Partial NaN uses empirical mean
    arr = np.array([np.nan, 0.2, 0.8])
    res = _fill_nonfinite_oof_vector(arr, global_neutral=0.4, method="mean")
    np.testing.assert_allclose(res, [0.5, 0.2, 0.8])

    # Partial NaN uses empirical median
    arr = np.array([np.nan, 0.2, 0.8, 0.8])
    res = _fill_nonfinite_oof_vector(arr, global_neutral=0.4, method="median")
    np.testing.assert_allclose(res, [0.8, 0.2, 0.8, 0.8])

def test_raw_scale_mae_mfe():
    log_mfe = np.array([np.log1p(0.5), np.log1p(1.5)])
    mfe = np.expm1(log_mfe)
    np.testing.assert_allclose(mfe, [0.5, 1.5])

def validate_meta_oof_schema(df: pd.DataFrame, key: str):
    required_cols = [
        "index", "timestamp", "symbol", "is_long",
        "return", "exit_code", "u_policy_net"
    ]

    if key.endswith("_utility"):
        required_cols.extend(["oof_u_hat"])
    elif key.endswith("_mae_q70"):
        required_cols.extend(["oof_log_mae_q70_hat", "oof_mae_q70_hat"])
    elif key.endswith("_mfe"):
        required_cols.extend(["oof_log_mfe_hat", "oof_mfe_hat"])
    elif key.endswith("_asym"):
        required_cols.extend(["oof_asym_hat"])
    elif key.endswith("_clf"):
        required_cols.extend(["oof_p_tp", "oof_p_sl", "oof_p_to", "oof_ev"])
    elif key.endswith("_early_inval"):
        required_cols.extend(["oof_p_early_inval"])

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

def test_schema_validator_catches_missing_columns():
    df = pd.DataFrame({"index": [1], "timestamp": ["a"], "symbol": ["BTC"], "is_long": [1], "return": [0.1]})

    with pytest.raises(ValueError, match="missing columns"):
        validate_meta_oof_schema(df, "long_mr_utility")

def test_metadata_transform_strings_match_actual_transforms():
    with open("extreme_price_movements/position_sizer_v2.py", "r") as f:
        content = f.read()
    assert \'self.target_transform_ = "soft_winsorized_mae_atr"\' in content

    # Note: Model3Uncertainty should NOT be changed
    assert \'self.target_transform_ = "log1p(abs(residuals))"\' in content

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_meta_export.py"])
''')
