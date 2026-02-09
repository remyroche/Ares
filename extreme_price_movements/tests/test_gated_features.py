import numpy as np
import pandas as pd

from extreme_price_movements.gated_features import (
    add_accept_gate_features,
    conditional_uplift_by_bin,
    cross_sectional_gate_aggregates,
    gate_stability_diagnostics,
)


def test_add_accept_gate_features_emits_expected_columns_and_warmup_defaults():
    idx = pd.date_range("2024-01-01", periods=32, freq="h")
    df = pd.DataFrame({"accept_score": np.linspace(-2.0, 2.0, len(idx), dtype=np.float32)}, index=idx)

    out = add_accept_gate_features(df.copy(), N=8)

    expected = {
        "s_mean_8",
        "s_std_8",
        "s_z_8",
        "s_pct_8",
        "s_bin3_8",
        "s_gt66_8",
        "s_gt85_8",
    }
    assert expected.issubset(out.columns)

    warmup = out.iloc[:8]
    assert np.allclose(warmup["s_z_8"].to_numpy(), 0.0)
    assert np.allclose(warmup["s_pct_8"].to_numpy(), 0.5)


def test_add_accept_gate_features_is_causal_shifted():
    idx = pd.date_range("2024-01-01", periods=40, freq="h")
    base = np.sin(np.linspace(0, 3, len(idx))).astype(np.float32)

    df_a = pd.DataFrame({"accept_score": base.copy()}, index=idx)
    df_b = pd.DataFrame({"accept_score": base.copy()}, index=idx)

    # Perturb only final value; shifted rolling stats should prevent any earlier-row impact.
    df_b.iloc[-1, 0] = 1_000.0

    out_a = add_accept_gate_features(df_a, N=12)
    out_b = add_accept_gate_features(df_b, N=12)

    for col in ["s_mean_12", "s_std_12", "s_z_12", "s_pct_12", "s_bin3_12", "s_gt66_12", "s_gt85_12"]:
        assert np.allclose(out_a[col].iloc[:-1].to_numpy(), out_b[col].iloc[:-1].to_numpy())


def test_cross_sectional_gate_aggregates_robust_outputs():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    x = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [1, 2, 30, 4, 5],
        "c": [1, 2, 3, 4, 5],
    }, index=idx, dtype=np.float32)

    out = cross_sectional_gate_aggregates(x)
    assert {"cs_median", "cs_trimmed_mean", "cs_p75", "cs_p90", "cs_iqr", "cs_std"}.issubset(out.columns)
    assert np.isfinite(out.to_numpy()).all()


def test_gate_diagnostics_helpers_emit_expected_shapes():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    s = pd.Series(np.linspace(0, 1, len(idx), dtype=np.float32), index=idx)
    bins = pd.Series(np.digitize(s.to_numpy(), bins=[1 / 3, 2 / 3]), index=idx)

    stab = gate_stability_diagnostics(s, bins)
    assert "lag1_autocorr" in stab
    assert "bin3_share_0" in stab

    pred = s
    ret = pd.Series(np.sin(np.linspace(0, 3, len(idx))), index=idx)
    uplift = conditional_uplift_by_bin(pred, ret, bins)
    assert set(["bin", "n", "ic", "prec10_ret"]).issubset(uplift.columns)
