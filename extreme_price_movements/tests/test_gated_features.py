import numpy as np
import pandas as pd

from extreme_price_movements.gated_features import (
    add_accept_gate_features,
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
        "s_gt75_8",
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

    for col in ["s_mean_12", "s_std_12", "s_z_12", "s_pct_12", "s_bin3_12", "s_gt66_12", "s_gt75_12"]:
        assert np.allclose(out_a[col].iloc[:-1].to_numpy(), out_b[col].iloc[:-1].to_numpy())


