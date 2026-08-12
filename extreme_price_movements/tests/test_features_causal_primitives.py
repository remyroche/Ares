import ast
import inspect
import textwrap

import numpy as np
import pandas as pd

from extreme_price_movements import features


def _frame(values, *, name="AAA/USDT") -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({name: values}, index=index, dtype=np.float32)


def test_volume_gap_validation_is_prefix_invariant_and_never_interpolates():
    volume = _frame([10.0, np.nan, 0.0, 20.0])
    open_ = _frame([100.0, 101.0, 102.0, 103.0])
    close = _frame([100.0, 102.0, 102.0, 104.0])

    prefix = features._backfill_short_volume_gaps(
        volume.iloc[:3], open_.iloc[:3], close.iloc[:3]
    )
    extended = features._backfill_short_volume_gaps(volume, open_, close)

    pd.testing.assert_frame_equal(prefix, extended.iloc[:3])
    assert np.isnan(extended.iloc[1, 0])
    assert extended.iloc[2, 0] == 0.0


def test_volume_gap_validation_does_not_read_the_next_candle():
    volume = _frame([10.0, 0.0, 0.0, 20.0])
    open_ = _frame([100.0, 101.0, 101.0, 103.0])
    close_a = _frame([100.0, 101.0, 101.0, 104.0])
    close_b = _frame([100.0, 101.0, 150.0, 104.0])

    out_a = features._backfill_short_volume_gaps(volume, open_, close_a)
    out_b = features._backfill_short_volume_gaps(volume, open_, close_b)

    # The no-trade zero at t=1 is determined from the candle at t=1 only.
    assert out_a.iloc[1, 0] == 0.0
    assert out_b.iloc[1, 0] == 0.0


def test_fixed_ffd_uses_declared_d_and_nulls_immature_history():
    close = _frame([100.0, 101.0])

    out = features._transform_close_fixed_ffd(
        close, d=0.4, _label="test_fixed_d", thres=0.1
    )

    # d=0.4 / threshold 0.1 needs three observations.  A short prefix may
    # not silently substitute an alternate d or raw EWMA value.
    assert out.isna().all().all()


def test_fixed_ffd_is_prefix_invariant_after_declared_warmup():
    values = np.linspace(100.0, 110.0, 12, dtype=np.float32)
    prefix = _frame(values[:7])
    extended = _frame(values)

    out_prefix = features._transform_close_fixed_ffd(
        prefix, d=0.4, _label="test_prefix", thres=0.1
    )
    out_extended = features._transform_close_fixed_ffd(
        extended, d=0.4, _label="test_extended", thres=0.1
    )

    pd.testing.assert_frame_equal(out_prefix, out_extended.iloc[: len(prefix)])
    assert out_prefix.iloc[:2].isna().all().all()
    assert np.isfinite(out_prefix.iloc[2:].to_numpy()).all()


def test_canonical_technical_features_keep_warmup_and_degenerate_windows_missing():
    """Unavailable technical state must not be converted to a neutral value."""
    close = _frame(np.full(70, 100.0, dtype=np.float32))

    bb = features._canonical_bollinger_band_width(np.log(close))
    autocorr = features._rolling_autocorr_df(np.log(close).diff(), 48)

    assert bb.iloc[:19].isna().all().all()
    # A complete flat Bollinger window is a valid zero width.
    assert np.allclose(bb.iloc[19:].to_numpy(), 0.0, equal_nan=False)
    # A constant return window has undefined correlation, rather than neutral
    # autocorrelation, even after the nominal lookback.
    assert autocorr.isna().all().all()


def test_technical_source_readiness_audit_separates_source_loss_from_warmup():
    close = _frame(np.linspace(100.0, 120.0, 30, dtype=np.float32))
    high = close + 1.0
    low = close - 1.0
    volume = _frame(np.linspace(10.0, 20.0, 30, dtype=np.float32))
    volume.iloc[28, 0] = np.nan
    panel = {"close": close, "high": high, "low": low, "volume": volume}
    output = features._canonical_bollinger_band_width(np.log(close))
    audit = features.technical_feature_source_readiness_audit(
        panel,
        outputs={"bollinger_band_width": output},
        feature_keys=["bollinger_band_width", "volume_z_24"],
    ).set_index("feature")

    assert audit.loc["bollinger_band_width", "source_missing_rows"] == 0
    assert audit.loc["bollinger_band_width", "warmup_rows"] == 20
    assert audit.loc["bollinger_band_width", "post_warmup_output_missing_rows"] == 0
    assert audit.loc["volume_z_24", "source_missing_rows"] == 1
    assert audit.loc["volume_z_24", "warmup_rows"] == 24


def test_public_technical_features_have_one_generator_assignment():
    """Position-sizer code must not silently overwrite canonical features."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(features._compute_features_impl)))
    target_names = {
        "choppiness_index_20",
        "atr_change_rate",
        "direction_entropy_20",
        "acceleration_of_move",
        "volume_zscore_48h",
        "variance_ratio_10_48",
        "volume_trend_48",
        "volatility_of_volatility_48",
        "trend_acceleration",
        "volatility_autocorr_48",
    }
    assignments = {name: 0 for name in target_names}
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            for subscript in ast.walk(target):
                if not isinstance(subscript, ast.Subscript):
                    continue
                if not isinstance(subscript.value, ast.Name) or subscript.value.id != "feats":
                    continue
                if isinstance(subscript.slice, ast.Constant) and subscript.slice.value in assignments:
                    assignments[subscript.slice.value] += 1
    assert assignments == {name: 1 for name in target_names}


def test_unsupported_position_sizer_public_outputs_are_not_generated():
    source = inspect.getsource(features._compute_features_impl)
    for name in ("MACD_histogram", "RSI", "bars_since_trend_flip", "dist_ema100_atr"):
        assert f'feats["{name}"]' not in source


def test_funding_cadence_is_prefix_invariant_and_never_reads_next_event():
    index = pd.date_range("2025-01-01", periods=7, freq="h", tz="UTC")
    prefix_source = pd.Series(
        [np.nan, 0.01, 0.01, 0.02, 0.02], index=index[:5], dtype=np.float32
    )
    extended_source = pd.Series(
        [np.nan, 0.01, 0.01, 0.02, 0.02, 0.50, 0.50],
        index=index,
        dtype=np.float32,
    )

    last_prefix, cadence_prefix = features._causal_funding_schedule_from_source(
        prefix_source, default_interval_hours=8.0
    )
    last_extended, cadence_extended = features._causal_funding_schedule_from_source(
        extended_source, default_interval_hours=8.0
    )

    pd.testing.assert_series_equal(last_prefix, last_extended.iloc[: len(prefix_source)])
    pd.testing.assert_series_equal(
        cadence_prefix, cadence_extended.iloc[: len(prefix_source)]
    )
    # At t=4 the observed two-hour cadence schedules t=5; the later realised
    # event must not be consulted to obtain that answer.
    scheduled_next = last_prefix.iloc[4] + pd.to_timedelta(cadence_prefix.iloc[4], unit="h")
    assert scheduled_next == index[5]


def test_technical_entropy_change_points_and_trend_r2_fail_closed_when_undefined():
    """Warm-up and flat technical windows must remain unavailable, never neutral."""
    flat = _frame(np.ones(128, dtype=np.float32))

    # A flat but mature path may legitimately have low entropy.  Only the
    # immature prefix must be missing instead of the old synthetic 0.5.
    assert features._rolling_shannon_entropy_df(flat, 16).iloc[:16].isna().all().all()
    assert features._rolling_permutation_entropy_df(flat, 24).iloc[:24].isna().all().all()
    assert features._rolling_spectral_entropy_df(flat, 24).iloc[:24].isna().all().all()

    r2 = features._rolling_trend_r2_df(flat, 24)
    assert r2.isna().all().all()

    cp = features._short_long_change_point_features(
        flat, prefix="test", short_window=8, long_window=32, sigma_window=96
    )
    assert cp["test_cp_z_8_32_96"].isna().all().all()
    assert cp["test_cp_logstd_8_32"].isna().all().all()
    # The absolute mean ratio is defined for a flat non-zero level and should
    # remain its genuine value rather than be treated as a scale feature.
    assert cp["test_cp_absratio_8_32"].iloc[96:].eq(1.0).all().all()
