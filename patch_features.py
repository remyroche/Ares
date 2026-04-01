import sys
import re

def patch_file():
    filepath = 'extreme_price_movements/features.py'
    with open(filepath, 'r') as f:
        content = f.read()

    new_imports = """        bars_since_flip_nb,
        binary_entropy_nb,
        binary_entropy_nb_parallel,
        consecutive_bars_nb,
        consecutive_bars_nb_parallel,
        up_down_semivol_ratio_nb,
        up_down_semivol_ratio_nb_parallel,
        up_down_return_mass_ratio_nb,
        up_down_return_mass_ratio_nb_parallel,"""

    content = content.replace("""        bars_since_flip_nb,
        binary_entropy_nb,
        binary_entropy_nb_parallel,""", new_imports)

    new_features = """        if _needs_feature("bars_since_trend_flip"):
            trend_slope = ff.apply_to_frame(c_log, ff.slope_nb, 6)
            trend_sign = (trend_slope > 0).astype(np.float32)
            feats["bars_since_trend_flip"] = ff.apply_to_frame(
                trend_sign, bars_since_flip_nb
            ).astype(np.float32)

        if _needs_feature("bars_since_ema20_ema50_cross_log_norm"):
            ema_20 = ff.apply_to_frame(c_log, ff.ema_nb, 20)
            ema_50 = ff.apply_to_frame(c_log, ff.ema_nb, 50)
            ema_diff_sign = ((ema_20 - ema_50) > 0).astype(np.float32)
            raw = ff.apply_to_frame(ema_diff_sign, bars_since_flip_nb)
            feats["bars_since_ema20_ema50_cross_log_norm"] = (np.log1p(np.minimum(raw, 100)) / np.log1p(100)).astype(np.float32)

        if _needs_feature("bars_in_high_vol_state_log_norm"):
            high_vol_state = (feats["atr_pct_rank"] >= 0.8).astype(np.float32)
            raw = ff.apply_to_frame(high_vol_state, consecutive_bars_nb)
            feats["bars_in_high_vol_state_log_norm"] = (np.log1p(np.minimum(raw, 50)) / np.log1p(50)).astype(np.float32)

        if _needs_feature("bars_outside_ema20_atr_band_log_norm"):
            ema_20 = ff.apply_to_frame(c_log, ff.ema_nb, 20)
            dist = np.abs(c_raw - np.exp(ema_20)) / np.maximum(atr, 1e-8)
            outside_state = (dist >= 1.0).astype(np.float32)
            raw = ff.apply_to_frame(outside_state, consecutive_bars_nb)
            feats["bars_outside_ema20_atr_band_log_norm"] = (np.log1p(np.minimum(raw, 50)) / np.log1p(50)).astype(np.float32)

        if _needs_feature("up_down_semivol_ratio_tanh"):
            feats["up_down_semivol_ratio_tanh"] = ff.apply_to_frame(ret_1, up_down_semivol_ratio_nb, 20).astype(np.float32)

        if _needs_feature("up_down_return_mass_ratio_tanh"):
            feats["up_down_return_mass_ratio_tanh"] = ff.apply_to_frame(ret_1, up_down_return_mass_ratio_nb, 20).astype(np.float32)

        if _needs_feature("tail_asymmetry_q90_q10_atr_norm"):
            q90 = ff.numba_rolling_quantile(ret_1, 50, 0.90)
            q10 = np.abs(ff.numba_rolling_quantile(ret_1, 50, 0.10))
            raw = np.log((q90 + 1e-8) / (q10 + 1e-8))
            feats["tail_asymmetry_q90_q10_atr_norm"] = np.tanh(raw).astype(np.float32)"""

    content = content.replace("""        if _needs_feature("bars_since_trend_flip"):
            trend_slope = ff.apply_to_frame(c_log, ff.slope_nb, 6)
            trend_sign = (trend_slope > 0).astype(np.float32)
            feats["bars_since_trend_flip"] = ff.apply_to_frame(
                trend_sign, bars_since_flip_nb
            ).astype(np.float32)""", new_features)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    patch_file()
