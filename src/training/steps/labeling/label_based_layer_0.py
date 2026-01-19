from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from src.training.steps.labeling.multi_label_voting_utils import (
    compute_kalman_smoothed_price_and_volatility,
    compute_volume_weighted_kalman_smoothed_price_and_volatility,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig
try:
    from src.training.steps.labeling.optimized_wavelet_decomposition import OptimizedWaveletDecomposition
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
from src.utils.tprint import tprint_info, tprint_success, tprint_warning


DEFAULT_RANDOM_SEED = 42


def get_reproducible_random_state(base_seed: int = DEFAULT_RANDOM_SEED, offset: int = 0) -> int:
    try:
        base_seed_i = int(base_seed)
    except Exception:
        base_seed_i = DEFAULT_RANDOM_SEED
    try:
        offset_i = int(offset)
    except Exception:
        offset_i = 0
    return int((base_seed_i + offset_i) % (2**31 - 1))


def _rolling_sum_prefix(values: np.ndarray, window: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = int(v.shape[0])
    w = int(max(1, min(int(window), n)))
    c = np.cumsum(np.where(np.isfinite(v), v, 0.0), dtype=float)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def compute_filter_diagnostics(
    raw: np.ndarray,
    filtered: np.ndarray,
    filter_name: str,
    sampling_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Compute comprehensive noise reduction diagnostics for a filter.
    
    Args:
        raw: Original raw signal
        filtered: Filtered signal
        filter_name: Name of the filter for reporting
        sampling_rate: Sampling rate (Hz) for frequency analysis
    
    Returns:
        Dictionary of diagnostic metrics (all with valid float values, no NaN)
    """
    # Initialize ALL expected metrics with safe defaults
    diagnostics = {
        f"{filter_name}_snr_improvement": 0.0,
        f"{filter_name}_noise_reduction": 0.0,
        f"{filter_name}_smoothness_ratio": 1.0,  # 1.0 = no change
        f"{filter_name}_tracking_rmse": 0.0,
        f"{filter_name}_high_freq_reduction": 0.0,
        f"{filter_name}_low_freq_preservation": 1.0,  # 1.0 = perfect preservation
    }
    
    try:
        # Ensure arrays are valid
        raw = np.asarray(raw, dtype=float)
        filtered = np.asarray(filtered, dtype=float)
        
        # Check for valid data
        valid_mask = np.isfinite(raw) & np.isfinite(filtered)
        if valid_mask.sum() < 10:
            return diagnostics
        
        raw = raw[valid_mask]
        filtered = filtered[valid_mask]
        
        # Basic signal quality metrics
        raw_var = np.var(raw)
        filtered_var = np.var(filtered)
        noise_var = np.var(raw - filtered)
        
        # SNR improvement (signal variance / noise variance)
        if noise_var > 1e-12:
            snr_improvement = filtered_var / noise_var
            diagnostics[f"{filter_name}_snr_improvement"] = float(np.clip(snr_improvement, 0, 1e6))
        
        # Noise reduction percentage
        if raw_var > 1e-12:
            noise_reduction = 1.0 - (noise_var / raw_var)
            diagnostics[f"{filter_name}_noise_reduction"] = float(np.clip(noise_reduction, -1, 1))
        
        # Smoothness metric (lower is smoother)
        if len(raw) > 2:
            raw_roughness = np.mean(np.diff(raw, n=2) ** 2)
            filtered_roughness = np.mean(np.diff(filtered, n=2) ** 2)
            if raw_roughness > 1e-12:
                smoothness_ratio = filtered_roughness / raw_roughness
                diagnostics[f"{filter_name}_smoothness_ratio"] = float(np.clip(smoothness_ratio, 0, 100))
        
        # Tracking error (RMSE)
        tracking_error = np.sqrt(np.mean((raw - filtered) ** 2))
        diagnostics[f"{filter_name}_tracking_rmse"] = float(np.clip(tracking_error, 0, 1e6))
        
        # Frequency domain analysis (if scipy available)
        try:
            from scipy import signal as scipy_signal
            
            # Power spectral density analysis
            freqs_raw, psd_raw = scipy_signal.periodogram(raw, fs=sampling_rate)
            freqs_filt, psd_filt = scipy_signal.periodogram(filtered, fs=sampling_rate)
            
            # Compute noise reduction in different frequency bands
            if len(freqs_raw) > 4 and len(freqs_filt) > 4:
                # High frequency noise (upper 25% of spectrum)
                high_freq_idx = max(1, int(0.75 * len(freqs_raw)))
                high_freq_power_raw = np.mean(psd_raw[high_freq_idx:])
                high_freq_power_filt = np.mean(psd_filt[high_freq_idx:])
                
                if high_freq_power_raw > 1e-12 and np.isfinite(high_freq_power_filt):
                    high_freq_reduction = 1.0 - (high_freq_power_filt / high_freq_power_raw)
                    diagnostics[f"{filter_name}_high_freq_reduction"] = float(np.clip(high_freq_reduction, -1, 1))
                
                # Low frequency preservation (lower 25% of spectrum)
                low_freq_idx = max(1, int(0.25 * len(freqs_raw)))
                low_freq_power_raw = np.mean(psd_raw[:low_freq_idx])
                low_freq_power_filt = np.mean(psd_filt[:low_freq_idx])
                
                if low_freq_power_raw > 1e-12 and np.isfinite(low_freq_power_filt):
                    low_freq_preservation = low_freq_power_filt / low_freq_power_raw
                    diagnostics[f"{filter_name}_low_freq_preservation"] = float(np.clip(low_freq_preservation, 0, 10))
                    
        except ImportError:
            # scipy not available, skip frequency analysis (defaults already set)
            pass
            
    except Exception:
        # Return defaults if any computation fails
        pass
    
    return diagnostics


def compute_fisher_transform(
    prices: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Apply Fisher transform to price changes for normalization.
    
    Args:
        prices: Price series
        window: Lookback window for price change calculation
    
    Returns:
        Fisher-transformed series
    """
    try:
        # Calculate price changes
        price_changes = prices.pct_change(window)
        
        # Fisher transform: atanh(2 * normalized_change - 1)
        # Normalize changes to [0, 1] range first
        min_change = price_changes.rolling(window * 2, min_periods=window).min()
        max_change = price_changes.rolling(window * 2, min_periods=window).max()
        
        # Avoid division by zero
        range_change = max_change - min_change
        range_change = range_change.replace(0.0, 1e-12)
        
        normalized = (price_changes - min_change) / range_change
        normalized = normalized.clip(0.01, 0.99)  # Avoid atanh boundaries
        
        # Apply Fisher transform
        fisher = np.arctanh(2 * normalized - 1)
        
        return fisher
    except Exception:
        return pd.Series(0.0, index=prices.index)


def _ffill_nan(arr: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    n = int(x.shape[0])
    if n == 0:
        return x
    mask = np.isfinite(x)
    if not bool(mask.any()):
        if fallback is None:
            return x
        return np.asarray(fallback, dtype=float)
    idx = np.where(mask, np.arange(n, dtype=int), 0)
    idx = np.maximum.accumulate(idx)
    out = x[idx]
    if fallback is not None:
        fb = np.asarray(fallback, dtype=float)
        out = np.where(np.isfinite(out), out, fb)
    return out


def compute_rolling_vwap(
    close: pd.Series,
    volume: Optional[pd.Series],
    lookback: int,
) -> pd.Series:
    close_s = pd.to_numeric(close, errors="coerce")
    close_vals = close_s.to_numpy(dtype=float)
    n = int(close_vals.shape[0])
    lb = int(max(2, min(int(lookback), max(2, n))))

    if volume is None:
        sum_close = _rolling_sum_prefix(close_vals, lb)
        denom = np.minimum(np.arange(1, n + 1, dtype=float), float(lb))
        out = sum_close / (denom + 1e-12)
        out = _ffill_nan(out, fallback=close_vals)
        return pd.Series(out, index=close_s.index)

    vol_s = pd.to_numeric(volume, errors="coerce")
    vol_vals = vol_s.to_numpy(dtype=float)
    if not bool(np.isfinite(vol_vals).any()):
        sum_close = _rolling_sum_prefix(close_vals, lb)
        denom = np.minimum(np.arange(1, n + 1, dtype=float), float(lb))
        out = sum_close / (denom + 1e-12)
        out = _ffill_nan(out, fallback=close_vals)
        return pd.Series(out, index=close_s.index)

    pv_vals = close_vals * np.where(np.isfinite(vol_vals), vol_vals, 0.0)
    v_safe = np.where(np.isfinite(vol_vals) & (vol_vals > 0.0), vol_vals, 0.0)
    sum_pv = _rolling_sum_prefix(pv_vals, lb)
    sum_v = _rolling_sum_prefix(v_safe, lb)
    out = sum_pv / (sum_v + 1e-12)
    out = np.where(sum_v > 0.0, out, np.nan)
    out = _ffill_nan(out, fallback=close_vals)
    return pd.Series(out, index=close_s.index)


def run_layer0_kalman_vwap(
    symbol: str,
    timeframe: str,
    market_data: pd.DataFrame,
    config: Dict[str, Any],
    outcomes_dir: Path,
    bundle_path: Optional[Path] = None,
    run_optimization: bool = True,
    train_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Use training data for optimization if provided
    opt_data = train_data if train_data is not None else market_data
    close_series = pd.to_numeric(opt_data.get("close"), errors="coerce")
    volume_series = opt_data.get("volume", None)
    if isinstance(volume_series, pd.Series):
        volume_series = pd.to_numeric(volume_series, errors="coerce")
        
    # --- CAUSAL DENOISING INTEGRATION (REPLACING WAVELET) ---
    if config.get("use_wavelets", True) and WAVELET_AVAILABLE:
        try:
            tprint_info("🌊 Running Causal Denoising (Strictly Causal EWMA)...")
            # Force causal mode to prevent lookahead bias
            decomposer = OptimizedWaveletDecomposition(verbose=False, causal=True)
            close_vals = close_series.to_numpy(dtype=float)
            
            # 1. Causal Denoising (EWMA)
            # denoise_signal_vectorized handles causal logic when initialized with causal=True
            denoised_wavelet = decomposer.denoise_signal_vectorized(
                close_vals, 
                threshold_method='visushrink', 
                threshold_mode='soft'
            )
            
            # 2. Median Filtering
            try:
                from scipy.signal import medfilt
                denoised_median = medfilt(denoised_wavelet, kernel_size=3)
            except ImportError:
                denoised_median = denoised_wavelet # Fallback
            
            # 3. Outlier Clipping (Robust z-score via MAD)
            # Use Pandas rolling for efficiency
            s_denoised = pd.Series(denoised_median)
            # Rolling MAD (approximated as 0.6745 * abs(x - median))
            roll_med = s_denoised.rolling(window=5, min_periods=1, center=True).median()
            roll_abs_dev = (s_denoised - roll_med).abs()
            roll_mad = roll_abs_dev.rolling(window=5, min_periods=1, center=True).median()
            # Sigma ~ 1.4826 * MAD
            roll_sigma = 1.4826 * roll_mad
            
            # Robust 5-sigma clip
            limit_upper = roll_med + 5 * roll_sigma
            limit_lower = roll_med - 5 * roll_sigma
            
            denoised_final = s_denoised.clip(lower=limit_lower, upper=limit_upper).to_numpy()
            
            # Store in market_data
            if len(denoised_final) == len(market_data):
                market_data['wavelet_close'] = denoised_final
                market_data['wavelet_noise'] = close_vals - denoised_final
                tprint_success("✅ Advanced Denoising complete (Wavelet+Median+Clip).")
            else:
                tprint_warning("⚠️ Wavelet output length mismatch, skipping assignment.")
                
        except Exception as e:
            tprint_warning(f"⚠️ Wavelet Denoising failed: {e}")
    elif config.get("use_wavelets", False) and not WAVELET_AVAILABLE:
        tprint_warning("⚠️ Wavelet requested but module not available.")


    if bundle_path is None:
        bundle_path = outcomes_dir / "layer0_kalman_bundle.joblib"

    best_params: Dict[str, Any] = {}
    loaded_from: Optional[str] = None
    if (not run_optimization) and bundle_path.exists():
        try:
            payload = joblib.load(bundle_path)
            best_params = dict(payload.get("best_params", {}) or {})
            loaded_from = str(bundle_path)
        except Exception:
            best_params = {}

    if run_optimization or not best_params:
        def _objective(params: Dict[str, Any]) -> float:
            Q = float(params.get("kalman_Q", 1e-4))
            R = float(params.get("kalman_R", 0.01))
            volume_weight = float(params.get("volume_weight", 1.0))
            volume_adaptive = bool(params.get("volume_adaptive", True))

            # Optimization Subsampling: Use last 10,000 bars
            n_eval = 10000
            eval_close = close_series.iloc[-n_eval:] if len(close_series) > n_eval else close_series
            eval_vol = volume_series.iloc[-n_eval:] if (volume_series is not None and len(volume_series) > n_eval) else volume_series
            
            try:
                # Use volume-weighted Kalman filter
                smoothed_close, _smoothed_vol = compute_volume_weighted_kalman_smoothed_price_and_volatility(
                    prices=eval_close,
                    volume=eval_vol,
                    process_noise=Q,
                    measurement_noise=R,
                    vol_window=20,
                    volume_weight=volume_weight,
                    volume_adaptive=volume_adaptive,
                )

                raw = eval_close.to_numpy(dtype=float)
                smooth = pd.to_numeric(smoothed_close, errors="coerce").to_numpy(dtype=float)

                mask = np.isfinite(raw) & np.isfinite(smooth)
                if int(mask.sum()) < 100:
                    return 10.0

                raw_m = raw[mask]
                smooth_m = smooth[mask]
                denom = float(np.nanstd(np.diff(raw_m))) + 1e-9

                # Compute diagnostics for volume-weighted Kalman filter
                all_diagnostics = {}
                
                # 1. Volume-weighted Kalman filter diagnostics
                kalman_diags = compute_filter_diagnostics(raw_m, smooth_m, "volume_kalman", sampling_rate=4.0)
                all_diagnostics.update(kalman_diags)
                
                # 2. Simple moving average for comparison
                ma_window = 20
                ma_series = eval_close.rolling(ma_window).mean()
                ma_vals = pd.to_numeric(ma_series, errors="coerce").to_numpy(dtype=float)
                ma_m = ma_vals[mask] if len(ma_vals) == len(raw) else np.zeros_like(raw_m)
                ma_diags = compute_filter_diagnostics(raw_m, ma_m, "moving_average", sampling_rate=4.0)
                all_diagnostics.update(ma_diags)
                
                # 3. Fisher transform for comparison
                fisher_series = compute_fisher_transform(eval_close, window=14)
                fisher_vals = pd.to_numeric(fisher_series, errors="coerce").to_numpy(dtype=float)
                fisher_m = fisher_vals[mask] if len(fisher_vals) == len(raw) else np.zeros_like(raw_m)
                fisher_diags = compute_filter_diagnostics(raw_m, fisher_m, "fisher", sampling_rate=4.0)
                all_diagnostics.update(fisher_diags)

                smooth_pen = float(np.mean(np.diff(smooth_m, n=2) ** 2) / (denom**2))
                track_pen = float(np.mean((smooth_m - raw_m) ** 2) / (denom**2))
                
                # Add regularization for parameter stability
                param_reg = 0.01 * (abs(np.log10(Q + 1e-12)) + abs(np.log10(R + 1e-12)))
                
                loss = smooth_pen + track_pen + param_reg
                
                # Store diagnostics for later analysis
                _objective.last_diagnostics = all_diagnostics
                _objective.last_loss_components = {
                    "smoothness_penalty": smooth_pen,
                    "tracking_penalty": track_pen,
                    "volume_weight": volume_weight,
                    "volume_adaptive": volume_adaptive,
                    "parameter_regularization": param_reg,
                    "total_loss": loss
                }
                
                return float(loss) if np.isfinite(loss) else 10.0
            except Exception:
                return 10.0

        exec_mode = str(config.get("execution_mode", "light")).lower()
        # Reduce grid density from 5 to 3 for all modes to satisfy user request for speed
        # (3^4 = 81 trials, vs 5^4 = 625)
        # We process 'full' mode with reduced grid points to keep data intact but speed up HPO
        grid_points = 3
        
        optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(
                n_trials=int(config.get("layer0_n_trials", config.get("stage0_n_trials", 50))),
                execution_mode=exec_mode,
                direction="minimize",  # Minimize loss, not maximize
                seed=int(config.get("random_state", 42)),
                coarse_grid_points=grid_points,
                fine_grid_points=grid_points,
            )
        )
        search_space = {
            "kalman_Q": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},      # Expanded range
            "kalman_R": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},      # Expanded range
            "volume_weight": {"type": "float", "low": 0.0, "high": 3.0, "log": False}, # Volume-based weighting
            "volume_adaptive": {"type": "categorical", "choices": [True, False]},       # Adaptive vs simple
        }
        opt_res = optimizer.optimize(objective=_objective, search_space=search_space)
        best_params = dict(opt_res.get("best_params", {}) or {})
        loaded_from = None

    try:
        Q_best = float(best_params.get("kalman_Q", 1e-4))
        R_best = float(best_params.get("kalman_R", 0.01))
    except Exception:
        Q_best, R_best = 1e-4, 0.01

    try:
        volume_weight = float(best_params.get("volume_weight", 1.0))
    except Exception:
        volume_weight = 1.0
    
    try:
        volume_adaptive = bool(best_params.get("volume_adaptive", True))
    except Exception:
        volume_adaptive = True

    # Apply volume-weighted Kalman filtering to full market_data
    try:
        kalman_price, kalman_vol = compute_volume_weighted_kalman_smoothed_price_and_volatility(
            prices=market_data["close"],
            volume=market_data.get("volume", None),
            process_noise=float(Q_best),
            measurement_noise=float(R_best),
            vol_window=20,
            volume_weight=volume_weight,
            volume_adaptive=volume_adaptive,
        )
        market_data["kalman_price"] = kalman_price
        market_data["kalman_volatility"] = kalman_vol
    except Exception:
        pass

    payload = {
        "best_params": {
            "kalman_Q": float(Q_best),
            "kalman_R": float(R_best),
            "volume_weight": float(volume_weight),
            "volume_adaptive": bool(volume_adaptive),
        },
        "loaded_from": loaded_from,
    }
    try:
        joblib.dump(payload, bundle_path)
    except Exception:
        pass

    try:
        outcomes_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        ts = str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    except Exception:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    try:
        symbol = str(config.get("symbol", ""))
    except Exception:
        symbol = ""
    try:
        timeframe = str(config.get("timeframe", ""))
    except Exception:
        timeframe = ""

    try:
        idx = market_data.index
        start_ts = str(idx.min()) if len(idx) else ""
        end_ts = str(idx.max()) if len(idx) else ""
    except Exception:
        start_ts, end_ts = "", ""

    # Capture best diagnostics from optimization (only if optimization was run)
    try:
        best_diagnostics = getattr(_objective, 'last_diagnostics', {})
        best_loss_components = getattr(_objective, 'last_loss_components', {})
    except NameError:
        # _objective not defined because optimization was skipped (loaded from bundle)
        best_diagnostics = {}
        best_loss_components = {}

    try:
        nan_metrics = [
            key for key, value in best_diagnostics.items()
            if isinstance(value, (int, float)) and np.isnan(value)
        ]
        if nan_metrics:
            tprint_warning(f"⚠️ Layer0 diagnostics contain NaNs: {nan_metrics}")
            if bool(config.get("fail_on_layer0_nan_diagnostics", False)):
                raise RuntimeError("Layer0 diagnostics contain NaNs")
    except Exception:
        pass
    
    try:
        md_path = outcomes_dir / f"layer0_report_{symbol}_{timeframe}_{ts}.md"
        lines = [
            "# Layer0 Report\n",
            f"- timestamp: {ts}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- run_optimization: {bool(run_optimization)}\n",
            f"- bundle_path: {str(bundle_path)}\n",
            f"- loaded_from: {str(loaded_from) if loaded_from else ''}\n",
            f"- n_bars: {int(len(market_data))}\n",
            f"- date_range: {start_ts} -> {end_ts}\n",
            "\n## Best Params\n",
            f"- kalman_Q: {float(Q_best)}\n",
            f"- kalman_R: {float(R_best)}\n",
            f"- volume_weight: {float(volume_weight)}\n",
            f"- volume_adaptive: {bool(volume_adaptive)}\n",
            "\n## Loss Components\n",
            f"- smoothness_penalty: {best_loss_components.get('smoothness_penalty', 'N/A')}\n",
            f"- tracking_penalty: {best_loss_components.get('tracking_penalty', 'N/A')}\n",
            f"- volume_weight: {best_loss_components.get('volume_weight', 'N/A')}\n",
            f"- volume_adaptive: {best_loss_components.get('volume_adaptive', 'N/A')}\n",
            f"- parameter_regularization: {best_loss_components.get('parameter_regularization', 'N/A')}\n",
            f"- total_loss: {best_loss_components.get('total_loss', 'N/A')}\n",
            "\n## Filter Diagnostics\n",
        ]
        
        # Add filter diagnostics in organized sections
        filter_sections = {
            'volume_kalman': ['volume_kalman_snr_improvement', 'volume_kalman_noise_reduction', 'volume_kalman_smoothness_ratio', 'volume_kalman_tracking_rmse'],
            'moving_average': ['moving_average_snr_improvement', 'moving_average_noise_reduction', 'moving_average_smoothness_ratio', 'moving_average_tracking_rmse'],
            'fisher': ['fisher_snr_improvement', 'fisher_noise_reduction', 'fisher_smoothness_ratio', 'fisher_tracking_rmse']
        }
        
        for filter_name, metrics in filter_sections.items():
            lines.append(f"\n### {filter_name.title()} Filter\n")
            for metric in metrics:
                value = best_diagnostics.get(metric, 'N/A')
                if isinstance(value, (int, float)):
                    lines.append(f"- {metric}: {value:.6f}\n")
                else:
                    lines.append(f"- {metric}: {value}\n")
        
        # Add frequency domain diagnostics if available
        freq_metrics = [k for k in best_diagnostics.keys() if 'high_freq' in k or 'low_freq' in k]
        if freq_metrics:
            lines.append("\n### Frequency Domain Analysis\n")
            for metric in sorted(freq_metrics):
                value = best_diagnostics.get(metric, 'N/A')
                if isinstance(value, (int, float)):
                    lines.append(f"- {metric}: {value:.6f}\n")
                else:
                    lines.append(f"- {metric}: {value}\n")
        
        md_path.write_text("".join(lines))
    except Exception:
        pass

    try:
        # Create comprehensive summary with diagnostics
        summary_row = {
            "timestamp": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "run_optimization": bool(run_optimization),
            "loaded_from": str(loaded_from) if loaded_from else "",
            "bundle_path": str(bundle_path),
            "n_bars": int(len(market_data)),
            "start": start_ts,
            "end": end_ts,
            "kalman_Q": float(Q_best),
            "kalman_R": float(R_best),
            "volume_weight": float(volume_weight),
            "volume_adaptive": bool(volume_adaptive),
            "total_loss": best_loss_components.get('total_loss', None),
            "smoothness_penalty": best_loss_components.get('smoothness_penalty', None),
            "tracking_penalty": best_loss_components.get('tracking_penalty', None),
            "volume_weight_used": best_loss_components.get('volume_weight', None),
            "volume_adaptive_used": best_loss_components.get('volume_adaptive', None),
            "parameter_regularization": best_loss_components.get('parameter_regularization', None),
        }
        
        # Add key diagnostic metrics to summary
        key_metrics = [
            'volume_kalman_snr_improvement', 'volume_kalman_noise_reduction', 'volume_kalman_tracking_rmse',
            'moving_average_snr_improvement', 'moving_average_noise_reduction', 'moving_average_tracking_rmse',
            'fisher_snr_improvement', 'fisher_noise_reduction', 'fisher_tracking_rmse'
        ]
        
        for metric in key_metrics:
            summary_row[metric] = best_diagnostics.get(metric, None)
        
        csv_path = outcomes_dir / f"layer0_summary_{symbol}_{timeframe}_{ts}.csv"
        pd.DataFrame([summary_row]).to_csv(csv_path, index=False)
        
        # Also save detailed diagnostics as separate CSV
        if best_diagnostics:
            diag_path = outcomes_dir / f"layer0_diagnostics_{symbol}_{timeframe}_{ts}.csv"
            # Convert diagnostics to flat format
            diag_data = []
            for metric, value in best_diagnostics.items():
                diag_data.append({
                    'metric': metric,
                    'value': value,
                    'filter_type': metric.split('_')[0] if '_' in metric else 'unknown'
                })
            pd.DataFrame(diag_data).to_csv(diag_path, index=False)
            
    except Exception:
        pass

    # Apply all filters to full dataset and save for visualization
    try:
        filter_outputs = {}
        
        # 1. Volume-weighted Kalman filter (already computed)
        if 'kalman_price' in market_data.columns:
            filter_outputs['volume_kalman'] = market_data['kalman_price']
        
        # 2. Simple moving average for comparison
        filter_outputs['moving_average'] = market_data['close'].rolling(20).mean()
        
        # 3. Fisher transform
        fisher_series = compute_fisher_transform(market_data['close'], window=14)
        filter_outputs['fisher'] = fisher_series
        
        # 4. Raw price for reference
        filter_outputs['raw_price'] = market_data['close']
        
        # Save filter outputs for analysis
        if filter_outputs:
            filter_df = pd.DataFrame(filter_outputs, index=market_data.index)
            filter_path = outcomes_dir / f"layer0_filter_outputs_{symbol}_{timeframe}_{ts}.csv"
            filter_df.to_csv(filter_path)
            
    except Exception:
        pass

    return market_data, payload
