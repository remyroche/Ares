"""Meta-Labeling HPO Layer 2: Trading Parameters Optimization.

This module handles Layer 2 of the hierarchical HPO process:
- Optimizes trading parameters (TPSL, horizon, spacing)
- Optimizes probability thresholds and risk/reward ratios
- Supports regime-conditional barrier geometry
- Supports Mixture of Experts (MoE) for ensemble trading

Layer 2 focuses on:
1. Stop-loss and take-profit multipliers
2. Risk/reward ratio optimization
3. Horizon and event spacing tuning
4. Probability threshold calibration
5. Trailing stop configuration
6. Regime-adaptive barrier geometry
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
    compute_uniqueness,
)

# Import shared utilities
try:
    from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
        _write_hpo_stage_report,
        _find_latest_path,
        _sanitize_json_value,
        tprint_info,
        tprint_success,
        tprint_warning,
        DEFAULT_TRANSACTION_COST,
        calculate_hpo_utility,
    )
except ImportError:
    # Fallback for standalone testing
    DEFAULT_TRANSACTION_COST = 0.0003
    def tprint_info(msg: str) -> None:
        print(f"[INFO] {msg}")
    def tprint_success(msg: str) -> None:
        print(f"[SUCCESS] {msg}")
    def tprint_warning(msg: str) -> None:
        print(f"[WARNING] {msg}")
    def _find_latest_path(outcomes_dir: Path, pattern: str) -> Optional[Path]:
        try:
            candidates = list(outcomes_dir.glob(pattern))
            if not candidates:
                return None
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            return None
    def _write_hpo_stage_report(**kwargs) -> Dict[str, Any]:
        return {}
    def _sanitize_json_value(obj: Any) -> Any:
        return obj
    def calculate_hpo_utility(**kwargs) -> Dict[str, Any]:
        return {"utility": 0.0}


def get_layer2_search_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get the Layer 2 search space for trading parameters.
    
    Args:
        config: HPO configuration
        
    Returns:
        Dictionary defining the search space for Layer 2
    """
    try:
        l2_prob_thr_high = float(config.get("layer2_prob_threshold_high", 0.70))
    except Exception:
        l2_prob_thr_high = 0.70
    l2_prob_thr_high = float(np.clip(l2_prob_thr_high, 0.55, 0.85))
    
    try:
        l2_vol_penalty_high = float(config.get("layer2_volatility_penalty_lambda_high", 0.25))
    except Exception:
        l2_vol_penalty_high = 0.25
    if not np.isfinite(l2_vol_penalty_high):
        l2_vol_penalty_high = 0.25
    l2_vol_penalty_high = float(np.clip(l2_vol_penalty_high, 0.0, 1.0))
    
    return {
        # Core barrier parameters
        "profit_floor_tx_mult": {"type": "float", "low": 1.0, "high": 4.0},
        "sl_atr_mult": {"type": "float", "low": 0.5, "high": 3.0},
        "risk_reward_ratio": {"type": "float", "low": 1.0, "high": 5.0},
        "horizon_bars": {"type": "int", "low": 6, "high": 48},
        "min_event_spacing": {"type": "int", "low": 0, "high": 6},
        "trail_distance_atr_mult": {"type": "float", "low": 0.5, "high": 3.0},
        
        # Probability and volatility parameters
        "prob_threshold": {"type": "float", "low": 0.50, "high": float(l2_prob_thr_high)},
        "ev_margin": {"type": "float", "low": 0.0, "high": 0.25},
        "volatility_penalty_lambda": {"type": "float", "low": 0.0, "high": float(l2_vol_penalty_high)},
        
        # Regime-conditional barrier geometry
        "barrier_regime_strength": {"type": "float", "low": 0.0, "high": 1.0},
        "barrier_regime_power": {"type": "float", "low": 0.5, "high": 2.0},
        
        # Strength-Adaptive Threshold
        "sig_strength_sensitivity": {"type": "float", "low": 0.0, "high": 0.3},
        
        # Trailing Stop Trend Modulation
        "trail_trend_modulation": {"type": "float", "low": 0.0, "high": 2.0},
        
        # Barrier Asymmetry Regime Modulation
        "barrier_trend_asymmetry": {"type": "float", "low": 0.0, "high": 1.5},
        
        # Volume-Weighted Time
        "horizon_volume_modulation": {"type": "float", "low": 0.0, "high": 2.0},
        
        # Volatility-of-Volatility Adjustment
        "barrier_vol_vol_exp": {"type": "float", "low": 0.0, "high": 1.5},
        
        # Mixture of Experts (MoE) Parameters
        "moe_trend_dominance": {"type": "float", "low": 0.0, "high": 1.0},
        "moe_scalp_dominance": {"type": "float", "low": 0.0, "high": 1.0},
        "moe_vol_sensitivity": {"type": "float", "low": 0.0, "high": 1.0},
        "moe_adx_trend_q": {"type": "float", "low": 0.55, "high": 0.95},
        "moe_adx_chop_q": {"type": "float", "low": 0.05, "high": 0.45},
        "moe_vol_spike_q": {"type": "float", "low": 0.70, "high": 0.99},
        
        # Probabilistic Stops (First Passage Time Veto)
        "prob_stop_enable": {"type": "int", "low": 0, "high": 1},
        "prob_stop_threshold": {"type": "float", "low": 0.55, "high": 0.95},
        "prob_stop_drift_window": {"type": "int", "low": 12, "high": 96},
        
        # Regime-Adaptive Probability Thresholds
        "prob_threshold_adj_vol_low": {"type": "float", "low": -0.15, "high": 0.05},
        "prob_threshold_adj_vol_high": {"type": "float", "low": -0.05, "high": 0.10},
        "prob_threshold_adj_trend_high": {"type": "float", "low": -0.10, "high": 0.05},
        "prob_threshold_adj_trend_low": {"type": "float", "low": -0.05, "high": 0.10},
    }


def compute_regime_conditional_barrier_geometry(
    *,
    params: Dict[str, Any],
    market_index: pd.Index,
    default_horizon: int,
    atr_frac_series: pd.Series,
    market_data: pd.DataFrame,
    config: Dict[str, Any],
    enable_regime_conditional: bool = True,
    barrier_geometry_regime_col: str = "hmm_regime_label_1h",
    barrier_geometry_by_regime: Optional[Dict[str, Any]] = None,
    regime_scalar_for_barriers: Optional[pd.Series] = None,
    layer2_profit_floor_tx_mult: float = 1.05,
) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[pd.Series]]:
    """Compute regime-conditional barrier geometry for Layer 2.
    
    Args:
        params: Trial parameters
        market_index: Index for output series
        default_horizon: Default horizon in bars
        atr_frac_series: ATR fraction series
        market_data: Market OHLCV data
        config: HPO configuration
        enable_regime_conditional: Whether to enable regime conditioning
        barrier_geometry_regime_col: Column name for regime labels
        barrier_geometry_by_regime: Optional regime-specific geometry settings
        regime_scalar_for_barriers: Optional pre-computed regime scalars
        layer2_profit_floor_tx_mult: Profit floor multiplier
        
    Returns:
        Tuple of (profit_threshold, stop_threshold, horizon_series, trail_series)
    """
    try:
        profit_floor_tx_mult = float(params.get("profit_floor_tx_mult", layer2_profit_floor_tx_mult))
    except Exception:
        profit_floor_tx_mult = float(layer2_profit_floor_tx_mult)
    if (not np.isfinite(profit_floor_tx_mult)) or profit_floor_tx_mult <= 0.0:
        profit_floor_tx_mult = float(layer2_profit_floor_tx_mult)
    profit_floor_tx_mult = float(np.clip(profit_floor_tx_mult, 1.0, 10.0))
    min_profit_floor_local = float(DEFAULT_TRANSACTION_COST) * float(profit_floor_tx_mult)
    
    base_h = int(params.get("horizon_bars", default_horizon))
    base_sl = float(params.get("sl_atr_mult", 1.0))
    base_rr = float(params.get("risk_reward_ratio", 2.0))
    base_trail = float(params.get("trail_distance_atr_mult", 0.0))
    
    if (not np.isfinite(base_sl)) or base_sl <= 0.0:
        base_sl = 1.0
    if (not np.isfinite(base_rr)) or base_rr <= 0.0:
        base_rr = 2.0
    if (not np.isfinite(base_trail)) or base_trail < 0.0:
        base_trail = 0.0
    if base_h <= 0:
        base_h = int(default_horizon)
    
    stop_mult = pd.Series(float(base_sl), index=market_index, dtype=float)
    rr_series = pd.Series(float(base_rr), index=market_index, dtype=float)
    horizon_series = pd.Series(float(base_h), index=market_index, dtype=float)
    trail_mult_series: Optional[pd.Series] = None
    try:
        trail_mult_series = pd.Series(float(base_trail), index=market_index, dtype=float)
    except Exception:
        trail_mult_series = None
    
    # Apply regime-conditional geometry if enabled
    if bool(enable_regime_conditional) and barrier_geometry_by_regime is not None:
        try:
            if barrier_geometry_regime_col in market_data.columns:
                regimes = market_data[barrier_geometry_regime_col].reindex(market_index)
                reg_keys = regimes.astype(object).astype(str)
                
                def _map_param(key: str, default_v: float) -> pd.Series:
                    out = pd.Series(float(default_v), index=market_index, dtype=float)
                    for rk in pd.unique(reg_keys.dropna()):
                        spec = barrier_geometry_by_regime.get(str(rk))
                        if not isinstance(spec, dict):
                            continue
                        v = spec.get(key)
                        vm = spec.get(f"{key}_mult")
                        if v is None and vm is None:
                            continue
                        try:
                            if v is not None:
                                vv = float(v)
                            else:
                                vv = float(default_v) * float(vm)
                            if not np.isfinite(vv):
                                continue
                        except Exception:
                            continue
                        out.loc[reg_keys == str(rk)] = float(vv)
                    return out
                
                stop_mult = _map_param("sl_atr_mult", float(base_sl))
                rr_series = _map_param("risk_reward_ratio", float(base_rr))
                horizon_series = _map_param("horizon_bars", float(base_h))
                if trail_mult_series is not None:
                    trail_mult_series = _map_param("trail_distance_atr_mult", float(base_trail))
        except Exception:
            pass
    
    if (
        bool(enable_regime_conditional)
        and barrier_geometry_by_regime is None
        and regime_scalar_for_barriers is not None
    ):
        try:
            s = regime_scalar_for_barriers.reindex(market_index).astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # Tunable regime scaling strength/power
            try:
                barrier_regime_strength = float(params.get("barrier_regime_strength", 1.0))
            except Exception:
                barrier_regime_strength = 1.0
            if not np.isfinite(barrier_regime_strength):
                barrier_regime_strength = 1.0
            barrier_regime_strength = float(np.clip(barrier_regime_strength, 0.0, 1.0))
            
            try:
                barrier_regime_power = float(params.get("barrier_regime_power", 1.0))
            except Exception:
                barrier_regime_power = 1.0
            if not np.isfinite(barrier_regime_power) or barrier_regime_power <= 1e-6:
                barrier_regime_power = 1.0
            barrier_regime_power = float(np.clip(barrier_regime_power, 0.25, 4.0))
            
            # Blend toward 1.0 when strength < 1
            s_eff = 1.0 + float(barrier_regime_strength) * (np.power(s.astype(float), float(barrier_regime_power)) - 1.0)
            s_eff = pd.to_numeric(s_eff, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # Apply scaling to geometry
            s_thr_s = stop_mult * s_eff
            
            # Volatility-of-Volatility Adjustment
            try:
                vol_vol_exp = float(params.get("barrier_vol_vol_exp", 0.0))
                if vol_vol_exp > 0.01:
                    if isinstance(s, pd.Series):
                        s_std = s.rolling(24).std().fillna(0.0)
                        s_mean = s.rolling(24).mean().replace(0, 1.0)
                        vol_of_vol = (s_std / s_mean).fillna(0.0)
                        vov_factor = 1.0 + vol_vol_exp * vol_of_vol
                        vov_factor = vov_factor.clip(1.0, 2.0)
                        s_thr_s = s_thr_s * vov_factor
            except Exception:
                pass
            
            # Barrier Asymmetry Regime Modulation
            try:
                barrier_asym = float(params.get("barrier_trend_asymmetry", 0.0))
            except Exception:
                barrier_asym = 0.0
            
            if barrier_asym > 0.01:
                asym_factor = np.where(s_eff > 1.0, 1.0 + barrier_asym * (s_eff - 1.0), 1.0)
                p_thr_s = rr_series * s_thr_s * asym_factor
            else:
                p_thr_s = rr_series * s_thr_s
            
            s_eff = s_eff.clip(lower=0.25, upper=4.0)
            
            stop_mult = stop_mult.astype(float) * s_eff
            if trail_mult_series is not None:
                trail_mult_series = trail_mult_series.astype(float) * s_eff
            horizon_series = horizon_series.astype(float) / s_eff
        except Exception:
            pass
    
    stop_mult = stop_mult.reindex(market_index).astype(float)
    rr_series = rr_series.reindex(market_index).astype(float)
    horizon_series = horizon_series.reindex(market_index).astype(float)
    try:
        if trail_mult_series is not None:
            trail_mult_series = trail_mult_series.reindex(market_index).astype(float)
    except Exception:
        trail_mult_series = None
    
    stop_thr = (stop_mult * atr_frac_series.reindex(market_index).astype(float)).astype(float)
    stop_thr = stop_thr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    profit_thr = (stop_thr * rr_series).astype(float)
    profit_thr = profit_thr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Enforce small positive floor
    profit_thr = profit_thr.clip(lower=float(min_profit_floor_local))
    stop_thr = stop_thr.clip(lower=float(min_profit_floor_local) / 2.0)
    
    # Horizon bounds
    horizon_series = horizon_series.replace([np.inf, -np.inf], np.nan).fillna(float(base_h))
    horizon_series = horizon_series.clip(lower=4.0, upper=256.0)
    
    if trail_mult_series is not None:
        trail_mult_series = trail_mult_series.replace([np.inf, -np.inf], np.nan).fillna(float(base_trail))
        trail_mult_series = trail_mult_series.clip(lower=0.0, upper=10.0)
    
    return profit_thr, stop_thr, horizon_series, trail_mult_series


def save_layer2_results(
    *,
    best_trading_params: Dict[str, Any],
    best_l2_score: float,
    l2_metrics: Dict[str, Any],
    layer2_search_space: Dict[str, Any],
    l2_history: List[Dict[str, Any]],
    config: Dict[str, Any],
    outcomes_dir: Path,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
) -> Tuple[Optional[Path], Optional[Path], Dict[str, Any]]:
    """Save Layer 2 optimization results.
    
    Args:
        best_trading_params: Best trading parameters
        best_l2_score: Best Layer 2 score
        l2_metrics: Layer 2 metrics dictionary
        layer2_search_space: Search space definition
        l2_history: Optimization history
        config: HPO configuration
        outcomes_dir: Directory for saving results
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe string
        direction: Trading direction
        
    Returns:
        Tuple of (params_path, history_path, stage_report)
    """
    timestamp = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Persist Layer 2 params
    l2_path: Optional[Path] = None
    try:
        l2_path = outcomes_dir / f"hpo_layer2_best_params_{symbol}_{timeframe}_{timestamp}.json"
        l2_payload = {
            "best_params": _sanitize_json_value(best_trading_params),
            "best_score": best_l2_score,
            "timestamp": timestamp,
        }
        l2_path.parent.mkdir(parents=True, exist_ok=True)
        with open(l2_path, "w") as f:
            json.dump(l2_payload, f, indent=2, default=str)
        tprint_info(f"   💾 Saved Layer 2 best params to {l2_path}")
    except Exception as l2_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 2 params: {l2_exc}")
    
    # Save Layer 2 History
    l2_history_path: Optional[Path] = None
    try:
        l2_history_path = outcomes_dir / f"hpo_layer2_history_{symbol}_{timeframe}_{timestamp}.json"
        with open(l2_history_path, "w") as f:
            json.dump(_sanitize_json_value(l2_history), f, indent=2, default=str)
        tprint_info(f"   💾 Saved Layer 2 history to {l2_history_path}")
    except Exception as e:
        tprint_warning(f"   ⚠️ Failed to save Layer 2 history: {e}")
    
    # Save Layer 2 trials CSV
    l2_trials_path: Optional[Path] = None
    try:
        if l2_history and len(l2_history) > 0:
            l2_trials_path = outcomes_dir / f"hpo_layer2_trials_{symbol}_{timeframe}_{timestamp}.csv"
            trial_rows = []
            for trial in l2_history:
                if isinstance(trial, dict):
                    row = {
                        "trial_number": trial.get("trial_number"),
                        "value": trial.get("value"),
                        **trial.get("params", {}),
                    }
                    trial_rows.append(row)
            if trial_rows:
                pd.DataFrame(trial_rows).to_csv(l2_trials_path, index=False)
                tprint_info(f"   💾 Saved Layer 2 trial metrics to {l2_trials_path}")
    except Exception as l2_trials_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 2 trial metrics: {l2_trials_exc}")
    
    # Write stage report
    stage_report: Dict[str, Any] = {}
    try:
        l2_extra = {}
        try:
            if isinstance(l2_metrics, dict):
                for k in ["valid_events", "n_trades", "sharpe_mean", "sharpe_std", "trades_per_day"]:
                    if k in l2_metrics:
                        l2_extra[k] = l2_metrics[k]
        except Exception:
            pass
        
        stage_report = _write_hpo_stage_report(
            outcomes_dir=outcomes_dir,
            run_timestamp=timestamp,
            stage_id="layer2_trading",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            best_params=_sanitize_json_value(dict(best_trading_params)) if isinstance(best_trading_params, dict) else {},
            metrics={
                "best_score": best_l2_score,
                **(l2_metrics if isinstance(l2_metrics, dict) else {}),
            },
            search_space=layer2_search_space,
            trials_csv_path=l2_trials_path,
            history_json_path=l2_history_path,
            extra=l2_extra,
        )
    except Exception as l2_report_exc:
        tprint_warning(f"   ⚠️ Failed to write Layer 2 report: {l2_report_exc}")
    
    return l2_path, l2_history_path, stage_report
