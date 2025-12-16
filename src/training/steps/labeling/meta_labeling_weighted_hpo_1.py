"""Meta-Labeling HPO Layer 1: Sample Weighting Optimization.

This module handles Layer 1 of the hierarchical HPO process:
- Optimizes sample weighting parameters for meta-labeling
- Uses uniqueness, magnitude, and learnability-based weighting
- Supports committee agreement factors

The weighting parameters control how individual training samples
are weighted based on:
1. Magnitude compression of returns
2. Learnability (how easy is the sample to predict)
3. Uniqueness (inverse of overlap with other events)
4. Cross-feature quality
5. Downside multiplier for negative samples
6. Committee agreement and magnitude factors
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
    compute_horizon_consistency,
    compute_multi_horizon_consistency,
    compute_label_agreement_consistency,
    compute_return_sign_consistency,
    compute_uniqueness,
    run_layer1_optimization,
)

# Import shared utilities
try:
    from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
        _write_hpo_stage_report,
        _find_latest_path,
        tprint_info,
        tprint_success,
        tprint_warning,
    )
except ImportError:
    # Fallback for standalone testing
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


# Default weighting parameters
DEFAULT_WEIGHTING_PARAMS: Dict[str, Any] = {
    'mag_compression': 0.8,
    'learn_slope': 10.0,
    'learn_center': 0.4,
    'uniq_intensity': 2.0,
    'exp_mag': 1.5,
    'exp_learn': 1.0,
    'exp_uniq': 1.5,
    'exp_cross': 1.0,
    'downside_multiplier': 1.0,
}


def compute_committee_weight_factors(
    *,
    baseline_t_events: pd.DatetimeIndex,
    committee_label_matrix_values: np.ndarray,
    committee_returns_matrix_values: np.ndarray,
    committee_confidence_matrix_values: Optional[np.ndarray],
    committee_event_idx: pd.DatetimeIndex,
    best_committee_params: Dict[str, Any],
    n_experts: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute committee agreement scores and magnitude factors for Layer 1.
    
    Args:
        baseline_t_events: Event timestamps for baseline returns
        committee_label_matrix_values: Matrix of expert labels (events x experts)
        committee_returns_matrix_values: Matrix of expert returns (events x experts)
        committee_confidence_matrix_values: Matrix of expert confidences (events x experts)
        committee_event_idx: Event index for committee matrices
        best_committee_params: Committee voting parameters
        n_experts: Number of experts
        
    Returns:
        Tuple of (agreement_scores, magnitude_factors) aligned to baseline_t_events
    """
    try:
        w_scalp = float(best_committee_params.get("w_scalp", 1.0))
        w_swing = float(best_committee_params.get("w_swing", 1.0))
        w_trend = float(best_committee_params.get("w_trend", 1.0))
        
        # Build weights vector matching matrix columns
        if n_experts > 6:
            # Include new expert weights
            w_breakout = float(best_committee_params.get("w_breakout", 0.5))
            w_vwap = float(best_committee_params.get("w_vwap_rev", 0.5))
            w_vol_shock = float(best_committee_params.get("w_vol_shock", 0.5))
            weights_vec = np.array(
                [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend,
                 w_breakout, w_vwap, w_vol_shock],
                dtype=float,
            )
        else:
            weights_vec = np.array(
                [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend],
                dtype=float,
            )
        
        weights_vec = np.where(np.isfinite(weights_vec) & (weights_vec >= 0.0), weights_vec, 0.0)
        if float(np.sum(weights_vec)) <= 1e-12:
            weights_vec = np.ones_like(weights_vec, dtype=float)
        
        lbl_mat = np.asarray(committee_label_matrix_values, dtype=float)
        ret_mat = np.asarray(committee_returns_matrix_values, dtype=float)
        conf_mat = committee_confidence_matrix_values
        if conf_mat is None:
            conf_mat = np.ones_like(ret_mat, dtype=float)
        conf_mat = np.asarray(conf_mat, dtype=float)
        conf_mat = np.where(np.isfinite(conf_mat) & (conf_mat >= 0.0), conf_mat, 0.0)
        
        fired = lbl_mat != 0.0
        fired_w = fired.astype(float) * conf_mat * weights_vec.reshape(1, -1)
        denom = np.sum(fired_w, axis=1).astype(float) + 1e-8
        
        sign_mat = np.where(fired, np.sign(lbl_mat), np.nan)
        sign_w = np.where(np.isfinite(sign_mat), sign_mat, 0.0) * conf_mat * weights_vec.reshape(1, -1)
        mean_sign = np.sum(sign_w, axis=1).astype(float) / denom
        agree = np.abs(mean_sign)
        agree = np.where(np.isfinite(agree), agree, 0.0)
        agree = np.clip(agree, 0.0, 1.0)
        
        # Coverage-adjust agreement: damp when few experts fired
        try:
            fired_simple = fired.astype(float)
            fired_weight = np.sum(fired_simple * weights_vec.reshape(1, -1), axis=1).astype(float)
            total_weight = float(np.sum(weights_vec)) + 1e-8
            coverage = fired_weight / total_weight
            coverage = np.where(np.isfinite(coverage), coverage, 0.0)
            coverage = np.clip(coverage, 0.0, 1.0)
            agree = agree * np.sqrt(coverage)
            agree = np.clip(agree, 0.0, 1.0)
        except Exception:
            pass
        
        # Compute magnitude factors
        abs_ret = np.abs(ret_mat)
        abs_ret = np.where(fired, abs_ret, np.nan)
        abs_w = np.where(np.isfinite(abs_ret), abs_ret, 0.0) * conf_mat * weights_vec.reshape(1, -1)
        mean_abs = np.sum(abs_w, axis=1).astype(float) / denom
        mean_abs = np.where(np.isfinite(mean_abs), mean_abs, 0.0)
        pos_abs = mean_abs[mean_abs > 0.0]
        med_abs = float(np.nanmedian(pos_abs)) if pos_abs.size > 0 else 0.0
        if np.isfinite(med_abs) and med_abs > 0.0:
            mag_factor = mean_abs / (med_abs + 1e-12)
        else:
            mag_factor = np.ones_like(mean_abs, dtype=float)
        
        # Align to baseline_t_events
        agreement_scores = (
            pd.Series(agree, index=pd.DatetimeIndex(committee_event_idx))
            .reindex(baseline_t_events)
            .fillna(0.0)
            .values.astype(float)
        )
        magnitude_factors = (
            pd.Series(mag_factor, index=pd.DatetimeIndex(committee_event_idx))
            .reindex(baseline_t_events)
            .fillna(1.0)
            .values.astype(float)
        )
        
        return agreement_scores, magnitude_factors
        
    except Exception as e:
        tprint_warning(f"⚠️ Failed to compute committee weight factors: {e}")
        return None, None


def run_layer1_weighting_optimization(
    *,
    symbol: str,
    timeframe: str,
    market_data: pd.DataFrame,
    baseline_returns_clean: pd.Series,
    baseline_t_events: pd.DatetimeIndex,
    config: Dict[str, Any],
    outcomes_dir: Path,
    exchange: str,
    direction: str,
    committee_agreement_scores: Optional[np.ndarray] = None,
    committee_mag_factors: Optional[np.ndarray] = None,
    start_rank: int = 0,
    stage_rank: Dict[str, int] = None,
    load_stage_best_params_fn: Optional[callable] = None,
    start_at_canonical: str = "layer0",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Layer 1 sample weighting optimization.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        market_data: Market OHLCV data
        baseline_returns_clean: Clean baseline returns series
        baseline_t_events: Event timestamps
        config: HPO configuration
        outcomes_dir: Directory for saving results
        exchange: Exchange name
        direction: Trading direction
        committee_agreement_scores: Optional committee agreement scores
        committee_mag_factors: Optional committee magnitude factors
        start_rank: Current start rank for layer skipping
        stage_rank: Dictionary mapping stage names to ranks
        load_stage_best_params_fn: Function to load best params for a stage
        start_at_canonical: Canonical name of starting stage
        
    Returns:
        Tuple of (best_weighting_params, stage_report)
    """
    if stage_rank is None:
        stage_rank = {"layer1": 2}
    
    tprint_info("🧪 Layer 1: Optimizing Sample Weighting Parameters...")
    
    layer1_loaded_from: Optional[str] = None
    
    # Check if we should skip this layer
    if stage_rank.get("layer1", 2) < start_rank and load_stage_best_params_fn is not None:
        loaded_params, loaded_path = load_stage_best_params_fn("layer1")
        best_weighting_params = dict(loaded_params or {})
        layer1_loaded_from = str(loaded_path) if loaded_path is not None else None
        tprint_info(
            f"♻️ Layer 1 skipped (start_at={start_at_canonical}); loaded best params from {layer1_loaded_from}"
        )
    else:
        if len(baseline_t_events) < 50:
            tprint_warning(f"⚠️ Too few baseline events ({len(baseline_t_events)}) for Layer 1. Using defaults.")
            best_weighting_params = DEFAULT_WEIGHTING_PARAMS.copy()
        else:
            try:
                best_weighting_params = run_layer1_optimization(
                    symbol=symbol,
                    timeframe=timeframe,
                    market_data=market_data,
                    labels=baseline_returns_clean,
                    committee_agreement_scores=committee_agreement_scores,
                    committee_mag_factors=committee_mag_factors,
                    n_trials=int(config.get("layer1_n_trials", 60)),
                    objective_mode=str(config.get("layer1_objective_mode", "proxy")),
                )
            except Exception as e:
                tprint_warning(f"⚠️ Layer 1 optimization failed: {e}. Using defaults.")
                best_weighting_params = DEFAULT_WEIGHTING_PARAMS.copy()
    
    tprint_success(f"✅ Layer 1 Complete. Best Weighting Params: {best_weighting_params}")
    
    # Persist Layer 1 params immediately
    l1_path: Optional[Path] = None
    try:
        ts = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        l1_path = Path("outcomes") / f"hpo_layer1_best_params_{symbol}_{timeframe}_{ts}.json"
        l1_payload = {
            "best_params": best_weighting_params,
            "timestamp": ts,
        }
        l1_path.parent.mkdir(parents=True, exist_ok=True)
        with open(l1_path, "w") as f:
            json.dump(l1_payload, f, indent=2, default=str)
        tprint_info(f"   💾 Saved Layer 1 best params to {l1_path}")
    except Exception as l1_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 1 params: {l1_exc}")
    
    # Write stage report
    stage_report: Dict[str, Any] = {}
    try:
        l1_trials_csv = _find_latest_path(
            outcomes_dir=outcomes_dir,
            pattern=f"hpo_layer1_trials_{symbol}_{timeframe}_*.csv",
        )
        stage_report = _write_hpo_stage_report(
            outcomes_dir=outcomes_dir,
            run_timestamp=str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
            stage_id="layer1_weighting",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            best_params=dict(best_weighting_params) if isinstance(best_weighting_params, dict) else {},
            metrics={
                "best_params_path": str(l1_path) if l1_path is not None else None,
            },
            search_space=None,
            trials_csv_path=l1_trials_csv,
            history_json_path=None,
        )
    except Exception as l1_report_exc:
        tprint_warning(f"   ⚠️ Failed to write Layer 1 report: {l1_report_exc}")
    
    return best_weighting_params, stage_report
