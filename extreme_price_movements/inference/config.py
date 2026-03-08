"""
Inference Configuration Loader.

This module loads all configuration needed for inference:
- Best parameters from offline optimizer (candidate thresholds, TBM params, etc.)
- Model paths using find_latest_run_id and load_full_state
- Other config parameters
"""

from typing import Dict, Any, Optional, List
import csv
from pathlib import Path

from extreme_price_movements.config import CFG
from extreme_price_movements.offline_optimisers.params_store import (
    apply_offline_optimizer_best_params,
    INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
)
from extreme_price_movements.model_loader import (
    find_latest_run_id,
    load_full_state,
    load_model_bundle,
)
from extreme_price_movements.utils import tprint

def _resolve_runtime_cfg() -> Dict[str, Any]:
    """Refresh runtime config from persisted offline optimiser outputs."""
    return apply_offline_optimizer_best_params(dict(CFG))


def _load_inference_candidate_mask_params() -> Dict[str, Any]:
    path = Path(INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV)
    if not path.exists():
        return {}
    try:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        return rows[-1]
    except Exception:
        return {}

# Default paths
DEFAULT_DATA_ROOT = "data"


def get_candidate_thresholds(thresholds_csv: Optional[str] = None) -> Dict[str, float]:
    """Load candidate thresholds from runtime config (populated by offline optimizer).
    
    Args:
        thresholds_csv: Deprecated parameter, kept for backward compatibility.
        
    Returns:
        Dictionary with threshold parameters:
        - extreme_pct: Percentage of top/bottom performers to consider
        - min_range_pct: Minimum 12h high/low range percentage
        - min_vol_zscore: Minimum volatility z-score threshold
    """
    runtime_cfg = _resolve_runtime_cfg()
    thresholds = {
        "extreme_pct": runtime_cfg.get("train_extreme_pct_hourly", 0.05),
        "min_move_12h_pct": runtime_cfg.get("train_min_move_12h_pct", 0.06),
        "min_range_pct": runtime_cfg.get("train_min_range_pct", 0.06),
        "min_vol_zscore": runtime_cfg.get("train_min_vol_zscore", 1.5),
        "metric": runtime_cfg.get("train_candidate_metric", "ret12h"),
    }
    infer_mask = _load_inference_candidate_mask_params()
    if infer_mask:
        if infer_mask.get("train_extreme_pct_hourly"):
            thresholds["extreme_pct"] = float(infer_mask["train_extreme_pct_hourly"])
        if infer_mask.get("train_min_move_12h_pct"):
            thresholds["min_move_12h_pct"] = float(infer_mask["train_min_move_12h_pct"])
        if infer_mask.get("train_min_vol_zscore"):
            thresholds["min_vol_zscore"] = float(infer_mask["train_min_vol_zscore"])
        if infer_mask.get("train_candidate_metric"):
            thresholds["metric"] = str(infer_mask["train_candidate_metric"])
    return thresholds


def load_inference_config(
    data_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load complete inference configuration.
    
    Args:
        data_root: Root data directory. If None, uses "data"
        run_id: Specific run ID to load. If None, finds latest run
        
    Returns:
        Dictionary with all config needed for inference:
        - run_id: The run ID used
        - thresholds: Candidate threshold parameters
        - tbm_params: TBM barrier parameters from offline optimizer
        - model_bundle: Loaded model bundle
        - full_state: Complete training state
        - data_root: Data root path
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    
    # Find latest run ID if not provided
    if run_id is None:
        run_id = find_latest_run_id(data_root)
        if run_id is None:
            raise ValueError("No run ID found and none provided")
    
    tprint(f"Loading inference config for run_id: {run_id}")
    
    # Load thresholds from runtime_cfg (populated by offline optimizer)
    runtime_cfg = _resolve_runtime_cfg()
    thresholds = get_candidate_thresholds()
    tprint(f"Using thresholds: {thresholds}")
    
    # Load TBM params from runtime_cfg
    tbm_params = get_tbm_params()
    tprint(f"Using TBM params: {tbm_params}")
    
    # Load model bundle
    model_bundle = load_model_bundle(run_id, data_root)
    
    # Load full state
    full_state = load_full_state(run_id, data_root)
    
    config = {
        "run_id": run_id,
        "thresholds": thresholds,
        "tbm_params": tbm_params,
        "model_bundle": model_bundle,
        "full_state": full_state,
        "data_root": data_root,
    }
    
    tprint(f"Inference config loaded successfully for run {run_id}")
    return config


def get_tbm_params() -> Dict[str, Any]:
    """Get TBM (Triple Barrier Model) parameters from runtime config.
    
    These parameters are populated by apply_offline_optimizer_best_params()
    and are the same parameters that were optimized during training.
    
    Returns:
        Dictionary with TBM barrier parameters:
        - barrier_k_tp: TP multiplier
        - barrier_sl_base_mult: SL as TP percentage
        - barrier_tp_lo: TP absolute lower bound
        - barrier_tp_hi: TP absolute upper bound
        - barrier_sl_lo: SL absolute lower bound
        - barrier_sl_hi: SL absolute upper bound
        - barrier_tp_base_pct: TP base percentage
        - barrier_tp_method: TP method
        - barrier_sl_method: SL method
        - barrier_atr_window: ATR window for barrier calculation
        - label_horizon_base: Base horizon for labels
        - label_horizon_scaling: Horizon scaling factor
        - barrier_mode: Barrier mode
    """
    # TBM parameters from runtime_cfg (populated by apply_offline_optimizer_best_params)
    tbm_keys = [
        "barrier_k_tp",
        "barrier_sl_base_mult",
        "barrier_tp_lo",
        "barrier_tp_hi",
        "barrier_sl_lo",
        "barrier_sl_hi",
        "barrier_tp_base_pct",
        "barrier_tp_abs_pct",
        "barrier_tp_method",
        "barrier_sl_method",
        "barrier_atr_window",
        "label_horizon_base",
        "label_horizon_scaling",
        "barrier_mode",
    ]
    
    runtime_cfg = _resolve_runtime_cfg()
    params = {}
    for key in tbm_keys:
        if key in runtime_cfg and runtime_cfg[key] is not None:
            params[key] = runtime_cfg[key]
    
    return params


def get_sample_weight_params() -> Dict[str, Any]:
    """Get sample weight parameters from runtime config.
    
    These parameters are populated by apply_offline_optimizer_best_params()
    and are the same parameters that were optimized during training.
    
    Returns:
        Dictionary with sample weight parameters:
        - sample_weight_component_alphas: Component alphas for sample weighting
        - sample_weight_component_alphas_base: Base component alphas
        - sample_weight_component_alphas_meta: Meta component alphas
        - sample_weight_vol_power: Volume power for sample weighting
        - sample_weight_distance_k: Distance parameter k
        - sample_weight_distance_min_dist: Minimum distance
        - sample_weight_recency_half_life_bars: Recency half-life in bars
    """
    # Sample weight parameters from runtime_cfg
    sample_weight_keys = [
        "sample_weight_component_alphas",
        "sample_weight_component_alphas_base",
        "sample_weight_component_alphas_meta",
        "sample_weight_vol_power",
        "sample_weight_distance_k",
        "sample_weight_distance_min_dist",
        "sample_weight_recency_half_life_bars",
    ]
    
    runtime_cfg = _resolve_runtime_cfg()
    params = {}
    for key in sample_weight_keys:
        if key in runtime_cfg and runtime_cfg[key] is not None:
            params[key] = runtime_cfg[key]
    
    return params


def get_runtime_cfg() -> Dict[str, Any]:
    """Get the full runtime config with all optimized parameters.
    
    Returns:
        Dictionary with all runtime config parameters including:
        - Candidate thresholds (extreme_pct, min_range_pct, min_vol_zscore)
        - TBM barrier parameters
        - Sample weight parameters
        - All other config from CFG
    """
    return _resolve_runtime_cfg()


def get_inference_defaults() -> Dict[str, Any]:
    """Get default inference parameters.
    
    Returns:
        Dictionary with default parameters for inference
    """
    return {
        # Data fetching
        "lookback_periods": 24 * 60,  # Number of 1h periods to look back (~2 months)
        "symbols_per_batch": 50,  # Symbols to fetch per batch
        
        # Feature generation
        "trend_sma_hours": 24 * 14,  # 14 days
        "gate_vol_lookback_hours": 24 * 7,  # 7 days
        "gate_trend_thr": 0.0,
        
        # Model inference
        "use_multi_horizon": True,
        
        # Execution
        "max_position_size": 0.1,  # 10% of capital
        "default_stop_loss_pct": 0.05,  # 5%
        "default_take_profit_pct": 0.15,  # 15%
    }


# Margin universe cache - lazy loaded
_MARGIN_UNIVERSE_CACHE = None


def get_margin_universe(exchange=None) -> List[str]:
    """Get list of margin-enabled symbols from cache.
    
    Args:
        exchange: Optional exchange instance (ignored, kept for API compatibility)
        
    Returns:
        List of margin-enabled trading symbols
    """
    global _MARGIN_UNIVERSE_CACHE
    
    if _MARGIN_UNIVERSE_CACHE is None:
        import json
        import os
        
        cache_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            ".margin_universe_cache.json"
        )
        
        # Try multiple possible locations
        possible_paths = [
            cache_path,
            os.path.join(os.path.dirname(__file__), ".margin_universe_cache.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), ".margin_universe_cache.json"),
            "/Users/remyroche/Documents/Ares/extreme_price_movements/.margin_universe_cache.json",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cache_path = path
                break
        
        tprint(f"Loading margin universe from: {cache_path}")
        
        with open(cache_path, 'r') as f:
            margin_data = json.load(f)
        
        # Extract symbols that have margin trading enabled
        _MARGIN_UNIVERSE_CACHE = [
            item["symbol"] for item in margin_data 
            if item.get("isMarginTradingAllowed", False)
        ]
        
        tprint(f"Loaded {len(_MARGIN_UNIVERSE_CACHE)} margin-enabled symbols")
    
    return _MARGIN_UNIVERSE_CACHE
