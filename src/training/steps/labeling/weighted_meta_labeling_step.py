"""
Weighted Meta-Labeling Step (Production Pipeline).

This step extends the feature_generation_meta_labeling_step with sample weighting
capabilities discovered by meta_labeling_hpo_sample_weighted.

Key additions:
1. Loads optimal weighting parameters from HPO output
2. Applies generate_weights_per_label for sample weighting during training
3. Uses calibration-adjusted position sizing for evaluation
4. Integrates with the weighted HPO pipeline

Usage:
    python src/launcher/ares_launcher.py \\
        --step weighted_meta_labeling \\
        --symbol ETHUSDT --exchange binance --timeframe 15m --direction long \\
        --execution-mode full

Prerequisites:
    Run meta_labeling_hpo_sample_weighted first to generate optimal parameters.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import roc_auc_score

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

# Import core functionality from feature_generation_meta_labeling_step
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    # Core labeling functions
    compute_realized_returns,
    generate_primary_signals,
    kalman_smooth_labels,
    compute_vol_scaled_returns_for_events,
    create_quantile_labels_from_vol_scaled_returns,
    create_regime_aware_quantile_labels_from_vol_scaled_returns,
    attach_rolling_hmm_regimes_to_market_data,
    # Feature engineering
    create_meta_features,
    build_meta_features_for_model,
    # Model training
    train_bagged_lgbm_with_kfold,
    create_base_models,
    # Calibration
    fit_probability_to_return_mapping,
    translate_to_targets_with_isotonic,
    # Diagnostics
    generate_diagnostics_report,
    compute_learnability_score,
    compute_label_entropy_score,
    # Constants
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    DEFAULT_TRANSACTION_COST,
    ECON_MIN_RETURN_MULTIPLE,
    # The base step class
    FeatureGenerationMetaLabelingStep,
)

# Import sample weighting utilities
from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
    compute_horizon_consistency,
    compute_uniqueness,
    run_layer1_optimization,
)

# Import Kalman/RTS functions from HPO module
from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
    generate_kalman_features,
    rts_smoother_1d,
    kalman_filter_1d,
    smooth_prices_rts,
    # Feature selection functions (De Prado pipeline)
    select_features_with_quality,
    calculate_feature_quality,
    calculate_time_robust_quality,
    calculate_all_feature_qualities,
    reduce_features_by_correlation,
    select_features_hierarchical,
    lgbm_magnitude_sweep,
    generate_multi_horizon_features,
    # Cross-feature interactions
    generate_cross_features,
    get_cross_feature_inventory,
    get_feature_inventory,
    # Caching utilities
    load_cached_feature_selection,
    save_feature_selection_cache,
    invalidate_feature_selection_cache,
)

logger = logging.getLogger(__name__)


# Default weighting parameters (fallback if HPO not run)
DEFAULT_WEIGHTING_PARAMS = {
    'mag_compression': 0.8,
    'learn_slope': 10.0,
    'learn_center': 0.4,
    'uniq_intensity': 1.0,
    'exp_mag': 1.0,
    'exp_learn': 1.0,
    'exp_uniq': 1.0,
    'exp_cross': 1.0,
    'downside_multiplier': 1.0,
}

# Default Kalman/RTS parameters (fallback if HPO not run)
DEFAULT_KALMAN_PARAMS = {
    'kalman_Q': 1e-4,  # Process noise
    'kalman_R': 0.01,   # Measurement noise
}


def _load_weighting_params_from_hpo(
    symbol: str,
    timeframe: str,
    direction: str = "long",
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Path]]:
    """Load weighting and Kalman parameters from the multi-stage HPO output.
    
    Searches for files matching:
        outcomes/hpo_multi_stage_best_params_{symbol}_*.json
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        direction: Trading direction (long/short)
        
    Returns:
        Tuple of (weighting_params, kalman_params, file_path)
    """
    outcomes_dir = Path("outcomes")
    if not outcomes_dir.exists():
        return DEFAULT_WEIGHTING_PARAMS.copy(), DEFAULT_KALMAN_PARAMS.copy(), None
    
    # Look for multi-stage HPO output first
    pattern = f"hpo_multi_stage_best_params_{symbol}_*.json"
    candidates = sorted(outcomes_dir.glob(pattern))
    
    # Also check for standard HPO output
    if not candidates:
        pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
        candidates = sorted(outcomes_dir.glob(pattern))
    
    if not candidates:
        return DEFAULT_WEIGHTING_PARAMS.copy(), DEFAULT_KALMAN_PARAMS.copy(), None
    
    latest = candidates[-1]
    try:
        with open(latest, "r") as f:
            hpo_cfg = json.load(f)
        
        # Extract weighting params from the full config
        weighting_params = {}
        weighting_keys = list(DEFAULT_WEIGHTING_PARAMS.keys())
        
        for key in weighting_keys:
            if key in hpo_cfg:
                weighting_params[key] = float(hpo_cfg[key])
        
        # Extract Kalman params
        kalman_params = DEFAULT_KALMAN_PARAMS.copy()
        if 'kalman_Q' in hpo_cfg:
            kalman_params['kalman_Q'] = float(hpo_cfg['kalman_Q'])
        if 'kalman_R' in hpo_cfg:
            kalman_params['kalman_R'] = float(hpo_cfg['kalman_R'])
        
        # If we found at least some params, use them (fill missing with defaults)
        merged_weighting = DEFAULT_WEIGHTING_PARAMS.copy()
        if weighting_params:
            merged_weighting.update(weighting_params)
        
        tprint_info(f"📊 Loaded params from {latest}")
        tprint_info(f"   Kalman: Q={kalman_params['kalman_Q']:.2e}, R={kalman_params['kalman_R']:.2e}")
        return merged_weighting, kalman_params, latest
        
    except Exception as e:
        tprint_warning(f"⚠️ Failed to load params from {latest}: {e}")
    
    return DEFAULT_WEIGHTING_PARAMS.copy(), DEFAULT_KALMAN_PARAMS.copy(), None


def compute_sample_weights_for_events(
    realized_returns: pd.Series,
    market_data: pd.DataFrame,
    weighting_params: Dict[str, Any],
    horizon: int = 12,
) -> np.ndarray:
    """Compute sample weights for labeled events using the weighted pipeline.
    
    Args:
        realized_returns: Series of realized returns (only labeled events)
        market_data: Full market data for computing bar-level features
        weighting_params: Parameters for generate_weights_per_label
        horizon: Lookahead horizon for consistency calculation
        
    Returns:
        Array of sample weights aligned with realized_returns
    """
    # Filter to valid (labeled) events
    valid_mask = ~realized_returns.isna()
    valid_returns = realized_returns[valid_mask]
    
    if len(valid_returns) < 10:
        return np.ones(len(realized_returns))
    
    t_events = valid_returns.index
    
    # Compute bar-level features
    close_series = market_data["close"]
    returns_series = close_series.pct_change().fillna(0.0)
    
    # Pre-calculate heavy features
    full_consistency = compute_horizon_consistency(close_series, horizon=horizon)
    full_volatility = returns_series.rolling(20).std().fillna(0.0)
    
    # Create t_events Series for uniqueness (with estimated end times)
    try:
        t_events_series = pd.Series(
            index=t_events,
            data=t_events + pd.Timedelta(minutes=15 * horizon)  # Assuming 15m bars
        )
    except Exception:
        t_events_series = pd.Series(index=t_events, data=t_events)
    
    # Align features to event timestamps
    consistency_aligned = full_consistency.reindex(t_events).fillna(0.0).values
    volatility_aligned = full_volatility.reindex(t_events).fillna(0.0).values
    uniqueness_aligned = compute_uniqueness(t_events_series, market_data.index)
    
    if isinstance(uniqueness_aligned, pd.Series):
        uniqueness_aligned = uniqueness_aligned.values
    
    # Generate weights
    weights = generate_weights_per_label(
        returns=valid_returns.values,
        t_events=t_events,
        close_series=None,
        consistency_scores=consistency_aligned,
        uniqueness_scores=uniqueness_aligned,
        vol_proxy=volatility_aligned,
        **weighting_params
    )
    
    # Map back to full index (non-labeled events get weight 0)
    full_weights = np.zeros(len(realized_returns))
    full_weights[valid_mask.values] = weights
    
    return full_weights


def train_weighted_bagged_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
    n_splits: int = 5,
    n_bags: int = 10,
    base_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[Any]]:
    """Train bagged LightGBM with sample weighting.
    
    Args:
        X: Feature matrix
        y: Binary labels
        sample_weights: Sample weights from generate_weights_per_label
        n_splits: Number of CV splits
        n_bags: Number of bagged estimators
        base_params: LightGBM parameters
        
    Returns:
        Tuple of (OOF predictions DataFrame, trained models list)
    """
    tprint_info("🔧 train_weighted_bagged_lgbm() called")
    tprint_info(f"   X_shape={X.shape}, n_splits={n_splits}, n_bags={n_bags}")
    
    if base_params is None:
        base_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'verbose': -1,
            'random_state': 42,
        }
    
    # Prepare output
    oof_probs = np.full(len(y), np.nan, dtype=float)
    models = []
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weights[train_idx]
        
        # Skip if insufficient class variety
        if len(np.unique(y_train.dropna())) < 2:
            continue
        
        # Train bagged ensemble for this fold
        fold_probs = []
        for bag_idx in range(n_bags):
            # Bootstrap sample
            rng = np.random.RandomState(42 + fold_idx * 100 + bag_idx)
            n_train = len(X_train)
            boot_idx = rng.choice(n_train, size=n_train, replace=True)
            
            # Weighted bootstrap
            X_boot = X_train.iloc[boot_idx]
            y_boot = y_train.iloc[boot_idx]
            w_boot = w_train[boot_idx]
            
            # Train model
            model = lgb.LGBMClassifier(**base_params)
            try:
                model.fit(X_boot, y_boot, sample_weight=w_boot)
                probs = model.predict_proba(X_val)[:, 1]
                fold_probs.append(probs)
                models.append(model)
            except Exception as e:
                tprint_warning(f"⚠️ Bag {bag_idx} fold {fold_idx} failed: {e}")
                continue
        
        # Average predictions across bags
        if fold_probs:
            oof_probs[val_idx] = np.mean(fold_probs, axis=0)
    
    # Build output DataFrame
    oof_df = pd.DataFrame({
        'lgbm_bag_mean': oof_probs,
        'lgbm_bag_lower': oof_probs * 0.9,  # Approximate lower bound
    }, index=X.index)
    
    return oof_df, models


class WeightedMetaLabelingStep(FeatureGenerationMetaLabelingStep):
    """Production meta-labeling step with sample weighting from HPO.
    
    This step extends FeatureGenerationMetaLabelingStep by:
    1. Loading optimal weighting and Kalman parameters from HPO output
    2. Computing sample weights using generate_weights_per_label
    3. Generating Kalman-based features (KF_Close, KF_Velocity, KF_RSI, etc.)
    4. Training weighted bagged LightGBM models
    5. Using calibration-adjusted position sizing
    
    Kalman Features Added:
    - KF_Close, KF_High, KF_Low: Filtered OHLC using causal Kalman filter
    - KF_Velocity, KF_Acceleration: 1st/2nd derivatives of filtered close
    - KF_Slope: Rolling slope of filtered close
    - KF_P: Error covariance (uncertainty)
    - KF_RSI: RSI computed on filtered close
    - KF_BB_Distance: Distance from Kalman Bollinger Band
    - KF_Volume, KF_LogVolume_Slope, KF_Volume_Zscore, KF_Volume_Ratio, KF_Volume_P
    
    Config keys (in addition to base class):
    - use_hpo_weighting: bool - Whether to use HPO weighting params (default: True)
    - weighting_params: dict - Override weighting params (optional)
    - kalman_params: dict - Override Kalman Q/R params (optional)
    - weight_optimization_enabled: bool - Run Layer 1 optimization if HPO not found
    """
    
    def __init__(self, step_name: str = "weighted_meta_labeling") -> None:
        super().__init__(step_name)
        self.weighting_params = DEFAULT_WEIGHTING_PARAMS.copy()
        self.kalman_params = DEFAULT_KALMAN_PARAMS.copy()
        self.weighting_source = None
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute weighted meta-labeling pipeline.
        
        Steps:
        1. Load market data and generate primary signals
        2. Load or compute weighting parameters
        3. Generate labels with triple-barrier method
        4. Compute sample weights for training
        5. Train weighted bagged LGBM models
        6. Generate meta-probability outputs
        7. Save labeled data artifacts
        """
        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        
        tprint_info(
            f"🚀 Starting Weighted Meta-Labeling for {symbol}/{exchange} [{timeframe}] ({direction})"
        )
        
        # ------------------------------------------------------------------
        # 1. Load weighting and Kalman parameters from HPO
        # ------------------------------------------------------------------
        use_hpo_weighting = config.get("use_hpo_weighting", True)
        
        if use_hpo_weighting:
            self.weighting_params, self.kalman_params, hpo_path = _load_weighting_params_from_hpo(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
            )
            if hpo_path:
                self.weighting_source = str(hpo_path)
                tprint_success(f"✅ Using params from: {hpo_path}")
            else:
                tprint_warning("⚠️ No HPO params found, using defaults")
                self.weighting_source = "defaults"
        else:
            self.weighting_source = "config"
            if "weighting_params" in config:
                self.weighting_params.update(config["weighting_params"])
            if "kalman_params" in config:
                self.kalman_params.update(config["kalman_params"])
        
        tprint_info(f"   Weighting params: {self.weighting_params}")
        tprint_info(f"   Kalman params: Q={self.kalman_params['kalman_Q']:.2e}, R={self.kalman_params['kalman_R']:.2e}")
        
        # ------------------------------------------------------------------
        # 2. Load market data (delegate to base class)
        # ------------------------------------------------------------------
        pipeline_state: Dict[str, Any] = {}
        market_data, source = self.load_market_data_or_fail(
            config,
            pipeline_state,
            allow_config_override=True,
            skip_artifacts=True,
        )
        
        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            msg = "❌ No market data available for weighted meta-labeling"
            tprint_error(msg)
            return {"success": False, "error": msg, "metrics": {}, "artifacts": {}}
        
        tprint_info(f"   Loaded {len(market_data)} bars from {source}")
        
        # ------------------------------------------------------------------
        # 3. Generate primary signals
        # ------------------------------------------------------------------
        try:
            primary_signals = generate_primary_signals(market_data.copy())
            n_signals = int((primary_signals != 0).sum())
            tprint_info(f"   Generated {n_signals} primary signals")
        except Exception as e:
            tprint_error(f"❌ Primary signal generation failed: {e}")
            return {"success": False, "error": str(e), "metrics": {}, "artifacts": {}}
        
        # ------------------------------------------------------------------
        # 4. Attach regimes (optional)
        # ------------------------------------------------------------------
        try:
            market_data = attach_rolling_hmm_regimes_to_market_data(
                market_data=market_data,
                config=config,
            )
        except Exception as e:
            tprint_warning(f"⚠️ Regime attachment failed: {e}")
        
        # ------------------------------------------------------------------
        # 5. Compute realized returns with triple-barrier
        # ------------------------------------------------------------------
        # Load HPO-calibrated TPSL parameters if available
        profit_thr = float(config.get("profit_threshold", DEFAULT_PROFIT_THRESHOLD))
        stop_thr = float(config.get("stop_threshold", DEFAULT_STOP_THRESHOLD))
        horizon = int(config.get("horizon_bars", 20))
        min_spacing = int(config.get("min_event_spacing", 2))
        tx_cost = float(config.get("transaction_cost", DEFAULT_TRANSACTION_COST))
        
        # Use volatility-adaptive thresholds
        log_ret = np.log(market_data["close"]).diff()
        volatility_1d = log_ret.rolling(96).std()
        vol_baseline = volatility_1d.rolling(96).mean()
        vol_factor = volatility_1d / (vol_baseline + 1e-8)
        
        adaptive_profit = profit_thr * vol_factor.clip(0.5, 2.0)
        adaptive_stop = stop_thr * vol_factor.clip(0.5, 2.0)
        
        tprint_info(f"   Computing realized returns (horizon={horizon}, spacing={min_spacing})")
        
        (
            realized_returns,
            binary_labels,
            exit_reasons,
            event_durations,
            mfe_series,
            mae_series,
            binary_labels_long,
            binary_labels_short,
        ) = compute_realized_returns(
            market_data,
            primary_signals,
            profit_threshold=adaptive_profit,
            stop_threshold=adaptive_stop,
            horizon=horizon,
            transaction_cost=tx_cost,
            min_event_spacing=min_spacing,
        )
        
        labeled_mask = ~binary_labels.isna()
        n_events = int(labeled_mask.sum())
        tprint_info(f"   Labeled events: {n_events}")
        
        if n_events < 100:
            tprint_error(f"❌ Insufficient labeled events ({n_events})")
            return {"success": False, "error": "insufficient_events", "metrics": {}, "artifacts": {}}
        
        # ------------------------------------------------------------------
        # 6. Compute sample weights
        # ------------------------------------------------------------------
        tprint_info("   Computing sample weights...")
        sample_weights = compute_sample_weights_for_events(
            realized_returns=realized_returns,
            market_data=market_data,
            weighting_params=self.weighting_params,
            horizon=horizon,
        )
        
        # Summarize weights
        valid_weights = sample_weights[labeled_mask.values]
        tprint_info(
            f"   Weight stats: mean={np.mean(valid_weights):.3f}, "
            f"std={np.std(valid_weights):.3f}, "
            f"min={np.min(valid_weights):.3f}, max={np.max(valid_weights):.3f}"
        )
        
        # ------------------------------------------------------------------
        # 7. Build meta-features
        # ------------------------------------------------------------------
        tprint_info("   Building meta-features...")
        volume_available = "volume" in market_data.columns
        
        _, meta_features, _, _ = build_meta_features_for_model(
            market_data=market_data,
            primary_signals=primary_signals,
            realized_returns=realized_returns,
            binary_labels=binary_labels,
            event_durations=event_durations,
            mfe_series=mfe_series,
            mae_series=mae_series,
            adaptive_stop_threshold=adaptive_stop,
            horizon=horizon,
            volume_available=volume_available,
            meta_feature_cfg=config.get("meta_feature_engineering", {}),
        )
        
        n_base_features = meta_features.shape[1]
        tprint_info(f"   Built {n_base_features} base features")
        
        # ------------------------------------------------------------------
        # 7b. Add Kalman-based features (WEIGHTED PIPELINE ONLY)
        # ------------------------------------------------------------------
        # Uses CAUSAL Kalman Filter for live-compatible features
        # (RTS is acausal and only used for label generation in HPO)
        tprint_info("   Generating Kalman-based features...")
        
        try:
            kalman_features = generate_kalman_features(
                market_data=market_data,
                kalman_Q=self.kalman_params['kalman_Q'],
                kalman_R=self.kalman_params['kalman_R'],
            )
            
            # Align indices and merge
            kalman_features_aligned = kalman_features.reindex(meta_features.index).fillna(0)
            
            for col in kalman_features_aligned.columns:
                meta_features[col] = kalman_features_aligned[col]
            
            n_kalman_features = len(kalman_features.columns)
            tprint_success(f"   ✅ Added {n_kalman_features} Kalman features")
        except Exception as kf_exc:
            tprint_warning(f"   ⚠️ Kalman feature generation failed: {kf_exc}")
            n_kalman_features = 0
        
        tprint_info(f"   Total raw features: {meta_features.shape[1]} ({n_base_features} base + {n_kalman_features} Kalman)")
        
        # ------------------------------------------------------------------
        # 7c. Quality-based feature selection
        # ------------------------------------------------------------------
        # Solves circular dependency: select features using unsupervised
        # Signal-to-Noise ratio rather than label-dependent metrics.
        target_feature_count = int(config.get("target_feature_count", 70))
        feature_correlation_threshold = float(config.get("feature_correlation_threshold", 0.85))
        enable_multi_horizon = config.get("enable_multi_horizon_features", True)
        enable_cross_features = config.get("enable_cross_features", True)
        use_hierarchical_selection = config.get("use_hierarchical_selection", True)
        use_lgbm_sweep = config.get("use_lgbm_sweep", True)
        lgbm_lookahead = int(config.get("lgbm_sweep_lookahead", 4))
        lgbm_max_features = int(config.get("lgbm_max_features", 200))
        quality_drop_percentile = float(config.get("quality_drop_percentile", 20.0))
        use_feature_cache = config.get("use_feature_selection_cache", True)
        force_recompute_features = config.get("force_recompute_features", False)
        
        horizon_config = config.get("feature_horizon_config", {
            "Short": 5,
            "Medium": 20,
            "Long": 60,
        })
        
        tprint_info("   Running De Prado feature selection pipeline...")
        try:
            meta_features, self._feature_quality_scores = select_features_with_quality(
                df_features=meta_features,
                target_n=target_feature_count,
                correlation_threshold=feature_correlation_threshold,
                generate_horizons=enable_multi_horizon,
                horizon_config=horizon_config,
                enable_cross_features=enable_cross_features,
                market_data=market_data,
                # De Prado pipeline parameters
                use_hierarchical=use_hierarchical_selection,
                use_lgbm_sweep=use_lgbm_sweep,
                lgbm_lookahead=lgbm_lookahead,
                lgbm_max_features=lgbm_max_features,
                quality_drop_percentile=quality_drop_percentile,
                # Caching parameters
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                use_cache=use_feature_cache,
                force_recompute=force_recompute_features,
            )
            tprint_success(f"   ✅ Selected {meta_features.shape[1]} features (target={target_feature_count})")
        except Exception as fs_exc:
            tprint_warning(f"   ⚠️ Feature selection failed: {fs_exc}. Using all features.")
            self._feature_quality_scores = {}
        
        # ------------------------------------------------------------------
        # 8. Train weighted bagged LGBM
        # ------------------------------------------------------------------
        tprint_info("   Training weighted bagged LGBM...")
        
        X = meta_features.loc[labeled_mask].fillna(0)
        y = binary_labels[labeled_mask]
        w = sample_weights[labeled_mask.values]
        
        oof_df, models = train_weighted_bagged_lgbm(
            X=X,
            y=y,
            sample_weights=w,
            n_splits=config.get("cv_splits", 5),
            n_bags=config.get("n_bags", 10),
        )
        
        # Compute AUC
        valid_oof = ~oof_df['lgbm_bag_mean'].isna()
        if valid_oof.sum() > 50 and len(y[valid_oof].unique()) >= 2:
            oof_auc = roc_auc_score(y[valid_oof], oof_df.loc[valid_oof, 'lgbm_bag_mean'])
            tprint_success(f"   ✅ OOF AUC: {oof_auc:.4f}")
        else:
            oof_auc = 0.5
        
        # ------------------------------------------------------------------
        # 9. Assemble labeled data output
        # ------------------------------------------------------------------
        tprint_info("   Assembling labeled data...")
        
        labeled_data = market_data.copy()
        labeled_data["realized_return"] = realized_returns
        labeled_data["binary_label"] = binary_labels
        labeled_data["binary_label_long"] = binary_labels_long
        labeled_data["binary_label_short"] = binary_labels_short
        labeled_data["exit_reason"] = exit_reasons
        labeled_data["event_duration_bars"] = event_durations
        labeled_data["target_sample_weight"] = sample_weights
        
        # Add meta-probability from weighted model
        labeled_data["meta_probability"] = np.nan
        labeled_data.loc[oof_df.index, "meta_probability"] = oof_df['lgbm_bag_mean']
        
        # Add bagged variants
        labeled_data["meta_probability_lgbm_bag_mean"] = np.nan
        labeled_data.loc[oof_df.index, "meta_probability_lgbm_bag_mean"] = oof_df['lgbm_bag_mean']
        
        labeled_data["meta_probability_lgbm_bag_lower"] = np.nan
        labeled_data.loc[oof_df.index, "meta_probability_lgbm_bag_lower"] = oof_df['lgbm_bag_lower']
        
        # Add metadata columns
        labeled_data["meta_probability_source"] = "weighted_lgbm_bag"
        labeled_data["labeled_data_schema_version"] = "2.0_weighted"
        labeled_data["labeling_timestamp"] = datetime.utcnow().isoformat()
        labeled_data["labeling_method_id"] = f"weighted_meta_labeling_{symbol}_{timeframe}"
        
        # ------------------------------------------------------------------
        # 10. Save artifacts
        # ------------------------------------------------------------------
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save labeled data CSV
        csv_path = outcomes_dir / f"weighted_labeled_data_{symbol}_{timeframe}_{timestamp}.csv"
        labeled_data.to_csv(csv_path)
        tprint_success(f"   ✅ Saved labeled data to {csv_path}")
        
        # Compile metrics
        metrics = {
            "oof_auc": float(oof_auc),
            "n_events": n_events,
            "n_features": meta_features.shape[1],
            "weighting_source": self.weighting_source,
            "weighting_params": self.weighting_params,
        }
        
        artifacts = {
            "labeled_data_csv": str(csv_path),
        }
        
        tprint_success(f"✅ Weighted Meta-Labeling complete (AUC={oof_auc:.4f}, events={n_events})")
        
        return {
            "success": True,
            "metrics": metrics,
            "artifacts": artifacts,
            "labeled_data": labeled_data,
        }


def register_weighted_meta_labeling_step() -> None:
    """Register the weighted meta-labeling step in the step registry."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("weighted_meta_labeling", WeightedMetaLabelingStep)
    step_registry.register("weighted_meta_labeling_step", WeightedMetaLabelingStep)
    
    tprint(
        "✅ Weighted meta-labeling step registered "
        "(aliases: weighted_meta_labeling, weighted_meta_labeling_step)",
        "SUCCESS"
    )


# Auto-register when module is imported
register_weighted_meta_labeling_step()
