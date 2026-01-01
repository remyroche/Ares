
"""
Layer 3 Core Orchestration

Main orchestration function for Layer 3 meta-modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from .geometry_system import generate_geometries_adaptive, select_best_geometries_adaptive
from .feature_engineering import enhance_layer3_features_optimized, hierarchical_feature_filtering
from .model_training import train_dual_head_models
from .weighting_system import create_enhanced_weighting_schemes
from .reporting import generate_layer3_reports
from .utils import finalize_sample_weights, calculate_alpha_target, validate_feature_matrix

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

logger = logging.getLogger(__name__)

def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    layer2_weight_quality: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Initialize config first 
    cfg = config if isinstance(config, dict) else {}
    
    # Memory optimization: avoid full copy, use view when possible
    df = oof_df.copy(deep=False) if cfg.get('fast_mode', False) else oof_df.copy()
    
    # Initialize timestamps and paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outcomes_dir = Path(cfg.get('outcomes_dir', 'outcomes'))
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Input data: {len(df)} rows, {len(base_model_cols)} base features")
    print(f"🎯 Target column: {target_col}")
    tprint_info(f"📂 Outcomes directory: {outcomes_dir}")
    tprint_info(f"🕐 Timestamp: {ts}")

    # ---------------------------------------------------------
    # 1. Feature Engineering with Layer 0 Integration
    # ---------------------------------------------------------
    tprint_info("🔧 PHASE 1: Enhanced Feature Engineering")
    
    # Base feature preparation
    safe_base_cols = [c for c in base_model_cols if c in df.columns]
    if safe_base_cols:
        df[safe_base_cols] = df[safe_base_cols].fillna(0.5)
        tprint_success(f"✅ Prepared {len(safe_base_cols)} base features")

    # Market data alignment
    if market_data is not None and isinstance(market_data, pd.DataFrame) and not market_data.empty:
        market_cols_added = []
        for c in ['volume', 'high', 'low', 'close']:
            if c in market_data.columns:
                df[c] = market_data[c].reindex(df.index)
                market_cols_added.append(c)
        tprint_success(f"✅ Aligned {len(market_cols_added)} market data columns: {market_cols_added}")

    # Generate standard Layer 3 features
    initial_feature_count = len(df.columns)

    # Fast mode: skip expensive feature generation
    if cfg.get('fast_mode', False):
        tprint_info("⚡ Fast mode: skipping standard Layer 3 feature generation")
        standard_features_added = 0
    else:
        try:
            from src.training.steps.labeling.label_based_layer_3 import generate_layer3_features
            df = generate_layer3_features(df, safe_base_cols)
            standard_features_added = len(df.columns) - initial_feature_count
            tprint_success(f"✅ Generated {standard_features_added} standard Layer 3 features")
        except Exception as e:
            tprint_warning(f"⚠️ Standard Layer 3 feature generation failed: {e}")
            standard_features_added = 0

    # Enhanced Layer 0 feature integration
    pre_layer0_count = len(df.columns)

    # Fast mode: skip expensive Layer 0 integration
    if cfg.get('fast_mode', False) or cfg.get('skip_layer0_integration', False):
        tprint_info("⚡ Skipping Layer 0 feature integration for speed")
        layer0_features_added = 0
    else:
        try:
            df = enhance_layer3_features_optimized(df, market_data, layer1_weight, fast_mode=cfg.get('fast_mode', False))
            layer0_features_added = len(df.columns) - pre_layer0_count
            tprint_success(f"✅ Enhanced with {layer0_features_added} Layer 0 optimized features")
            tprint_info(f"📈 Total features after Layer 0 integration: {len(df.columns)}")
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced feature generation failed: {e}")
            layer0_features_added = 0

    # ---------------------------------------------------------
    # 2. Data Preparation and Validation
    # ---------------------------------------------------------
    tprint_info("📊 PHASE 2: Data Preparation and Validation")
    
    # Prepare returns and volatility - vectorized
    if net_returns is None:
        if 'close' in df.columns:
            # Vectorized log returns calculation
            close_values = df['close'].values
            log_returns = np.zeros_like(close_values)
            log_returns[1:] = np.log(close_values[1:] / close_values[:-1])
            net_returns = pd.Series(log_returns, index=df.index)
            tprint_info("📈 Computed log returns from close prices")
        else:
            net_returns = pd.Series(0, index=df.index)
            tprint_warning("⚠️ No close column found, using zero returns")

    ret_series = net_returns.reindex(df.index)
    
    if 'volatility_1d' in df.columns:
        # Vectorized volatility processing
        vol_values = df['volatility_1d'].values
        vol_values = np.where(vol_values == 0, np.nan, vol_values)
        # Forward fill NaNs efficiently
        mask = np.isnan(vol_values)
        if np.any(mask):
            idx = np.where(~mask, np.arange(len(vol_values)), 0)
            np.maximum.accumulate(idx, out=idx)
            vol_values = vol_values[idx]
        vol_values = np.nan_to_num(vol_values, nan=0.001)
        vol_series = pd.Series(vol_values, index=df.index)
        tprint_success("✅ Using existing volatility_1d series")
    else:
        # Vectorized rolling volatility
        ret_values = ret_series.values
        vol_values = np.full_like(ret_values, 0.001, dtype=float)
        window = 24
        for i in range(window, len(ret_values)):
            vol_values[i] = np.std(ret_values[i-window:i])
        vol_series = pd.Series(vol_values, index=df.index)
        tprint_info("📊 Computed 24-bar rolling volatility")

    tprint_info(f"📈 Returns statistics: mean={ret_series.mean():.6f}, std={ret_series.std():.6f}")
    tprint_info(f"📊 Volatility statistics: mean={vol_series.mean():.6f}, std={vol_series.std():.6f}")

    # Define candidate features
    candidate_features = []
    candidate_features.extend(safe_base_cols)
    
    # Import centralized feature patterns
    from .feature_registry import get_layer3_feature_patterns, get_core_layer3_features
    
    # Add enhanced features from centralized registry
    enhanced_feature_patterns = get_layer3_feature_patterns()
    core_features = get_core_layer3_features()
    
    # Process feature patterns - highly vectorized and memory efficient
    pattern_counts = {}
    df_columns_array = np.array(df.columns.tolist(), dtype=str)  # Convert to numpy array for vectorized operations

    for pattern in enhanced_feature_patterns:
        # Vectorized string matching using numpy
        pattern_matches = np.char.find(df_columns_array, pattern) >= 0
        matching_cols = df_columns_array[pattern_matches].tolist()

        if matching_cols:
            candidate_features.extend(matching_cols)
            pattern_counts[pattern] = len(matching_cols)
    
    # Add core features
    for core_feat in core_features:
        if core_feat in df.columns:
            candidate_features.append(core_feat)
    
    meta_features = [c for c in list(dict.fromkeys(candidate_features)) if c in df.columns]
    
    # Clean features - vectorized operation for speed
    other_cols = [c for c in meta_features if c not in set(safe_base_cols)]
    if other_cols:
        # Vectorized cleaning
        X_other = df[other_cols].values
        inf_mask = np.isinf(X_other)
        inf_count = inf_mask.sum()
        nan_mask = np.isnan(X_other)
        nan_count = nan_mask.sum()

        # Replace inf and nan with 0.0
        X_other[inf_mask | nan_mask] = 0.0
        df[other_cols] = X_other

        tprint_info(f"🧹 Cleaned {inf_count} infinite and {nan_count} NaN values in features")

    tprint_success(f"📈 Final feature set: {len(meta_features)} total features")
    for pattern, count in pattern_counts.items():
        if count > 0:
            tprint_info(f"   - {pattern}: {count} features")

    # ---------------------------------------------------------
    # 3. Geometry Generation and Selection (4 Geometries)
    # ---------------------------------------------------------
    tprint_info("🔮 PHASE 3: Geometry Generation and Selection")
    
    # Get CUSUM signals for geometry generation
    cusum_cols = []
    cusum_df = None
    
    if market_data is not None:
        cusum_cols = [c for c in market_data.columns if 'trend_signal' in c or 'reversal_signal' in c]
        if cusum_cols:
            cusum_df = market_data[cusum_cols].reindex(df.index).fillna(0.0)
            tprint_success(f"✅ Found {len(cusum_cols)} CUSUM signals in market_data")
    
    if not cusum_cols:
        cusum_cols = [c for c in df.columns if 'trend_signal' in c or 'reversal_signal' in c]
        if cusum_cols:
            cusum_df = df[cusum_cols]
            tprint_info(f"📊 Found {len(cusum_cols)} CUSUM signals in main dataframe")
    
    if not cusum_cols or cusum_df is None:
        tprint_warning("⚠️ No CUSUM signals found, using fallback signals")
        cusum_df = pd.DataFrame(index=df.index)
        cusum_df['trend_signal_24'] = np.zeros(len(df))
        cusum_df['reversal_signal_24'] = np.zeros(len(df))
        cusum_cols = ['trend_signal_24', 'reversal_signal_24']

    # Generate geometries (4 by default, or fewer in fast mode)
    geometry_start_time = datetime.now()
    n_geoms = cfg.get('n_geometries', 2) if cfg.get('fast_mode', False) else cfg.get('n_geometries', 4)
    geometries_dict = generate_geometries_adaptive(
        base_signals=cusum_df,
        volatility=vol_series,
        mfe=df.get('mfe', pd.Series(0.02, index=df.index)),
        mae=df.get('mae', pd.Series(0.01, index=df.index)),
        n_geometries=n_geoms,
        fast_mode=cfg.get('fast_mode', False)
    )
    
    geometry_time = (datetime.now() - geometry_start_time).total_seconds()
    tprint_success(f"✅ Generated {len(geometries_dict)} geometries in {geometry_time:.2f}s")
    
    # Display geometry objectives
    for geom_id, geom_data in geometries_dict.items():
        tprint_info(f"   - {geom_id}: alpha={geom_data.get('alpha', 'N/A'):.2f}, activation={geom_data.get('activation', 'N/A')}")

    # Select best 4 geometries with correlation-based selection
    selection_start_time = datetime.now()
    y_target = df[target_col].fillna(0)
    
    top_k = cfg.get('top_k_geometries', 2) if cfg.get('fast_mode', False) else cfg.get('top_k_geometries', 4)
    selected_geoms_df = select_best_geometries_adaptive(
        geometries_dict, y_target, X=df[meta_features],
        model_type='classifier', objective_func='binary_logloss',
        top_k=top_k, correlation_weight=0.3
    )
    
    selection_time = (datetime.now() - selection_start_time).total_seconds()
    
    if selected_geoms_df.empty:
        tprint_warning("⚠️ No geometries selected, using fallback")
        selected_ids = ['fallback']
        geometries_dict['fallback'] = {
            'composite_signal': np.zeros(len(df)),
            'sigma_eff': vol_series.values,
            'alpha': 0.5,
            'activation': 'linear'
        }
    else:
        selected_ids = selected_geoms_df['id'].values.tolist()
        tprint_success(f"✅ Selected {len(selected_ids)} geometries in {selection_time:.2f}s: {selected_ids}")
        
        # Show selection scores
        if 'score' in selected_geoms_df.columns:
            for _, row in selected_geoms_df.iterrows():
                tprint_info(f"   - {row['id']}: score={row.get('score', 'N/A'):.4f}")

    # ---------------------------------------------------------
    # 4. Enhanced Weighting System
    # ---------------------------------------------------------
    tprint_info("⚖️  PHASE 4: Enhanced Weighting System")
    
    weighting_start_time = datetime.now()
    weighting_schemes = create_enhanced_weighting_schemes(
        layer1_weight, layer2_weight, layer2_weight_quality, market_data, df
    )
    
    weighting_time = (datetime.now() - weighting_start_time).total_seconds()
    
    tprint_success(f"✅ Created {len(weighting_schemes)} weighting schemes in {weighting_time:.2f}s:")
    for scheme_name, weights in weighting_schemes.items():
        weight_stats = f"mean={weights.mean():.3f}, std={weights.std():.3f}"
        tprint_info(f"   - {scheme_name}: {weight_stats}")

    # ---------------------------------------------------------
    # 5. Target and Weight Preparation
    # ---------------------------------------------------------
    tprint_info("🎯 PHASE 5: Target and Weight Preparation")
    
    # Alpha target: Volatility-standardized returns (JIT compiled)
    y_alpha = calculate_alpha_target(ret_series.values, vol_series.values)
    alpha_stats = f"mean={y_alpha.mean():.4f}, std={y_alpha.std():.4f}"
    tprint_info(f"📈 Alpha target: {alpha_stats}")
    
    # Probability target: Binary (highly vectorized)
    if target_col in df.columns:
        # Vectorized numeric conversion and thresholding
        target_values = df[target_col].values
        # Use numpy operations for better performance
        numeric_values = np.asarray(pd.to_numeric(target_values, errors='coerce'))
        # Fill NaN with 0.5 and convert to binary
        valid_mask = np.isfinite(numeric_values)
        numeric_values[~valid_mask] = 0.5
        y_prob = (numeric_values >= 0.5).astype(np.int32)
        positive_rate = np.mean(y_prob)
        tprint_info(f"🎯 Probability target: {positive_rate:.1%} positive rate")
    else:
        # Vectorized return-based target
        y_prob = (ret_series.values > 0).astype(np.int32)
        positive_rate = np.mean(y_prob)
        tprint_info(f"🎯 Probability target (from returns): {positive_rate:.1%} positive rate")

    # Optimized weight calculation using efficient JIT-compiled function
    layer1_weights = layer1_weight.reindex(df.index).fillna(1.0).values if layer1_weight is not None else None
    layer2_weights = pd.Series(layer2_weight).reindex(df.index).fillna(1.0).values if layer2_weight is not None else None

    # Calculate alpha weights (volatility-based)
    w_alpha = calculate_sample_weights_efficient(
        ret_series.values, vol_series.values,
        layer1_weights=layer1_weights
    )
    alpha_weight_stats = f"mean={w_alpha.mean():.3f}, std={w_alpha.std():.3f}"
    tprint_info(f"⚖️  Alpha weights: {alpha_weight_stats}")

    # Calculate probability weights (Layer 2 composite)
    if layer2_weights is not None:
        w_prob = calculate_sample_weights_efficient(
            ret_series.values, vol_series.values,
            layer1_weights=layer1_weights,
            layer2_weights=layer2_weights
        )
        tprint_success("✅ Using provided Layer 2 weights")
    else:
        w_prob = calculate_sample_weights_efficient(
            ret_series.values, vol_series.values,
            layer1_weights=layer1_weights
        )
        tprint_info("📊 Using uniform probability weights")
    
    prob_weight_stats = f"mean={w_prob.mean():.3f}, std={w_prob.std():.3f}"
    tprint_info(f"⚖️  Probability weights: {prob_weight_stats}")

    # ---------------------------------------------------------
    # 6. Cross-Validation Setup
    # ---------------------------------------------------------
    tprint_info("🔄 PHASE 6: Cross-Validation Setup")
    
    from src.utils.purged_kfold import PurgedKFoldTime
    
    n_splits = cfg.get('cv_folds', 3) if cfg.get('fast_mode', False) else cfg.get('cv_folds', 5)
    if len(df) < n_splits * 2:
        tprint_error(f"❌ Insufficient data for CV ({len(df)} rows < {n_splits*2} minimum)")
        fallback_df = df.copy()
        fallback_df['meta_alpha'] = 0.0
        fallback_df['meta_prob'] = 0.5
        return fallback_df, {'error': 'Insufficient data'}

    cv = PurgedKFoldTime(n_splits=n_splits, purge=100, embargo=50)
    splits = list(cv.split(df))
    
    tprint_success(f"✅ Created {n_splits}-fold purged CV with {len(splits)} splits")
    tprint_info(f"   - Purge: 100 bars, Embargo: 50 bars")
    
    # Show split sizes
    split_sizes = [len(val_idx) for _, val_idx in splits]
    tprint_info(f"   - Validation split sizes: {split_sizes}")

    # ---------------------------------------------------------
    # 7. Feature Selection (CMI + De Prado)
    # ---------------------------------------------------------
    
    enable_advanced_selection = cfg.get("enable_advanced_feature_selection", True)
    
    # Fast mode: skip expensive feature selection
    if cfg.get('fast_mode', False) or cfg.get('skip_feature_selection', False):
        tprint_info("⚡ Fast mode: skipping advanced feature selection")
    elif enable_advanced_selection and len(meta_features) > 20:
        tprint_info("🔍 PHASE 7: Advanced Feature Selection")
        
        selection_start_time = datetime.now()
        
        try:
            # Hierarchical filtering for performance
            pre_filter_count = len(meta_features)
            # Use first base model column instead of ensemble_prob for CMI filtering
            base_col_for_cmi = safe_base_cols[0] if safe_base_cols else None
            base_predictions_series = df[base_col_for_cmi] if base_col_for_cmi and base_col_for_cmi in df.columns else pd.Series(0.5, index=df.index)
            
            X_filtered = hierarchical_feature_filtering(
                df[meta_features],
                pd.Series(y_prob, index=df.index),
                base_predictions_series,
                fast_mode=cfg.get('fast_mode', False)
            )

            meta_features = X_filtered.columns.tolist()
            selection_time = (datetime.now() - selection_start_time).total_seconds()

            reduction_pct = (1 - len(meta_features) / pre_filter_count) * 100
            tprint_success(f"✅ Feature selection completed in {selection_time:.2f}s")
            tprint_info(f"📉 Features reduced: {pre_filter_count} → {len(meta_features)} ({reduction_pct:.1f}% reduction)")
        except Exception as e:
            tprint_warning(f"⚠️ Advanced feature selection failed: {e}")
            tprint_info(f"📊 Using all {len(meta_features)} features")
    else:
        if not enable_advanced_selection:
            tprint_info("📊 Advanced feature selection disabled in config")
        else:
            tprint_info(f"📊 Skipping advanced selection (only {len(meta_features)} features)")

    # ---------------------------------------------------------
    # 8. Dual-Head Model Training
    # ---------------------------------------------------------
    
    tprint_info("🤖 PHASE 8: Dual-Head Model Training")
    
    # Prepare feature matrix - avoid copy if possible for memory efficiency
    X = df[meta_features]  # Use view instead of copy to save memory
    tprint_info(f"📊 Feature matrix shape: {X.shape}")
    
    # Validate and clean
    validation_start_time = datetime.now()
    X, y_alpha_valid = validate_feature_matrix(X, pd.Series(y_alpha, index=df.index))
    X, y_prob_valid = validate_feature_matrix(X, pd.Series(y_prob, index=df.index))
    
    # Align weights
    w_alpha_aligned = w_alpha[:len(X)]
    w_prob_aligned = w_prob[:len(X)]
    
    validation_time = (datetime.now() - validation_start_time).total_seconds()
    tprint_success(f"✅ Data validation completed in {validation_time:.2f}s")
    tprint_info(f"📊 Final training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train models
    training_start_time = datetime.now()
    model_results = train_dual_head_models(
        X, y_alpha_valid, y_prob_valid, w_alpha_aligned, w_prob_aligned, splits, cfg, cfg.get('fast_mode', False)
    )
    training_time = (datetime.now() - training_start_time).total_seconds()
    
    tprint_success(f"✅ Dual-Head training completed in {training_time:.2f}s")
    
    # Add predictions to dataframe
    df['meta_alpha'] = model_results['alpha_oof']
    df['meta_prob'] = model_results['prob_oof']
    
    # Show model performance
    if 'alpha_metrics' in model_results:
        alpha_metrics = model_results['alpha_metrics']
        tprint_info(f"📈 Alpha Head Performance:")
        tprint_info(f"   - Final IC: {alpha_metrics.get('final_ic', 'N/A'):.4f}")
        tprint_info(f"   - Selected Models: {alpha_metrics.get('selected_models', [])}")
    
    if 'prob_metrics' in model_results:
        prob_metrics = model_results['prob_metrics']
        tprint_info(f"🎯 Probability Head Performance:")
        tprint_info(f"   - Final AUC: {prob_metrics.get('final_auc', 'N/A'):.4f}")
        tprint_info(f"   - Final LogLoss: {prob_metrics.get('final_logloss', 'N/A'):.4f}")
        tprint_info(f"   - Selected Models: {prob_metrics.get('selected_models', [])}")

    # ---------------------------------------------------------
    # 9. Reporting and Outputs
    # ---------------------------------------------------------
    
    tprint_info("📊 PHASE 9: Reporting and Outputs")
    
    # Prepare models dictionary
    models_dict = {
        'alpha_models': model_results['alpha_models'],
        'prob_models': model_results['prob_models'],
        'calibrated_models': model_results['calibrated_models'],
        'alpha_metrics': model_results['alpha_metrics'],
        'prob_metrics': model_results['prob_metrics'],
        'geometry_models': selected_ids,
        'meta_features': meta_features
    }
    
    # Generate comprehensive reports
    reporting_start_time = datetime.now()
    try:
        generate_layer3_reports(
            df, models_dict, [], meta_features, target_col, outcomes_dir, ts, cfg
        )
        reporting_time = (datetime.now() - reporting_start_time).total_seconds()
        tprint_success(f"✅ Reports generated in {reporting_time:.2f}s")
        tprint_success(f"📂 Reports saved to: {outcomes_dir}")
    except Exception as e:
        tprint_warning(f"⚠️ Report generation failed: {e}")

    # Final summary
    total_time = (datetime.now() - datetime.strptime(ts, "%Y%m%d_%H%M%S")).total_seconds()
    tprint_success(f"🎉 Layer 3 Complete! Total time: {total_time:.2f}s")
    tprint_success(f"📈 Generated {len(df)} rows with meta_alpha and meta_prob")
    tprint_success(f"🤖 Final models: {len(model_results['alpha_models'])} alpha, {len(model_results['prob_models'])} probability")
    tprint_success(f"🔮 Selected geometries: {len(selected_ids)} ({selected_ids})")
    tprint_success(f"📊 Final feature count: {len(meta_features)}")
    
    return df, models_dict
    