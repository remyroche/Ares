"""
Layer 3 Core Orchestration - Multi-Horizon ORF Integrated with Entropy Bars and CausalFeatureSieve

Main orchestration function for Layer 3 meta-modeling using 4 ORF horizons.
Enhanced with entropy bars for improved information-based sampling.
Replaced feature selection with CausalFeatureSieve for geometry-specific processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
from scipy.special import expit

from .geometry_system import generate_geometries_adaptive
from .model_training import train_dual_head_models
from .utils import calculate_alpha_target, validate_feature_matrix, calculate_sample_weights_efficient, calculate_studentized_har_target
from .enhanced_reporting import EnhancedLayer3Reporter

# Import CausalFeatureSieve
try:
    from src.training.steps.labeling.causal_feature_sieve import CausalFeatureSieve
    CAUSAL_SIEVE_AVAILABLE = True
except ImportError as e:
    CAUSAL_SIEVE_AVAILABLE = False
    print(f"⚠️ CausalFeatureSieve not available: {e}")

# Import entropy bars functionality
try:
    from src.utils.entropy_bars import (
        fetch_1min_data_for_entropy_bars,
        generate_entropy_bars_from_ohlcv,
        calculate_specialized_entropy_features
    )
    ENTROPY_BARS_AVAILABLE = True
except ImportError as e:
    ENTROPY_BARS_AVAILABLE = False
    print(f"⚠️ Entropy bars not available: {e}")

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

logger = logging.getLogger(__name__)


def integrate_entropy_bars_into_layer3(
    df: pd.DataFrame,
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integrate entropy bars and specialized features into Layer 3.
    
    Args:
        df: Original DataFrame with market data
        symbol: Trading symbol
        exchange: Exchange name
        config: Configuration dictionary
        
    Returns:
        Tuple of (enhanced_df, entropy_bars_df)
    """
    if not ENTROPY_BARS_AVAILABLE:
        tprint_warning("⚠️ Entropy bars not available, using original data")
        return df, pd.DataFrame()
    
    cfg = config or {}
    
    try:
        # Fetch 1-minute data for entropy bar generation
        tprint_info("🔧 Fetching 1-minute data for entropy bar generation")
        
        # Determine date range from existing data
        if not df.empty and hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
        else:
            # Default to last 30 days if no date range available
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        min_data = fetch_1min_data_for_entropy_bars(
            symbol=symbol,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            data_dir=cfg.get('data_dir', 'historical_data')
        )
        
        if min_data is None or min_data.empty:
            tprint_warning("⚠️ No 1-minute data available, skipping entropy bars")
            return df, pd.DataFrame()
        
        # Generate entropy bars
        tprint_info("🔄 Generating entropy bars from 1-minute data")
        entropy_bars = generate_entropy_bars_from_ohlcv(
            ohlcv_data=min_data,
            n_bins=cfg.get('entropy_bins', 10),
            window_size=cfg.get('entropy_window', 100),
            target_minutes=cfg.get('entropy_target_minutes', 15),
            symbol=symbol,
            exchange=exchange
        )
        
        if entropy_bars.empty:
            tprint_warning("⚠️ Failed to generate entropy bars")
            return df, pd.DataFrame()
        
        # Calculate specialized entropy features
        tprint_info("🎯 Calculating specialized entropy features")
        entropy_features = calculate_specialized_entropy_features(
            entropy_bars=entropy_bars,
            base_model_updates=df,  # Use df as proxy for base model updates
            specialist_prices=df['close'] if 'close' in df.columns else None,
            volatility_window=cfg.get('volatility_window', 20)
        )
        
        # Merge entropy features back to main dataframe
        # Align on timestamp index
        enhanced_df = df.copy()
        
        # Forward-fill entropy features to match main dataframe timestamps
        for col in entropy_features.columns:
            enhanced_df[col] = entropy_features[col].reindex(enhanced_df.index, method='ffill').fillna(0)
        
        # Add entropy bar OHLCV data as additional columns
        entropy_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'n_minutes', 'entropy_contribution']
        for col in entropy_ohlcv_cols:
            if col in entropy_bars.columns:
                enhanced_df[f'entropy_{col}'] = entropy_bars[col].reindex(enhanced_df.index, method='ffill').fillna(
                    enhanced_df[col] if col in enhanced_df.columns else 0
                )
        
        tprint_success(f"✅ Integrated entropy bars: {len(entropy_bars)} bars, {len(entropy_features.columns)} features")
        
        return enhanced_df, entropy_bars
        
    except Exception as e:
        tprint_error(f"❌ Error integrating entropy bars: {e}")
        return df, pd.DataFrame()


def apply_causal_feature_sieve(
    X: pd.DataFrame,
    y: pd.Series,
    geometry: str,
    sample_weight: Optional[pd.Series] = None,
    fast_mode: bool = False
) -> pd.DataFrame:
    """
    Apply CausalFeatureSieve for geometry-specific feature selection.
    
    Args:
        X: Feature matrix
        y: Target series
        geometry: '12_bar' or '48_bar'
        sample_weight: Optional sample weights
        fast_mode: Use fast settings
        
    Returns:
        Selected feature matrix
    """
    if not CAUSAL_SIEVE_AVAILABLE:
        tprint_warning("⚠️ CausalFeatureSieve not available, using all features")
        return X
    
    try:
        # Initialize CausalFeatureSieve for specific geometry
        sieve = CausalFeatureSieve(geometry=geometry, seed=42)
        
        # Apply the 4-sieve pipeline
        X_selected = sieve.fit_transform(X, y, sample_weight)
        
        return X_selected
        
    except Exception as e:
        tprint_error(f"❌ CausalFeatureSieve failed: {e}")
        tprint_warning("⚠️ Falling back to all features")
        return X


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
    cfg = config if isinstance(config, dict) else {}
    cfg['base_model_cols'] = base_model_cols
    
    tprint_info("🚀 Layer 3: Starting Multi-Horizon ORF Meta-Models Pipeline with CausalFeatureSieve")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outcomes_dir = Path(cfg.get('outcomes_dir', 'outcomes'))
    outcomes_dir.mkdir(parents=True, exist_ok=True)

    df = oof_df.copy()

    # 0. Entropy Bars Integration
    tprint_info("🔧 PHASE 0: Entropy Bars Integration")
    symbol = cfg.get('symbol', 'ETHUSDT')
    exchange = cfg.get('exchange', 'binance')
    
    if cfg.get('use_entropy_bars', True):
        df, entropy_bars_df = integrate_entropy_bars_into_layer3(df, symbol, exchange, cfg)
        cfg['entropy_bars_df'] = entropy_bars_df
    else:
        tprint_info("⏭️ Skipping entropy bars (disabled in config)")
        entropy_bars_df = pd.DataFrame()

    # 1. Feature Engineering
    tprint_info("🔧 PHASE 1: Feature Engineering")
    safe_base_cols = [c for c in base_model_cols if c in df.columns]
    try:
        from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
        df = generate_layer3_features(df, safe_base_cols)
    except Exception as e:
        tprint_warning(f"⚠️ Feature generation failed: {e}")

    # 2. Data Preparation (12 vs 48 bars)
    tprint_info("📊 PHASE 2: Data Preparation")
    if net_returns is None:
        if 'close' in df.columns:
            net_returns = df['close'].pct_change().fillna(0)
        else:
            net_returns = pd.Series(0, index=df.index)
    
    ret_series = net_returns.reindex(df.index)
    vol_series = ret_series.rolling(24).std().fillna(0.001)

    # 12-bar targets (Studentized HAR-Residual)
    tprint_info("🎯 Calculating Studentized HAR-Residual Target (12-bar)")
    y_alpha_12_series = calculate_studentized_har_target(ret_series, vol_series)
    y_alpha_12 = y_alpha_12_series.values
    y_prob_12 = (ret_series.values > 0).astype(np.int32)
    
    # 48-bar targets (Studentized HAR-Residual)
    if 'close' in df.columns:
        ret_48 = df['close'].shift(-48) / df['close'] - 1
        vol_48 = ret_series.rolling(48).std().fillna(0.001)

        tprint_info("🎯 Calculating Studentized HAR-Residual Target (48-bar)")
        # Note: We align NaN handling with original logic
        y_alpha_48_series = calculate_studentized_har_target(
            ret_48.fillna(0),
            vol_48.fillna(0)
        )
        y_alpha_48 = y_alpha_48_series.values

        y_prob_48 = (ret_48.fillna(0) > 0).astype(np.int32)
    else:
        y_alpha_48 = y_alpha_12 * 1.5
        y_prob_48 = y_prob_12

    cfg['y_alpha_48'] = y_alpha_48
    cfg['y_prob_48'] = y_prob_48

    # 3. Sample Weights
    w_alpha = calculate_sample_weights_efficient(ret_series.values, vol_series.values, layer1_weights=layer1_weight.values if layer1_weight is not None else None)

    # 4. Feature Matrix Preparation
    tprint_info("📊 PHASE 3: Feature Matrix Preparation")
    exclude = set(base_model_cols) | {target_col, 'close', 'high', 'low', 'volume', 'regime_label'}
    meta_features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
    
    X_full = df[meta_features].copy()
    tprint_info(f"📊 Full feature matrix: {X_full.shape}")
    
    # Add base model columns back for ORF training
    for col in safe_base_cols:
        if col in df.columns:
            X_full[col] = df[col].reindex(X_full.index)
    
    # 5. Geometry-Specific Feature Selection with CausalFeatureSieve
    tprint_info("🔍 PHASE 4: Geometry-Specific Feature Selection with CausalFeatureSieve")
    
    # Prepare target series for feature selection
    y_alpha_12_series = pd.Series(y_alpha_12, index=df.index)
    y_alpha_48_series = pd.Series(y_alpha_48, index=df.index)
    
    # Prepare sample weights
    sample_weight_series = None
    if sample_weight is not None:
        sample_weight_series = pd.Series(sample_weight, index=df.index)
    
    # Apply CausalFeatureSieve for 12-bar geometry
    tprint_info("🎯 12-Bar Geometry (Impulse) Feature Selection")
    X_12_selected = apply_causal_feature_sieve(
        X_full, 
        y_alpha_12_series, 
        geometry='12_bar',
        sample_weight=sample_weight_series,
        fast_mode=cfg.get('fast_mode', False)
    )
    
    # Apply CausalFeatureSieve for 48-bar geometry  
    tprint_info("🎯 48-Bar Geometry (Structural) Feature Selection")
    X_48_selected = apply_causal_feature_sieve(
        X_full,
        y_alpha_48_series,
        geometry='48_bar', 
        sample_weight=sample_weight_series,
        fast_mode=cfg.get('fast_mode', False)
    )
    
    # Validate selected feature matrices
    X_12_valid, y_alpha_12_valid = validate_feature_matrix(X_12_selected, y_alpha_12_series.loc[X_12_selected.index])
    X_48_valid, y_alpha_48_valid = validate_feature_matrix(X_48_selected, y_alpha_48_series.loc[X_48_selected.index])
    
    tprint_info(f"📊 Final feature matrices:")
    tprint_info(f"   - 12-bar: {X_12_valid.shape[1]} features, {X_12_valid.shape[0]} samples")
    tprint_info(f"   - 48-bar: {X_48_valid.shape[1]} features, {X_48_valid.shape[0]} samples")

    # 6. Model Training (ORF) - Geometry Specific
    tprint_info("🤖 PHASE 5: 4-Horizon ORF Model Training")
    
    # Train 12-bar models
    tprint_info("🎯 Training 12-bar ORF models")
    model_results_12 = train_dual_head_models(
        X_12_valid, y_alpha_12_valid, y_prob_12, w_alpha, w_alpha, [], cfg, cfg.get('fast_mode', False)
    )
    
    # Train 48-bar models
    tprint_info("🎯 Training 48-bar ORF models") 
    model_results_48 = train_dual_head_models(
        X_48_valid, y_alpha_48_valid, y_prob_48, w_alpha, w_alpha, [], cfg, cfg.get('fast_mode', False)
    )
    
    # Combine model results
    combined_models = {
        'orf_12_reg': model_results_12['models']['orf_12_reg'],
        'orf_12_cls': model_results_12['models']['orf_12_cls'], 
        'orf_48_reg': model_results_48['models']['orf_48_reg'],
        'orf_48_cls': model_results_48['models']['orf_48_cls']
    }
    model_results = {'models': combined_models}

    # 7. Propagation
    tprint_info("📊 PHASE 6: Model Propagation")
    
    # Horizon 12
    df['orf_cate_12_reg'] = pd.Series(model_results['models']['orf_12_reg']['cate'], index=X_12_valid.index).reindex(df.index).fillna(0)
    df['orf_se_12_reg'] = pd.Series(model_results['models']['orf_12_reg']['se'], index=X_12_valid.index).reindex(df.index).fillna(1.0)
    df['orf_cate_12_cls'] = pd.Series(model_results['models']['orf_12_cls']['cate'], index=X_12_valid.index).reindex(df.index).fillna(0)
    df['orf_se_12_cls'] = pd.Series(model_results['models']['orf_12_cls']['se'], index=X_12_valid.index).reindex(df.index).fillna(1.0)
    
    # Horizon 48
    df['orf_cate_48_reg'] = pd.Series(model_results['models']['orf_48_reg']['cate'], index=X_48_valid.index).reindex(df.index).fillna(0)
    df['orf_se_48_reg'] = pd.Series(model_results['models']['orf_48_reg']['se'], index=X_48_valid.index).reindex(df.index).fillna(1.0)
    df['orf_cate_48_cls'] = pd.Series(model_results['models']['orf_48_cls']['cate'], index=X_48_valid.index).reindex(df.index).fillna(0)
    df['orf_se_48_cls'] = pd.Series(model_results['models']['orf_48_cls']['se'], index=X_48_valid.index).reindex(df.index).fillna(1.0)

    # Legacy compatibility
    df['meta_alpha'] = df['orf_cate_12_reg']
    df['meta_prob'] = expit(df['orf_cate_12_cls'] / (df['orf_cate_12_cls'].std() + 1e-9))
    df['orf_cate'] = df['meta_alpha']
    df['orf_se'] = df['orf_se_12_reg']

    models_dict = {
        'orf_models': model_results['models'],
        'meta_features': meta_features,
        'selected_features_12': X_12_valid.columns.tolist(),
        'selected_features_48': X_48_valid.columns.tolist(),
        'entropy_bars': entropy_bars_df if not entropy_bars_df.empty else None,
        'causal_sieve_applied': True
    }

    # 8. Enhanced Reporting
    try:
        reporter = EnhancedLayer3Reporter(outcomes_dir=outcomes_dir)
        reporter.generate_all_reports(
            df=df,
            models=model_results,
            geometry_metrics=cfg.get('geometry_metrics', []),
            meta_features=meta_features,
            target_col='orf_cate_12_cls' if 'orf_cate_12_cls' in df.columns else target_col,
            config=cfg
        )
        
        # Add CausalFeatureSieve reporting
        if CAUSAL_SIEVE_AVAILABLE:
            tprint_info("📊 Generating CausalFeatureSieve summary report")
            sieve_report_path = outcomes_dir / f"causal_feature_sieve_summary_{ts}.md"
            with open(sieve_report_path, 'w') as f:
                f.write(f"# CausalFeatureSieve Summary Report\n\n")
                f.write(f"## Geometry-Specific Feature Selection\n\n")
                f.write(f"### 12-Bar Geometry (Impulse)\n")
                f.write(f"- Features selected: {len(X_12_valid.columns)}\n")
                f.write(f"- Feature reduction rate: {(1 - len(X_12_valid.columns)/len(X_full.columns)):.1%}\n")
                f.write(f"- Selected features: {X_12_valid.columns.tolist()}\n\n")
                f.write(f"### 48-Bar Geometry (Structural)\n")
                f.write(f"- Features selected: {len(X_48_valid.columns)}\n") 
                f.write(f"- Feature reduction rate: {(1 - len(X_48_valid.columns)/len(X_full.columns)):.1%}\n")
                f.write(f"- Selected features: {X_48_valid.columns.tolist()}\n\n")
                f.write(f"## Pipeline Configuration\n")
                f.write(f"- Entropy bars: {'Enabled' if cfg.get('use_entropy_bars', True) else 'Disabled'}\n")
                f.write(f"- Fast mode: {cfg.get('fast_mode', False)}\n")
                f.write(f"- Total samples: {len(df)}\n")
                f.write(f"- Total initial features: {len(X_full.columns)}\n")
                
        # Add entropy bars reporting if available
        if not entropy_bars_df.empty:
            tprint_info("📊 Generating entropy bars summary report")
            entropy_report_path = outcomes_dir / f"entropy_bars_summary_{ts}.md"
            with open(entropy_report_path, 'w') as f:
                f.write(f"# Entropy Bars Summary Report\n\n")
                f.write(f"## Generation Parameters\n")
                f.write(f"- Symbol: {symbol}\n")
                f.write(f"- Exchange: {exchange}\n")
                f.write(f"- Target Minutes: {cfg.get('entropy_target_minutes', 15)}\n")
                f.write(f"- Bins: {cfg.get('entropy_bins', 10)}\n")
                f.write(f"- Window Size: {cfg.get('entropy_window', 100)}\n\n")
                f.write(f"## Results\n")
                f.write(f"- Total Entropy Bars: {len(entropy_bars_df)}\n")
                f.write(f"- Date Range: {entropy_bars_df.index.min()} to {entropy_bars_df.index.max()}\n")
                if 'entropy_contribution' in entropy_bars_df.columns:
                    f.write(f"- Average Entropy: {entropy_bars_df['entropy_contribution'].mean():.4f}\n")
                    f.write(f"- Entropy Std: {entropy_bars_df['entropy_contribution'].std():.4f}\n")
                if 'n_minutes' in entropy_bars_df.columns:
                    f.write(f"- Average Minutes per Bar: {entropy_bars_df['n_minutes'].mean():.1f}\n")
                
    except Exception as e:
        tprint_warning(f"⚠️ Enhanced Layer 3 reporting failed: {e}")

    tprint_success(f"🎉 Layer 3 ORF Complete! CausalFeatureSieve applied with geometry-specific feature selection.")
    tprint_success(f"📊 Final summary:")
    tprint_success(f"   - 12-bar models: {len(X_12_valid.columns)} features")
    tprint_success(f"   - 48-bar models: {len(X_48_valid.columns)} features")
    tprint_success(f"   - Entropy bars: {'Integrated' if not entropy_bars_df.empty else 'Not available'}")
    
    return df, models_dict
