"""
Layer 3 Core Orchestration - Multi-Horizon Meta-Models (ExtraTrees + LGBM + XGB)

Main orchestration function for Layer 3 meta-modeling using 4 horizons (12/48 bars).
Enhanced with entropy bars for improved information-based sampling.
Replaced feature selection with CausalFeatureSieve for geometry-specific processing.
Replaced ORF with constrained ExtraTrees, LGBM, and XGBoost models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
from scipy.special import expit
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from .model_training import train_dual_head_models
from .utils import calculate_alpha_target, validate_feature_matrix, calculate_sample_weights_efficient, calculate_studentized_har_target
from .enhanced_reporting import EnhancedLayer3Reporter

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
        enhanced_df = df.copy()
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


def apply_mild_mp_clustering(
    X: pd.DataFrame,
    threshold: float = 0.98
) -> pd.DataFrame:
    """
    Phase 3: Mild MP-Clustering.
    Removes purely collinear clusters (correlation > threshold), keeping one redundant predictor.
    """
    tprint_info(f"🔍 Phase 3: Mild MP-Clustering (Threshold={threshold})...")

    if X.shape[1] < 2:
        return X

    # Compute correlation matrix
    corr = X.corr().abs().fillna(0)
    
    # Distance matrix
    dist = 1 - corr
    dist = dist.clip(0, 1) # Ensure valid range

    # Hierarchical Clustering
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            metric='precomputed',
            linkage='complete'
        ).fit(dist)

        labels = clustering.labels_
        n_clusters = len(np.unique(labels))

        selected_cols = []
        dropped_cols = []

        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_features = X.columns[cluster_indices]

            # Select feature with highest variance (or simple first one)
            # Variance check:
            variances = X[cluster_features].var()
            best_feature = variances.idxmax()

            selected_cols.append(best_feature)
            dropped_cols.extend([f for f in cluster_features if f != best_feature])

        tprint_success(f"   ✅ Reduced {X.shape[1]} -> {len(selected_cols)} features. Dropped {len(dropped_cols)} redundant features.")
        return X[selected_cols]

    except Exception as e:
        tprint_warning(f"   ⚠️ Mild MP-Clustering failed: {e}. Keeping all features.")
        return X

def select_best_model_per_task(
    models_dict: Dict[str, Any],
    y_target: np.ndarray,
    task_type: str,
    horizon: str
) -> Tuple[np.ndarray, str]:
    """
    Phase 9: Select best models (one per geometry + target type).
    Compares ET, LGBM, XGB outputs against y_target.
    """
    best_score = float('inf') if task_type == 'regression' else float('-inf')
    best_model_key = None
    best_pred = None

    # Candidate keys
    candidates = [f"et_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"lgbm_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"xgb_{horizon}_{'reg' if task_type == 'regression' else 'cls'}"]

    for key in candidates:
        if key not in models_dict:
            continue

        res = models_dict[key]
        pred = res['cate']

        # Calculate metric
        # Regression: MSE (lower is better) or IC (higher is better).
        # User usually likes IC for Alpha, but MSE is safer for 'selection' if scales match.
        # Let's use MSE for Reg, AUC for Cls.

        valid_mask = ~np.isnan(y_target) & ~np.isnan(pred)
        if np.sum(valid_mask) == 0:
            continue

        y_true = y_target[valid_mask]
        y_pred = pred[valid_mask]

        if task_type == 'regression':
            score = mean_squared_error(y_true, y_pred)
            if score < best_score:
                best_score = score
                best_model_key = key
                best_pred = pred
        else:
            # Classification: AUC (higher is better)
            # Ensure binary target
            try:
                score = roc_auc_score((y_true > 0).astype(int), y_pred)
            except ValueError:
                score = 0.5 # Single class?

            if score > best_score:
                best_score = score
                best_model_key = key
                best_pred = pred

    if best_model_key:
        tprint_info(f"   🏆 Best model for {horizon} {task_type}: {best_model_key} (Score: {best_score:.4f})")
        return best_pred, best_model_key
    else:
        tprint_warning(f"   ⚠️ No valid models found for {horizon} {task_type}")
        return np.zeros(len(y_target)), "none"


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
    
    tprint_info("🚀 Layer 3: Starting Multi-Horizon Meta-Models Pipeline (ET+LGBM+XGB)")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outcomes_dir = Path(cfg.get('outcomes_dir', 'outcomes'))
    outcomes_dir.mkdir(parents=True, exist_ok=True)

    df = oof_df.copy()

    # Phase 0: Entropy Bars
    tprint_info("🔧 Phase 0: Entropy Bars Integration")
    symbol = cfg.get('symbol', 'ETHUSDT')
    exchange = cfg.get('exchange', 'binance')
    
    if cfg.get('use_entropy_bars', True):
        df, entropy_bars_df = integrate_entropy_bars_into_layer3(df, symbol, exchange, cfg)
        cfg['entropy_bars_df'] = entropy_bars_df
    else:
        entropy_bars_df = pd.DataFrame()

    # Phase 1: Meta-Features
    tprint_info("🔧 Phase 1: Meta-Features Engineering")
    safe_base_cols = [c for c in base_model_cols if c in df.columns]
    try:
        from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
        df = generate_layer3_features(df, safe_base_cols)
    except Exception as e:
        tprint_warning(f"⚠️ Feature generation failed: {e}")

    # Phase 2: Targets + Volatility Weights
    tprint_info("📊 Phase 2: Targets + Volatility Weights")
    if net_returns is None:
        if 'close' in df.columns:
            net_returns = df['close'].pct_change().fillna(0)
        else:
            net_returns = pd.Series(0, index=df.index)
    
    ret_series = net_returns.reindex(df.index)
    vol_series = ret_series.rolling(24).std().fillna(0.001)

    # 12-bar targets
    y_alpha_12_series = calculate_studentized_har_target(ret_series, vol_series)
    y_alpha_12 = y_alpha_12_series.values
    y_prob_12 = (ret_series.values > 0).astype(np.int32)
    
    # 48-bar targets
    if 'close' in df.columns:
        ret_48 = df['close'].shift(-48) / df['close'] - 1
        vol_48 = ret_series.rolling(48).std().fillna(0.001)
        y_alpha_48_series = calculate_studentized_har_target(ret_48.fillna(0), vol_48.fillna(0))
        y_alpha_48 = y_alpha_48_series.values
        y_prob_48 = (ret_48.fillna(0) > 0).astype(np.int32)
    else:
        y_alpha_48 = y_alpha_12 * 1.5
        y_prob_48 = y_prob_12

    cfg['y_alpha_48'] = y_alpha_48
    cfg['y_prob_48'] = y_prob_48

    w_alpha = calculate_sample_weights_efficient(ret_series.values, vol_series.values, layer1_weights=layer1_weight.values if layer1_weight is not None else None)

    # Phase 3: Mild MP-Clustering (Feature Selection)
    exclude = set(base_model_cols) | {target_col, 'close', 'high', 'low', 'volume', 'regime_label'}
    meta_features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
    X_full = df[meta_features].copy()
    
    # Add base model columns back for training (important for stacking)
    for col in safe_base_cols:
        if col in df.columns:
            X_full[col] = df[col].reindex(X_full.index)

    # Apply Clustering
    X_clustered = apply_mild_mp_clustering(X_full, threshold=0.98)

    # Phase 4-8: Multi-Horizon Model Training
    # (Huber Teacher -> Rotation -> Train -> Optuna -> Predictions) handled in model_training.py
    tprint_info("🤖 Phase 4-8: Multi-Horizon Model Training")
    
    # Train 12-bar models
    model_results_12 = train_dual_head_models(
        X_clustered, y_alpha_12, y_prob_12, w_alpha, w_alpha, [], cfg, cfg.get('fast_mode', False)
    )
    
    # Train 48-bar models
    model_results_48 = train_dual_head_models(
        X_clustered, y_alpha_48, y_prob_48, w_alpha, w_alpha, [], cfg, cfg.get('fast_mode', False)
    )
    
    combined_models = {}
    combined_models.update(model_results_12['models'])
    combined_models.update(model_results_48['models'])

    # Phase 9: Select best models
    tprint_info("🏆 Phase 9: Select Best Models")

    best_pred_12_reg, best_key_12_reg = select_best_model_per_task(model_results_12['models'], y_alpha_12, 'regression', '12')
    best_pred_12_cls, best_key_12_cls = select_best_model_per_task(model_results_12['models'], y_prob_12, 'classification', '12')
    best_pred_48_reg, best_key_48_reg = select_best_model_per_task(model_results_48['models'], y_alpha_48, 'regression', '48')
    best_pred_48_cls, best_key_48_cls = select_best_model_per_task(model_results_48['models'], y_prob_48, 'classification', '48')

    # Phase 11: Save best models OOF predictions
    tprint_info("💾 Phase 11: Save Best Models OOF")
    
    # Helper to propagate predictions
    def propagate_simple(values, idx):
        return pd.Series(values, index=idx).reindex(df.index).fillna(0)

    # We need the indices from training (which might be subsets if splits were used, but here X_clustered is full)
    # The models return predictions aligned with input X (assuming no internal dropping)

    df['meta_alpha'] = propagate_simple(best_pred_12_reg, X_clustered.index)
    df['meta_prob'] = propagate_simple(best_pred_12_cls, X_clustered.index)

    # Save 48 bar outputs too if needed
    df['meta_alpha_48'] = propagate_simple(best_pred_48_reg, X_clustered.index)
    df['meta_prob_48'] = propagate_simple(best_pred_48_cls, X_clustered.index)

    # Legacy compatibility
    df['orf_cate'] = df['meta_alpha']
    df['orf_se'] = df['meta_prob'] * 0.1 # Placeholder

    # Store Best Model Keys in results
    best_models_info = {
        '12_reg': best_key_12_reg,
        '12_cls': best_key_12_cls,
        '48_reg': best_key_48_reg,
        '48_cls': best_key_48_cls
    }

    models_dict = {
        'all_models': combined_models,
        'best_models': best_models_info,
        'meta_features': meta_features,
        'entropy_bars': entropy_bars_df if not entropy_bars_df.empty else None
    }

    # Enhanced Reporting
    try:
        reporter = EnhancedLayer3Reporter(outcomes_dir=outcomes_dir)
        reporter.generate_all_reports(
            df=df,
            models=model_results_12, # Just pass one set for structure compatibility or handle dict
            geometry_metrics=cfg.get('geometry_metrics', []),
            meta_features=meta_features,
            target_col='meta_prob',
            config=cfg
        )
    except Exception as e:
        tprint_warning(f"⚠️ Enhanced Layer 3 reporting failed: {e}")

    tprint_success(f"🎉 Layer 3 Pipeline Complete! Best Models: {best_models_info}")
    
    return df, models_dict
