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
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, brier_score_loss

from .model_training import train_dual_head_models
from .utils import calculate_alpha_target, validate_feature_matrix, calculate_sample_weights_efficient, calculate_studentized_har_target, calculate_blended_forward_returns
from .enhanced_reporting import EnhancedLayer3Reporter
from .model_race_reporting import Layer3ModelRaceReporter
from .layer25_integration import integrate_layer25_into_layer3
from ..checkpoint_aware_runner import CheckpointAwareRunner, checkpoint_aware_step
from .feature_engineering import downcast_float
from src.training.steps.labeling.irm_regime_pipeline import (
    build_env_indices_for_index,
    get_or_fit_regime_labels
)

def generate_regime_aware_features(
    X: pd.DataFrame,
    volatility_col: str = 'volatility_20',
    prob_cols: Optional[List[str]] = None,
    asset_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generates Regime-Aware Features.
    1. Volatility Regime Probabilities (Z-Score Bucket).
    2. Meta-Model Disagreement (if prob_cols provided).
    3. Conservatism Scores.
    """
    tprint_info("Start: generate_regime_aware_features")
    X_reg = pd.DataFrame(index=X.index)

    # 1. Volatility Regime (Soft Gated)
    if volatility_col in X.columns:
        vol = X[volatility_col]
        # Robust Z-Score
        if asset_series is not None:
            med = (
                vol.groupby(asset_series)
                .rolling(100, min_periods=20)
                .median()
                .reset_index(level=0, drop=True)
            )
            mad = (
                (vol - med)
                .abs()
                .groupby(asset_series)
                .rolling(100, min_periods=20)
                .median()
                .reset_index(level=0, drop=True)
            )
        else:
            med = vol.rolling(100, min_periods=20).median()
            mad = (vol - med).abs().rolling(100, min_periods=20).median()

        z_score = (vol - med) / (mad * 1.4826 + 1e-6)

        # Softmax-like probabilities for regimes
        # Low Vol (Z < -1), Normal (-1 <= Z <= 1), High (Z > 1)
        # We use sigmoid to create soft features
        X_reg['regime_vol_z'] = z_score.fillna(0)
        X_reg['regime_prob_high_vol'] = expit(z_score - 1.0).fillna(0) # Prob Z > 1
        X_reg['regime_prob_low_vol'] = expit(-1.0 - z_score).fillna(0) # Prob Z < -1

    # 2. Meta-Disagreement
    if prob_cols:
        valid_probs = [c for c in prob_cols if c in X.columns]
        if len(valid_probs) > 1:
            probs = X[valid_probs]
            # Standard deviation of probabilities across models
            disagreement = probs.std(axis=1)
            X_reg['meta_disagreement'] = disagreement.fillna(0)

            # Conservatism: min(prob) / mean(prob)
            mean_prob = probs.mean(axis=1)
            min_prob = probs.min(axis=1)
            X_reg['meta_conservatism'] = (min_prob / (mean_prob + 1e-6)).fillna(0)

    tprint_success("End: generate_regime_aware_features")
    return X_reg

# Import Layer 3 Feature Cache
try:
    from src.training.steps.labeling.layer3_feature_cache import (
        save_layer3_features_to_cache,
        load_layer3_features_from_cache,
        should_use_cached_features
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️ Layer3 Feature Cache not available for core")

# Import optimized functions
try:
    from src.training.steps.labeling.optimized_layer2_functions import _vectorized_variance_scores
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    print("⚠️ Optimized Layer 2 functions not available for core")

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
    Includes caching to avoid re-generating expensive entropy calculations.
    """
    tprint_info("Start: integrate_entropy_bars_into_layer3")
    if not ENTROPY_BARS_AVAILABLE:
        raise RuntimeError("Entropy bars not available; Layer 3 requires entropy bars.")
    
    cfg = config or {}
    
    # Check cache
    cache_dir = Path(cfg.get('cache_dir', 'cache/entropy_bars'))
    cache_dir.mkdir(parents=True, exist_ok=True)

    asset_col = cfg.get("layer3_asset_id_col") or cfg.get("asset_id_col")
    asset_series = None
    asset_index_level = None
    time_index_level = None

    if asset_col and asset_col in df.columns:
        asset_series = df[asset_col]
    if asset_series is None:
        for candidate in ("asset_id", "asset", "ticker", "symbol"):
            if candidate in df.columns:
                asset_col = candidate
                asset_series = df[candidate]
                break
    if asset_series is None and isinstance(df.index, pd.MultiIndex):
        for candidate in ("asset_id", "asset", "ticker", "symbol"):
            if candidate in df.index.names:
                asset_index_level = candidate
                asset_col = candidate
                asset_series = df.index.get_level_values(candidate)
                break
        if asset_series is None:
            for i, level in enumerate(df.index.names):
                level_vals = df.index.get_level_values(i)
                if not pd.api.types.is_datetime64_any_dtype(level_vals):
                    asset_index_level = level
                    asset_col = level
                    asset_series = level_vals
                    break
    if isinstance(df.index, pd.MultiIndex):
        for level in df.index.names:
            level_vals = df.index.get_level_values(level)
            if pd.api.types.is_datetime64_any_dtype(level_vals):
                time_index_level = level
                break

    multi_asset = asset_series is not None and asset_series.nunique(dropna=False) > 1

    def _get_time_index(input_df: pd.DataFrame) -> pd.Index:
        if isinstance(input_df.index, pd.MultiIndex) and time_index_level is not None:
            return input_df.index.get_level_values(time_index_level)
        return input_df.index

    def _get_date_range(input_df: pd.DataFrame) -> Tuple[str, str]:
        time_index = _get_time_index(input_df)
        if not input_df.empty and hasattr(time_index, 'min') and hasattr(time_index, 'max'):
            start_dt_local = time_index.min()
            end_dt_local = time_index.max()
            return (
                start_dt_local.strftime('%Y-%m-%d'),
                end_dt_local.strftime('%Y-%m-%d'),
            )
        end_date_local = datetime.now().strftime('%Y-%m-%d')
        start_date_local = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        return start_date_local, end_date_local

    def _load_or_generate_entropy(
        asset_symbol: str,
        asset_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        start_date_local, end_date_local = _get_date_range(asset_df)
        cache_key = (
            f"{asset_symbol}_{exchange}_{start_date_local}_{end_date_local}_"
            f"{cfg.get('entropy_target_minutes', 15)}"
        )
        cache_file = cache_dir / f"entropy_features_{cache_key}.parquet"
        bars_cache_file = cache_dir / f"entropy_bars_{cache_key}.parquet"

        if cache_file.exists() and bars_cache_file.exists() and not cfg.get('force_refresh', False):
            try:
                tprint_info(f"♻️ Loading entropy features from cache: {cache_file}")
                entropy_features_local = pd.read_parquet(cache_file)
                entropy_bars_local = pd.read_parquet(bars_cache_file)
                return entropy_features_local, entropy_bars_local
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load entropy cache: {e}. Regenerating...")

        tprint_info("🔧 Fetching 1-minute data for entropy bar generation")
        min_data = fetch_1min_data_for_entropy_bars(
            symbol=asset_symbol,
            exchange=exchange,
            start_date=start_date_local,
            end_date=end_date_local,
            data_dir=cfg.get('data_dir', 'historical_data')
        )

        if min_data is None or min_data.empty:
            raise RuntimeError("No 1-minute data available for entropy bars.")

        tprint_info("🔄 Generating entropy bars from 1-minute data")
        entropy_bars_local = generate_entropy_bars_from_ohlcv(
            ohlcv_data=min_data,
            n_bins=cfg.get('entropy_bins', 10),
            window_size=cfg.get('entropy_window', 100),
            target_minutes=cfg.get('entropy_target_minutes', 15),
            symbol=asset_symbol,
            exchange=exchange
        )

        if entropy_bars_local.empty:
            raise RuntimeError("Failed to generate entropy bars.")

        tprint_info("🎯 Calculating specialized entropy features")
        specialist_prices = asset_df['close'] if 'close' in asset_df.columns else None
        entropy_features_local = calculate_specialized_entropy_features(
            entropy_bars=entropy_bars_local,
            base_model_updates=asset_df,
            specialist_prices=specialist_prices,
            volatility_window=cfg.get('volatility_window', 20)
        )

        try:
            entropy_features_local.to_parquet(cache_file)
            entropy_bars_local.to_parquet(bars_cache_file)
            tprint_success(f"💾 Saved entropy bars cache: {cache_file}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save entropy cache: {e}")

        return entropy_features_local, entropy_bars_local

    if multi_asset:
        tprint_info("🔀 Multi-asset data detected; generating per-asset entropy bars.")
        enhanced_groups = []
        entropy_bars_groups = []
        entropy_ohlcv_cols = [
            'open', 'high', 'low', 'close', 'volume', 'n_minutes', 'entropy_contribution'
        ]

        if asset_index_level:
            grouped = df.groupby(level=asset_index_level, sort=False)
        else:
            grouped = df.groupby(asset_col, sort=False)

        for asset_id, asset_df in grouped:
            asset_df_sorted = asset_df.sort_index()
            asset_symbol = str(asset_id)

            entropy_features, entropy_bars = _load_or_generate_entropy(asset_symbol, asset_df_sorted)

            time_index = _get_time_index(asset_df_sorted)
            aligned_features = entropy_features.reindex(time_index, method='ffill').fillna(0)
            aligned_bars = entropy_bars.reindex(time_index, method='ffill')

            for col in aligned_features.columns:
                asset_df_sorted[col] = aligned_features[col].to_numpy()

            for col in entropy_ohlcv_cols:
                if col in aligned_bars.columns:
                    asset_df_sorted[f'entropy_{col}'] = aligned_bars[col].to_numpy()

            entropy_bars_asset = entropy_bars.copy()
            entropy_bars_asset[asset_col] = asset_id
            entropy_bars_groups.append(entropy_bars_asset)
            enhanced_groups.append(asset_df_sorted)

        enhanced_df = pd.concat(enhanced_groups).sort_index()
        entropy_bars_df = pd.concat(entropy_bars_groups).sort_index()
        tprint_success("End: integrate_entropy_bars_into_layer3")
        return enhanced_df, entropy_bars_df

    # Determine date range from existing data
    start_date, end_date = _get_date_range(df)

    # Cache key based on symbol, exchange, dates and params
    cache_key = f"{symbol}_{exchange}_{start_date}_{end_date}_{cfg.get('entropy_target_minutes', 15)}"
    cache_file = cache_dir / f"entropy_features_{cache_key}.parquet"
    bars_cache_file = cache_dir / f"entropy_bars_{cache_key}.parquet"

    # Try loading from cache
    if cache_file.exists() and bars_cache_file.exists() and not cfg.get('force_refresh', False):
        try:
            tprint_info(f"♻️ Loading entropy features from cache: {cache_file}")
            entropy_features = pd.read_parquet(cache_file)
            entropy_bars = pd.read_parquet(bars_cache_file)

            # Align with requested dataframe if needed (though we usually reindex df to entropy bars)
            # Reindex DF to entropy bars index
            tprint_info("🧭 Resampling Layer 3 inputs to entropy bar timestamps")
            enhanced_df = df.reindex(entropy_bars.index, method='ffill')

            # Restore columns
            entropy_primary_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in entropy_primary_cols:
                if col in entropy_bars.columns:
                    enhanced_df[col] = entropy_bars[col]

            for col in entropy_features.columns:
                enhanced_df[col] = entropy_features[col].reindex(enhanced_df.index, method='ffill').fillna(0)

            entropy_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'n_minutes', 'entropy_contribution']
            for col in entropy_ohlcv_cols:
                if col in entropy_bars.columns:
                    enhanced_df[f'entropy_{col}'] = entropy_bars[col]

            tprint_success(f"✅ Loaded entropy bars from cache: {len(entropy_bars)} bars")
            tprint_success("End: integrate_entropy_bars_into_layer3")
            return enhanced_df, entropy_bars
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load entropy cache: {e}. Regenerating...")

    try:
        # Fetch 1-minute data for entropy bar generation
        tprint_info("🔧 Fetching 1-minute data for entropy bar generation")
        
        min_data = fetch_1min_data_for_entropy_bars(
            symbol=symbol,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            data_dir=cfg.get('data_dir', 'historical_data')
        )
        
        if min_data is None or min_data.empty:
            raise RuntimeError("No 1-minute data available for entropy bars.")
        
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
            raise RuntimeError("Failed to generate entropy bars.")
        
        # Calculate specialized entropy features
        tprint_info("🎯 Calculating specialized entropy features")
        entropy_features = calculate_specialized_entropy_features(
            entropy_bars=entropy_bars,
            base_model_updates=df,  # Use df as proxy for base model updates
            specialist_prices=df['close'] if 'close' in df.columns else None,
            volatility_window=cfg.get('volatility_window', 20)
        )

        tprint_info("🧭 Resampling Layer 3 inputs to entropy bar timestamps")
        enhanced_df = df.reindex(entropy_bars.index, method='ffill')
        entropy_primary_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in entropy_primary_cols:
            if col in entropy_bars.columns:
                enhanced_df[col] = entropy_bars[col]

        # Merge entropy features back to main dataframe
        for col in entropy_features.columns:
            enhanced_df[col] = entropy_features[col].reindex(enhanced_df.index, method='ffill').fillna(0)

        # Add entropy bar OHLCV data as additional columns
        entropy_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'n_minutes', 'entropy_contribution']
        for col in entropy_ohlcv_cols:
            if col in entropy_bars.columns:
                enhanced_df[f'entropy_{col}'] = entropy_bars[col]
        
        # Save to cache
        try:
            entropy_features.to_parquet(cache_file)
            entropy_bars.to_parquet(bars_cache_file)
            tprint_success(f"💾 Saved entropy bars cache: {cache_file}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save entropy cache: {e}")

        tprint_success(f"✅ Integrated entropy bars: {len(entropy_bars)} bars, {len(entropy_features.columns)} features")
        
        tprint_success("End: integrate_entropy_bars_into_layer3")
        return enhanced_df, entropy_bars
        
    except Exception as e:
        tprint_error(f"❌ Error integrating entropy bars: {e}")
        raise


def apply_mild_mp_clustering(
    X: pd.DataFrame,
    threshold: float = 0.98,
    target: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Phase 3: Mild MP-Clustering.
    Removes purely collinear clusters (correlation > threshold).
    Selection Strategy: Predictive Power (Correlation with Target) if target provided, else Variance.
    """
    tprint_info("Start: apply_mild_mp_clustering")
    tprint_info(f"🔍 Phase 3: Mild MP-Clustering (Threshold={threshold})...")

    if X.shape[1] < 2:
        tprint_success("End: apply_mild_mp_clustering")
        return X

    # Ensure float32 for speed
    X_vals = downcast_float(X).values

    # Compute correlation matrix using numpy
    if np.isnan(X_vals).any():
        X_vals = np.nan_to_num(X_vals, nan=0.0)

    # Compute correlation matrix (abs)
    corr = np.abs(np.corrcoef(X_vals.T))
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)
    
    # Distance matrix
    dist = 1.0 - corr
    dist = np.clip(dist, 0, 1)

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

        # Calculate Scores
        scores = np.zeros(X.shape[1])

        if target is not None:
            # Multi-Objective: IC (Predictive Power) * Stability (Proxy: Variance or Autocorr)
            # For speed, we use absolute correlation with target as proxy for IC
            y_vals = target.values if hasattr(target, 'values') else np.asarray(target)

            if len(y_vals) == len(X):
                # Valid target
                target_corrs = np.zeros(X.shape[1])
                for j in range(X.shape[1]):
                    # Check if constant
                    if np.std(X_vals[:, j]) > 1e-9:
                        target_corrs[j] = abs(np.corrcoef(X_vals[:, j], y_vals)[0, 1])

                scores = target_corrs
            else:
                scores = np.var(X_vals, axis=0) # Fallback to variance
        else:
            if OPTIMIZED_AVAILABLE:
                scores = _vectorized_variance_scores(X_vals)
            else:
                scores = np.var(X_vals, axis=0)

        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]

            if len(cluster_indices) == 1:
                selected_cols.append(X.columns[cluster_indices[0]])
                continue

            # Select feature with highest score
            cluster_scores = scores[cluster_indices]
            best_idx_in_cluster = np.argmax(cluster_scores)
            best_feature_idx = cluster_indices[best_idx_in_cluster]
            best_feature = X.columns[best_feature_idx]

            selected_cols.append(best_feature)
            dropped_cols.extend([X.columns[idx] for idx in cluster_indices if idx != best_feature_idx])

        tprint_success(f"   ✅ Reduced {X.shape[1]} -> {len(selected_cols)} features. Dropped {len(dropped_cols)} redundant features.")
        tprint_success("End: apply_mild_mp_clustering")
        return X[selected_cols]

    except Exception as e:
        tprint_warning(f"   ⚠️ Mild MP-Clustering failed: {e}. Keeping all features.")
        tprint_success("End: apply_mild_mp_clustering")
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
    Uses Information Coefficient (IC) for regression tasks to optimize for financial alpha.
    """
    tprint_info(f"Start: select_best_model_per_task (Task={task_type}, Horizon={horizon})")
    best_score = float('-inf')  # Use higher-is-better for both tasks (IC for reg, AUC for cls)
    best_model_key = None
    best_pred = None

    # Candidate keys
    candidates = [f"et_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"lgbm_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"xgb_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"catboost_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"huber_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"ridge_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"irm_ridge_{horizon}_{'reg' if task_type == 'regression' else 'cls'}",
                  f"irm_elasticnet_{horizon}_{'reg' if task_type == 'regression' else 'cls'}"]

    for key in candidates:
        if key not in models_dict:
            continue

        res = models_dict[key]
        pred = res['cate']

        # Calculate metric
        # Regression: Information Coefficient (IC) - higher is better for financial alpha
        # Classification: AUC - higher is better

        valid_mask = ~np.isnan(y_target) & ~np.isnan(pred)
        if np.sum(valid_mask) == 0:
            continue

        y_true = y_target[valid_mask]
        y_pred = pred[valid_mask]

        if task_type == 'regression':
            # Calculate Information Coefficient (Pearson correlation)
            try:
                # Use rank correlation for robustness to outliers
                ic = np.corrcoef(y_true, y_pred)[0, 1]
                score = ic
                # Handle NaN correlation
                if np.isnan(score) or np.isinf(score):
                    score = 0.0
            except:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_model_key = key
                best_pred = pred
        else:
            # Classification: Brier Score (lower is better) - de Prado standard
            # We prioritize calibration and probabilistic accuracy
            try:
                # Ensure binary target
                y_bin = (y_true > 0).astype(int)
                score = brier_score_loss(y_bin, y_pred)

                # Check bounds (Brier is [0, 1])
                if np.isnan(score): score = 1.0

            except ValueError:
                score = 1.0 # Error case

            # Lower Brier is better. Initialize best_score with float('inf') for cls too?
            # Originally initialized with -inf for AUC.
            # We need to adapt the initialization logic or invert the check.
            pass

    # Fix initialization for classification if switching to Brier
    if task_type == 'classification' and best_score == float('-inf'):
         best_score = float('inf')

    for key in candidates:
        if key not in models_dict:
            continue

        res = models_dict[key]
        pred = res['cate']

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
            # Classification: Brier Score (lower is better)
            try:
                y_bin = (y_true > 0).astype(int)
                score = brier_score_loss(y_bin, y_pred)
            except ValueError:
                score = 1.0

            if score < best_score:
                best_score = score
                best_model_key = key
                best_pred = pred

    if best_model_key:
        metric_name = "IC" if task_type == 'regression' else "AUC"
        tprint_info(f"   🏆 Best model for {horizon} {task_type}: {best_model_key} ({metric_name}: {best_score:.4f})")
        tprint_success("End: select_best_model_per_task")
        return best_pred, best_model_key
    else:
        tprint_warning(f"   ⚠️ No valid models found for {horizon} {task_type}")
        tprint_success("End: select_best_model_per_task")
        return np.zeros(len(y_target)), "none"

def prepare_layer3_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    symbol: str,
    exchange: str,
    config: Optional[Dict[str, Any]] = None,
    market_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Phase 1: Meta-Features Engineering.
    Wrapper for `generate_layer3_features` with caching and robust error handling.
    """
    tprint_info("Start: prepare_layer3_features")
    cfg = config or {}
    safe_base_cols = [c for c in base_model_cols if c in df.columns]

    asset_col = cfg.get("layer3_asset_id_col") or cfg.get("asset_id_col")
    asset_series = None
    if asset_col and asset_col in df.columns:
        asset_series = df[asset_col]
    if asset_series is None:
        for candidate in ("asset_id", "asset", "ticker", "symbol"):
            if candidate in df.columns:
                asset_col = candidate
                asset_series = df[candidate]
                break

    if asset_series is None and isinstance(df.index, pd.MultiIndex):
        for candidate in ("asset_id", "asset", "ticker", "symbol"):
            if candidate in df.index.names:
                asset_col = candidate
                asset_series = df.index.get_level_values(candidate)
                df = df.copy()
                df[candidate] = asset_series
                break

    multi_asset = bool(asset_series is not None and asset_series.nunique(dropna=False) > 1)
    if multi_asset:
        tprint_info(f"   🧩 Multi-asset Layer3 feature prep detected ({asset_col})")

    # Try loading features from cache
    features_loaded = False
    if not multi_asset and CACHE_AVAILABLE and should_use_cached_features(cfg, symbol, exchange, cfg.get('timeframe', '15m'), 'long'):
        tprint_info("📦 Checking cache for Layer 3 features...")
        cached_features, _ = load_layer3_features_from_cache(
            symbol=symbol,
            exchange=exchange,
            timeframe=cfg.get('timeframe', '15m'),
            direction='long',
            target_index=df.index,
            market_data=market_data,
            validate_hash=True
        )

        if cached_features is not None:
            # Merge cached features into df
            # Identify new columns
            new_cols = [c for c in cached_features.columns if c not in df.columns]
            if new_cols:
                df = pd.concat([df, cached_features[new_cols]], axis=1)
                tprint_success(f"✅ Loaded {len(new_cols)} features from cache")
                features_loaded = True

    if not features_loaded:
        try:
            from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
            if multi_asset and asset_col:
                grouped = df.groupby(asset_col, group_keys=False)
                df = grouped.apply(lambda g: generate_layer3_features(g.sort_index(), safe_base_cols))
            else:
                df = generate_layer3_features(df, safe_base_cols)

            # Save to cache if enabled
            if not multi_asset and CACHE_AVAILABLE and cfg.get('use_layer3_feature_cache', True):
                # Identify generated features (exclude base columns)
                exclude_cols = set(df.columns) - set(safe_base_cols) - {'close', 'high', 'low', 'open', 'volume'}
                generated_cols = [c for c in df.columns if c not in exclude_cols] # Logic error in previous implementation?
                # Correct logic: generated cols are newly added ones.
                # Actually, let's just save whatever generate_layer3_features returned minus original columns

                # Re-identify generated cols safely
                # (Assuming df grew)
                # For simplicity, we pass the subset to cache
                # The cache function handles column selection
                save_layer3_features_to_cache(
                    meta_features=df,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=cfg.get('timeframe', '15m'),
                    direction='long',
                    market_data=market_data,
                    config=cfg
                )
        except Exception as e:
            tprint_warning(f"⚠️ Feature generation failed: {e}")

    tprint_success("End: prepare_layer3_features")

    if multi_asset and asset_col and cfg.get("layer3_asset_onehot", True):
        max_assets = cfg.get("layer3_asset_onehot_max")
        asset_count = asset_series.nunique(dropna=False)
        if max_assets and asset_count > max_assets:
            tprint_warning(
                f"   ⚠️ Skipping Layer3 asset one-hot (assets={asset_count} > max={max_assets})"
            )
        else:
            asset_dummies = pd.get_dummies(df[asset_col].astype(str), prefix="asset")
            asset_dummies = asset_dummies.astype(np.float32)
            df = pd.concat([df, asset_dummies], axis=1)

    return df

def prepare_layer3_targets_and_weights(
    df: pd.DataFrame,
    layer1_weight: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Phase 2: Targets + Volatility Weights.
    Calculates targets (12/48 horizons) and sample weights.
    """
    tprint_info("Start: prepare_layer3_targets_and_weights")
    cfg = config or {}

    asset_col = cfg.get("layer3_asset_id_col") or cfg.get("asset_id_col")
    asset_series = None
    if asset_col and asset_col in df.columns:
        asset_series = df[asset_col]
    if asset_series is None:
        for candidate in ("asset_id", "asset", "ticker", "symbol"):
            if candidate in df.columns:
                asset_col = candidate
                asset_series = df[candidate]
                break

    if asset_series is None and isinstance(df.index, pd.MultiIndex):
        for candidate in ("asset_id", "asset", "ticker", "symbol"):
            if candidate in df.index.names:
                asset_col = candidate
                asset_series = df.index.get_level_values(candidate)
                df = df.copy()
                df[candidate] = asset_series
                break

    multi_asset = bool(asset_series is not None and asset_series.nunique(dropna=False) > 1)

    if net_returns is None:
        if 'close' in df.columns:
            if multi_asset and asset_series is not None:
                ret_series = df.groupby(asset_series)['close'].pct_change().fillna(0)
            else:
                ret_series = df['close'].pct_change().fillna(0)
        else:
            ret_series = pd.Series(0, index=df.index)
    else:
        ret_series = net_returns.reindex(df.index) if isinstance(net_returns, pd.Series) else pd.Series(net_returns, index=df.index)

    if multi_asset and asset_series is not None:
        vol_series = (
            ret_series.groupby(asset_series)
            .rolling(24)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.001)
        )
    else:
        vol_series = ret_series.rolling(24).std().fillna(0.001)

    # 12-bar targets
    if 'close' in df.columns:
        if multi_asset and asset_series is not None:
            blended_ret_series = (
                df.groupby(asset_series)['close']
                .apply(lambda s: calculate_blended_forward_returns(s, [16, 24]))
                .reset_index(level=0, drop=True)
            )
        else:
            blended_ret_series = calculate_blended_forward_returns(df['close'], [16, 24])
    else:
        blended_ret_series = ret_series

    y_alpha_12_series = calculate_studentized_har_target(blended_ret_series, vol_series)
    y_alpha_12 = y_alpha_12_series.values.astype(np.float32)
    y_prob_12 = (blended_ret_series.values > 0).astype(np.int32)
    
    # 48-bar targets
    if 'close' in df.columns:
        if multi_asset and asset_series is not None:
            ret_48 = (
                df.groupby(asset_series)['close']
                .apply(lambda s: s.shift(-48) / s - 1)
                .reset_index(level=0, drop=True)
            )
            vol_48 = (
                ret_series.groupby(asset_series)
                .rolling(48)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.001)
            )
        else:
            ret_48 = df['close'].shift(-48) / df['close'] - 1
            vol_48 = ret_series.rolling(48).std().fillna(0.001)

        y_alpha_48_series = calculate_studentized_har_target(ret_48.fillna(0), vol_48.fillna(0))
        y_alpha_48 = y_alpha_48_series.values.astype(np.float32)
        y_prob_48 = (ret_48.fillna(0) > 0).astype(np.int32)
    else:
        y_alpha_48 = y_alpha_12 * 1.5
        y_prob_48 = y_prob_12

    # Weights
    vol_values = df['volume'].values.astype(np.float32) if 'volume' in df.columns else None

    w_alpha = calculate_sample_weights_efficient(
        ret_series.values,
        vol_series.values,
        layer1_weights=layer1_weight.values if layer1_weight is not None else None,
        volume=vol_values
    )
    w_alpha = w_alpha.astype(np.float32)

    tprint_success("End: prepare_layer3_targets_and_weights")
    return {
        'y_alpha_12': y_alpha_12,
        'y_prob_12': y_prob_12,
        'y_alpha_12_series': y_alpha_12_series,
        'y_alpha_48': y_alpha_48,
        'y_prob_48': y_prob_48,
        'w_alpha': w_alpha
    }

def process_layer3_results(
    df: pd.DataFrame,
    combined_models: Dict[str, Any],
    best_models_info: Dict[str, str],
    X_index: pd.Index
) -> pd.DataFrame:
    """
    Phase 11: Save best models OOF predictions.
    Maps predictions back to DataFrame.
    """
    tprint_info("Start: process_layer3_results")

    def propagate_simple(values, idx):
        return pd.Series(values, index=idx).reindex(df.index).fillna(0)

    # Save ALL models OOF predictions
    for key, res in combined_models.items():
        if 'cate' in res:
            pred = res['cate']
            df[f"{key}_oof"] = propagate_simple(pred, X_index)

    # Map best models to meta columns
    # Using the keys from best_models_info to find the right prediction

    # 12-bar
    key_12_reg = best_models_info.get('12_reg')
    if key_12_reg and key_12_reg in combined_models:
        df['meta_alpha'] = propagate_simple(combined_models[key_12_reg]['cate'], X_index)

    key_12_cls = best_models_info.get('12_cls')
    if key_12_cls and key_12_cls in combined_models:
        df['meta_prob'] = propagate_simple(combined_models[key_12_cls]['cate'], X_index)

    # 48-bar
    key_48_reg = best_models_info.get('48_reg')
    if key_48_reg and key_48_reg in combined_models:
        df['meta_alpha_48'] = propagate_simple(combined_models[key_48_reg]['cate'], X_index)

    key_48_cls = best_models_info.get('48_cls')
    if key_48_cls and key_48_cls in combined_models:
        df['meta_prob_48'] = propagate_simple(combined_models[key_48_cls]['cate'], X_index)

    # Legacy compatibility
    if 'meta_alpha' in df.columns:
        df['orf_cate'] = df['meta_alpha']
    if 'meta_prob' in df.columns:
        df['orf_se'] = df['meta_prob'] * 0.1 # Placeholder

    tprint_success("End: process_layer3_results")
    return df

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
    tprint_info("Start: layer3_analyst_lgbm")
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
    df = prepare_layer3_features(
        df=df,
        base_model_cols=base_model_cols,
        symbol=symbol,
        exchange=exchange,
        config=cfg,
        market_data=market_data
    )

    # Phase 2: Targets + Volatility Weights
    targets_data = prepare_layer3_targets_and_weights(
        df=df,
        layer1_weight=layer1_weight,
        net_returns=net_returns,
        config=cfg
    )

    y_alpha_12 = targets_data['y_alpha_12']
    y_prob_12 = targets_data['y_prob_12']
    y_alpha_48 = targets_data['y_alpha_48']
    y_prob_48 = targets_data['y_prob_48']
    w_alpha = targets_data['w_alpha']
    y_alpha_12_series = targets_data['y_alpha_12_series']

    cfg['y_alpha_48'] = y_alpha_48
    cfg['y_prob_48'] = y_prob_48

    # Phase 3: Mild MP-Clustering (Feature Selection)
    safe_base_cols = [c for c in base_model_cols if c in df.columns]
    exclude = set(base_model_cols) | {target_col, 'close', 'high', 'low', 'volume', 'regime_label'}
    meta_features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
    X_full = df[meta_features].copy()
    
    # Add base model columns back for training (important for stacking)
    for col in safe_base_cols:
        if col in df.columns:
            X_full[col] = df[col].reindex(X_full.index)

    # Phase 3.5: Regime Aware Features
    prob_cols = [c for c in X_full.columns if 'prob_' in c and '_oof' not in c]
    asset_col = cfg.get("layer3_asset_id_col") or cfg.get("asset_id_col")
    asset_series = None
    if asset_col and asset_col in df.columns:
        asset_series = df[asset_col]
    if asset_series is None:
        for candidate in ("asset_id", "asset", "ticker", "symbol"):
            if candidate in df.columns:
                asset_col = candidate
                asset_series = df[candidate]
                break
    regime_feats = generate_regime_aware_features(
        X_full,
        'volatility_20',
        prob_cols,
        asset_series=asset_series if asset_series is not None and asset_series.nunique(dropna=False) > 1 else None
    )
    X_full = pd.concat([X_full, regime_feats], axis=1)

    # Phase 3.75: Layer 2.5 Chaser Integration (if available)
    layer25_enabled = cfg.get('layer25_chaser_enabled', True)
    layer25_results = cfg.get('layer25_chaser_results', None)
    
    if layer25_enabled and layer25_results is not None:
        tprint_info("🔗 Phase 3.75: Integrating Layer 2.5 Chaser Models...")
        try:
            # Integration parameters
            symbol = cfg.get('symbol', 'ETHUSDT')
            exchange = cfg.get('exchange', 'binance')
            timeframe = cfg.get('timeframe', '15m')
            top_n_models = cfg.get('layer25_top_models', 3)
            
            # Convert X_full back to DataFrame for integration
            df_for_integration = df.copy()
            for col in X_full.columns:
                if col not in df_for_integration.columns:
                    df_for_integration[col] = X_full[col]
            
            # Integrate Layer 2.5 models
            df_enhanced, integration_metadata = integrate_layer25_into_layer3(
                df=df_for_integration,
                chaser_results=layer25_results,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                top_n_models=top_n_models,
                outcomes_dir=outcomes_dir
            )
            
            # Update X_full with new chaser features
            chaser_features = [col for col in df_enhanced.columns if col.startswith('chaser_')]
            for feature in chaser_features:
                if feature not in X_full.columns:
                    X_full[feature] = df_enhanced[feature]
                    df[feature] = df_enhanced[feature]
            
            # Store integration metadata
            cfg['layer25_integration'] = integration_metadata
            
            tprint_success(f"✅ Integrated {len(chaser_features)} Layer 2.5 chaser features")
            
        except Exception as e:
            tprint_warning(f"⚠️ Layer 2.5 integration failed: {e}")
            cfg['layer25_integration'] = {'status': 'failed', 'error': str(e)}
    else:
        tprint_info("⏭️ Skipping Layer 2.5 Chaser integration (disabled or no results)")

    # Apply Clustering (Optimized)
    # Use 12-bar target for feature selection relevance
    X_clustered = apply_mild_mp_clustering(X_full, threshold=0.98, target=y_alpha_12_series)

    irm_env_indices = []
    if cfg.get("irm_meta_enabled", True):
        gmm_dir = Path(cfg.get("irm_regime_dir", "artifacts/irm_regimes"))
        gmm_dir.mkdir(parents=True, exist_ok=True)
        regime_labels = get_or_fit_regime_labels(
            df,
            gmm_dir / f"layer3_meta_gmm_{symbol}.pkl",
            n_regimes=cfg.get("irm_meta_regimes", 2),
            refit=cfg.get("irm_refit_regimes", False)
        )
        irm_env_indices = build_env_indices_for_index(regime_labels, X_clustered.index)

    # Phase 4-8: Multi-Horizon Model Training
    # (Huber Teacher -> Rotation -> Train -> Optuna -> Predictions) handled in model_training.py
    tprint_info("🤖 Phase 4-8: Multi-Horizon Model Training")
    
    # Run training for all horizons/tasks
    cfg["irm_env_indices"] = irm_env_indices
    cfg["irm_lambda"] = cfg.get("irm_meta_lambda", 2.0)
    model_results = train_dual_head_models(
        X_clustered, y_alpha_12, y_prob_12, w_alpha, w_alpha, [], cfg, cfg.get('fast_mode', False)
    )
    
    combined_models = model_results['models']

    # Phase 9: Select best models
    tprint_info("🏆 Phase 9: Select Best Models")

    best_pred_12_reg, best_key_12_reg = select_best_model_per_task(combined_models, y_alpha_12, 'regression', '12')
    best_pred_12_cls, best_key_12_cls = select_best_model_per_task(combined_models, y_prob_12, 'classification', '12')
    best_pred_48_reg, best_key_48_reg = select_best_model_per_task(combined_models, y_alpha_48, 'regression', '48')
    best_pred_48_cls, best_key_48_cls = select_best_model_per_task(combined_models, y_prob_48, 'classification', '48')

    # Store Best Model Keys in results
    best_models_info = {
        '12_reg': best_key_12_reg,
        '12_cls': best_key_12_cls,
        '48_reg': best_key_48_reg,
        '48_cls': best_key_48_cls
    }

    # Phase 11: Save best models OOF predictions
    tprint_info("💾 Phase 11: Save Models OOF")

    df = process_layer3_results(
        df=df,
        combined_models=combined_models,
        best_models_info=best_models_info,
        X_index=X_clustered.index
    )

    models_dict = {
        'all_models': combined_models,
        'best_models': best_models_info,
        'meta_features': meta_features,
        'entropy_bars': entropy_bars_df if not entropy_bars_df.empty else None,
        'irm_env_indices': irm_env_indices
    }

    # Enhanced Reporting
    try:
        reporter = EnhancedLayer3Reporter(outcomes_dir=outcomes_dir)
        reporter.generate_all_reports(
            df=df,
            models={'models': combined_models}, # Wrap in dict to match expected structure if needed
            geometry_metrics=cfg.get('geometry_metrics', []),
            meta_features=meta_features,
            target_col='meta_prob',
            config=cfg
        )
    except Exception as e:
        tprint_warning(f"⚠️ Enhanced Layer 3 reporting failed: {e}")

    # Model Race Reporting
    try:
        race_reporter = Layer3ModelRaceReporter(outcomes_dir=outcomes_dir)
        race_reporter.generate_model_race_report(
            models_dict=combined_models,
            y_alpha_12=y_alpha_12,
            y_prob_12=y_prob_12,
            y_alpha_48=y_alpha_48,
            y_prob_48=y_prob_48
        )
    except Exception as e:
        tprint_warning(f"⚠️ Layer 3 model race reporting failed: {e}")

    tprint_success(f"🎉 Layer 3 Pipeline Complete! Best Models: {best_models_info}")
    
    tprint_success("End: layer3_analyst_lgbm")
    return df, models_dict
