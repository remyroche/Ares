"""
Shared feature preparation utilities for NAS-TAS regime detection.

This module provides common feature engineering functionality that eliminates
redundancy between NAS and TAS components.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import time
import psutil
from src.utils.tprint import tprint, tprint_debug, tprint_success, tprint_warning, tprint_error

from src.config.regime_feature_thresholds import get_regime_feature_thresholds
from .feature_filters import (
    apply_quality_thresholds,
    filter_low_variance,
    prune_correlated_features,
    winsorize_frame,
)


@dataclass
class FeatureConfig:
    """Configuration for feature preparation."""
    # Feature categories to include
    feature_categories: List[str] = None
    
    # Lookback periods for various features
    returns_lookback: int = 1
    volatility_lookback: int = 20
    ma_short_lookback: int = 10
    ma_long_lookback: int = 30
    volume_lookback: int = 10
    momentum_lookback: int = 14
    entropy_lookback: int = 20
    
    # Feature processing options
    use_standardized_features: bool = True
    drop_highly_correlated: bool = True
    correlation_threshold: Optional[float] = None

    # Feature quality thresholds
    min_variance: Optional[float] = None
    winsorize_lower_quantile: Optional[float] = None
    winsorize_upper_quantile: Optional[float] = None
    min_persistence: Optional[float] = None
    max_noise_ratio: Optional[float] = None
    min_stability: Optional[float] = None
    min_feature_importance: Optional[float] = None
    
    # Data validation
    handle_missing_values: bool = True
    min_observations: int = 100
    
    def __post_init__(self):
        thresholds = get_regime_feature_thresholds()
        filter_thresholds = thresholds.get('filter_thresholds', {})
        quality_thresholds = thresholds.get('quality_thresholds', {})
        quality_filter_cfg = filter_thresholds.get('quality', {})

        if self.feature_categories is None:
            # Use regime-focused feature categories for regime classification
            self.feature_categories = [
                'regime_volatility',       # Volatility regime features (8 features)
                'regime_volume',           # Volume regime features
                'regime_structural_trend', # Structural trend features (6 features)
                'regime_statistical'       # Statistical regime features (11 features)
            ]

        if self.correlation_threshold is None:
            self.correlation_threshold = (
                filter_thresholds.get('correlation', {}).get('threshold', 0.95)
            )

        if self.min_variance is None:
            self.min_variance = filter_thresholds.get('variance', {}).get('min_variance', 1.0e-8)

        if self.winsorize_lower_quantile is None:
            self.winsorize_lower_quantile = (
                filter_thresholds.get('winsorization', {}).get('lower_quantile', 0.01)
            )

        if self.winsorize_upper_quantile is None:
            self.winsorize_upper_quantile = (
                filter_thresholds.get('winsorization', {}).get('upper_quantile', 0.99)
            )

        if self.min_persistence is None:
            self.min_persistence = quality_filter_cfg.get(
                'min_persistence', quality_thresholds.get('min_regime_persistence', 0.2)
            )

        if self.max_noise_ratio is None:
            self.max_noise_ratio = quality_filter_cfg.get(
                'max_noise_ratio', quality_thresholds.get('max_feature_noise_ratio', 1.2)
            )

        if self.min_stability is None:
            self.min_stability = quality_filter_cfg.get(
                'min_stability', quality_thresholds.get('min_temporal_stability', 0.1)
            )


@dataclass
class FeaturePreparationResult:
    """Container object for Stage 1 feature preparation outputs."""

    features_array: np.ndarray
    features_df: pd.DataFrame
    summary: Dict[str, Any]
    metadata: Dict[str, Any]

    def __array__(self) -> np.ndarray:  # pragma: no cover - helper for numpy interop
        return self.features_array


def prepare_market_features(
    market_data: pd.DataFrame,
    feature_config: Optional[FeatureConfig] = None,
    verbose: bool = False,
    return_metadata: bool = False,
) -> Optional[Union[np.ndarray, FeaturePreparationResult]]:

    """
    Prepare comprehensive market features for regime detection and clustering.
    
    This function uses the feature generation system to create comprehensive
    features from all specified categories, providing much richer feature sets
    than the basic implementation.
    
    Args:
        market_data: Market data DataFrame with OHLCV columns
        feature_config: Configuration for feature preparation
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple containing the processed feature DataFrame and associated metadata.
        
    Raises:
        ValueError: If market data is invalid or insufficient
    """
    if feature_config is None:
        feature_config = FeatureConfig()
    
    feature_prep_start = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    if verbose:
        tprint("🔧 [SHARED_FEATURES] ===== COMPREHENSIVE FEATURE PREPARATION =====", color="blue", bold=True)
        tprint_debug(f"📊 [SHARED_FEATURES] Market data shape: {market_data.shape}")
        tprint_debug(f"📊 [SHARED_FEATURES] Available columns: {list(market_data.columns)}")
        tprint_debug(f"📊 [SHARED_FEATURES] Feature categories: {feature_config.feature_categories}")
    
    try:
        # Validate input data
        if market_data is None or market_data.empty:
            if verbose:
                tprint_error("❌ [SHARED_FEATURES] Market data is None or empty")
            raise ValueError("Market data is None or empty")
        
        if len(market_data) < feature_config.min_observations:
            if verbose:
                tprint_error(f"❌ [SHARED_FEATURES] Insufficient data: {len(market_data)} < {feature_config.min_observations}")
            raise ValueError(f"Insufficient data: {len(market_data)} < {feature_config.min_observations}")
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        if missing_columns:
            if verbose:
                tprint_error(f"❌ [SHARED_FEATURES] Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if verbose:
            tprint_debug(f"✅ [SHARED_FEATURES] Data validation passed")
        
        # Use balanced feature extractor for comprehensive features
        if verbose:
            tprint_debug("🚀 [SHARED_FEATURES] Using balanced feature extractor for comprehensive features")
        
        try:
            # Use regime-focused feature generation instead of balanced extractor
            from src.feature_generation.categories.regime_feature_integration import (
                generate_regime_features, RegimeFeatureConfig
            )
            
            # Create regime-focused configuration
            # Check if regime categories are in the feature config, default to True if not specified
            has_regime_categories = any('regime_' in cat for cat in feature_config.feature_categories)
            
            regime_config = RegimeFeatureConfig(
                include_volatility_regime='regime_volatility' in feature_config.feature_categories if has_regime_categories else True,
                include_volume_regime='regime_volume' in feature_config.feature_categories if has_regime_categories else True,
                include_structural_trend='regime_structural_trend' in feature_config.feature_categories if has_regime_categories else True,
                include_statistical_regime='regime_statistical' in feature_config.feature_categories if has_regime_categories else True,
                min_regime_persistence=0.1,     # Further relaxed to get more features
                max_feature_noise_ratio=2.0,   # Much more relaxed to allow more features
                min_temporal_stability=0.05,   # Much more relaxed to allow more features
                optimize_for_15m=True,
                trade_duration_minutes=(5, 30),
                enable_feature_selection=False,  # Disable feature selection to get more features
                max_features_per_category=50,    # Increased from 30 to 50
                total_max_features=110          # Target 90-110 features for optimal regime detection
            )
            
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Extracting regime-focused features for categories: {feature_config.feature_categories}")
                tprint("🎯 [REGIME_FEATURES] Using RegimeFeatureIntegration targeting 90-110 regime-specific features", color="cyan", bold=True)
            
            # Generate regime-focused features
            features_dict, summary = generate_regime_features(
                data=market_data,
                config=regime_config
            )

            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Generated {len(features_dict)} regime features")
                if features_dict:
                    tprint_debug(f"📊 [SHARED_FEATURES] Feature keys: {list(features_dict.keys())}")
                    # Check for NaN/inf values in features
                    nan_counts = {k: np.sum(np.isnan(v)) for k, v in features_dict.items()}
                    inf_counts = {k: np.sum(np.isinf(v)) for k, v in features_dict.items()}
                    tprint_debug(f"📊 [SHARED_FEATURES] NaN counts: {nan_counts}")
                    tprint_debug(f"📊 [SHARED_FEATURES] Inf counts: {inf_counts}")

            if not features_dict or len(features_dict) == 0:
                error_msg = "❌ [SHARED_FEATURES] No regime features generated - trying fallback features"
                if verbose:
                    tprint_warning(error_msg)
                # Try fallback to basic features
                try:
                    fallback_features = _generate_basic_features(market_data, feature_config, verbose)
                    if fallback_features is not None and not fallback_features.empty:
                        features_dict = {f"fallback_{col}": fallback_features[col].values for col in fallback_features.columns}
                        if verbose:
                            tprint_warning(f"✅ [SHARED_FEATURES] Using fallback features: {len(features_dict)} features")
                    else:
                        error_msg = "❌ [SHARED_FEATURES] No fallback features available either"
                        if verbose:
                            tprint_error(error_msg)
                        raise ValueError(error_msg)
                except Exception as fallback_error:
                    error_msg = f"❌ [SHARED_FEATURES] Fallback feature generation also failed: {fallback_error}"
                    if verbose:
                        tprint_error(error_msg)
                    raise ValueError(error_msg)
            else:
                # Convert regime features to DataFrame
                if verbose:
                    tprint_success(f"✅ [REGIME_FEATURES] Generated {len(features_dict)} regime features")
                    tprint(f"📊 [REGIME_FEATURES] Category quotas:", color="blue")
                    for cat, info in summary['selection']['category_quota'].items():
                        tprint(
                            f"   - {cat}: {info['count']}/{info['max']} features",
                            color="cyan"
                        )
                    tprint(f"📊 [REGIME_FEATURES] Quality metrics:", color="blue")
                    tprint(
                        f"   - Avg persistence: {summary['quality_metrics']['avg_persistence']:.3f}",
                        color="cyan"
                    )
                    tprint(
                        f"   - Avg noise ratio: {summary['quality_metrics']['avg_noise_ratio']:.3f}",
                        color="cyan"
                    )
                    tprint(
                        f"   - Avg stability: {summary['quality_metrics']['avg_temporal_stability']:.3f}",
                        color="cyan"
                    )
                    weights = summary['selection'].get('weights', {})
                    if weights:
                        tprint(
                            f"📐 [REGIME_FEATURES] Composite weights — "
                            f"persistence: {weights.get('persistence', 0.0):.2f}, "
                            f"noise penalty: {weights.get('noise_penalty', 0.0):.2f}, "
                            f"stability: {weights.get('stability', 0.0):.2f}",
                            color="blue"
                        )
                    composite_scores = summary['selection'].get('composite_scores', {})
                    if composite_scores:
                        avg_score = float(np.mean(list(composite_scores.values()))) if composite_scores else 0.0
                        tprint(
                            f"📈 [REGIME_FEATURES] Avg composite score: {avg_score:.4f}",
                            color="blue"
                        )
                        top_ranked = summary['selection'].get('top_ranked_features', [])
                        if top_ranked:
                            tprint(f"🏆 [REGIME_FEATURES] Top ranked features:", color="blue")
                            for feature_name, score in top_ranked[:5]:
                                tprint(f"   • {feature_name}: {score:.4f}", color="cyan")

                # Ensure all feature arrays have the same length before stacking
                feature_lengths = [len(arr) for arr in features_dict.values()]
                min_length = min(feature_lengths)
                max_length = max(feature_lengths)

                if min_length != max_length:
                    if verbose:
                        tprint_warning(f"⚠️ [SHARED_FEATURES] Feature arrays have different lengths (min: {min_length}, max: {max_length}), truncating to minimum length")
                    # Truncate all arrays to the minimum length
                    features_dict = {
                        name: arr[:min_length] for name, arr in features_dict.items()
                    }

                # Convert features dict to DataFrame
                features_array = np.column_stack(list(features_dict.values()))
                features_df = pd.DataFrame(
                    features_array,
                    columns=list(features_dict.keys()),
                    index=market_data.index[:len(features_array)]
                )

            stage_metadata: Dict[str, Any] = {
                'original_feature_count': int(features_df.shape[1]),
                'original_row_count': int(features_df.shape[0]),
                'operations': [],
            }
            
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Generated {len(features_df.columns)} features")
                tprint_debug(f"📊 [SHARED_FEATURES] Feature columns: {list(features_df.columns)}")
            
        except Exception as e:
            error_msg = f"❌ [SHARED_FEATURES] Regime feature generation failed: {e} - fast failing as requested"
            if verbose:
                tprint_error(error_msg)
            raise ValueError(error_msg) from e
        
        metadata: Dict[str, Any] = {
            'columns': {col: {} for col in features_df.columns},
            'filters': {},
            'dropped_columns': {},
        }

        # Winsorize extreme values prior to downstream filtering
        if (
            feature_config.winsorize_lower_quantile is not None
            and feature_config.winsorize_upper_quantile is not None
        ):
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Applying winsorization")
            features_df, winsor_meta = winsorize_frame(
                features_df,
                feature_config.winsorize_lower_quantile,
                feature_config.winsorize_upper_quantile,
            )
            metadata['filters']['winsorization'] = winsor_meta
            for col, info in winsor_meta.items():
                metadata['columns'].setdefault(col, {}).update({'winsorization': info})

        # Handle missing and infinite values
        if verbose:
            tprint_debug("🔧 [SHARED_FEATURES] Handling missing and infinite values")

        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        rows_before = len(features_df)

        if feature_config.handle_missing_values:
            # Enhanced NaN analysis and handling
            initial_nan_count = np.isnan(features_array).sum()
            initial_nan_pct = initial_nan_count / (features_array.size) * 100
            
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Enhanced NaN handling: {initial_nan_count:,} NaN values ({initial_nan_pct:.2f}%)")
            
            # Enhanced NaN handling with regime-aware imputation
            features_df = _enhanced_nan_handling(features_df, verbose=verbose)
            features_array = features_df.values

            # Remove rows with any remaining NaN values
            valid_rows = ~np.isnan(features_array).any(axis=1)
            valid_indices = np.where(valid_rows)[0]
            features_array = features_array[valid_rows]
            features_df = features_df.iloc[valid_indices]
            removed_rows = int(len(valid_rows) - valid_rows.sum())

            # Log enhanced NaN handling results
            final_nan_count = np.isnan(features_array).sum()
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Enhanced NaN handling: {initial_nan_count:,} → {final_nan_count:,} remaining")
                if removed_rows > 0:
                    tprint_debug(f"📊 [SHARED_FEATURES] Removed {removed_rows} rows with NaN values")

            stage_metadata['operations'].append({
                'type': 'nan_row_filter',
                'removed_rows': removed_rows,
                'initial_nan_count': int(initial_nan_count),
                'final_nan_count': int(final_nan_count),
            })

        rows_after = len(features_df)
        if rows_after == 0:
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Removed {removed_rows} rows with NaN values")
            raise ValueError("No data remaining after NaN filtering")

        # Apply variance filtering
        variance_result = filter_low_variance(features_df, feature_config.min_variance)
        if verbose:
            tprint_debug(f"📊 [SHARED_FEATURES] Dropped low-variance features: {variance_result.dropped_columns}")
        features_df = variance_result.frame

        if features_df.empty:
            raise ValueError("No features remain after variance filtering")

        # Quality thresholds (persistence/noise/stability)
        quality_filtered_df, quality_metrics, quality_dropped = apply_quality_thresholds(
            features_df,
            feature_config.min_persistence,
            feature_config.max_noise_ratio,
            feature_config.min_stability,
        )
        metadata['filters']['quality'] = {
            'metrics': quality_metrics,
            'dropped': quality_dropped,
        }
        for col, info in quality_metrics.items():
            metadata['columns'].setdefault(col, {}).update(info)
        if quality_dropped:
            metadata['dropped_columns']['quality'] = list(quality_dropped.keys())
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Removing highly correlated features")
            
            # Ensure arrays are aligned before correlation analysis
            if features_array.shape[1] != features_df.shape[1]:
                if verbose:
                    tprint_warning(f"⚠️ [SHARED_FEATURES] Array/DataFrame column mismatch: array={features_array.shape[1]}, df={features_df.shape[1]}")
                # Use the minimum number of columns
                min_cols = min(features_array.shape[1], features_df.shape[1])
                features_array = features_array[:, :min_cols]
                features_df = features_df.iloc[:, :min_cols]
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features_array.T)
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > 0.95:  # High correlation threshold
                        high_corr_pairs.append((i, j))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for i, j in high_corr_pairs:
                if i not in features_to_remove:
                    features_to_remove.add(j)
            
            # Remove highly correlated features
            if features_to_remove:
                keep_indices = [i for i in range(features_array.shape[1]) if i not in features_to_remove]
                # Ensure indices are within bounds for both arrays
                max_cols_array = features_array.shape[1]
                max_cols_df = features_df.shape[1]
                keep_indices = [i for i in keep_indices if i < max_cols_array and i < max_cols_df]
                
                if keep_indices:
                    features_array = features_array[:, keep_indices]
                    features_df = features_df.iloc[:, keep_indices]
                else:
                    # If no features remain, keep the first feature (ensure it exists)
                    if max_cols_array > 0 and max_cols_df > 0:
                        features_array = features_array[:, [0]]
                        features_df = features_df.iloc[:, [0]]
                    else:
                        raise ValueError("No features available after correlation filtering")
                stage_metadata['operations'].append({
                    'type': 'correlation_filter',
                    'removed_features': int(len(features_to_remove)),
                    'threshold': feature_config.correlation_threshold,
                })


                if verbose:
                    tprint_debug(f"📊 [SHARED_FEATURES] Dropped correlated features: {len(features_to_remove)} features")

        if features_df.empty:
            raise ValueError("No features remain after correlation pruning")

        # Final NaN guard
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

        if len(features_df) < feature_config.min_observations:
            if verbose:
                tprint_error(
                    f"❌ [SHARED_FEATURES] Insufficient valid observations: {len(features_df)} < {feature_config.min_observations}"
                )
            raise ValueError(
                f"Insufficient valid observations: {len(features_df)} < {feature_config.min_observations}"
            )

        # Scale features if requested (using MinMaxScaler for better regime separation)
        if feature_config.use_standardized_features:
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Scaling features with MinMaxScaler")

            from sklearn.preprocessing import MinMaxScaler
            
            # Scale features
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features_df)
            features_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
            
            if verbose:
                tprint_debug("✅ [SHARED_FEATURES] Features scaled successfully")
        
        # Create result
        result = FeaturePreparationResult(
            features=features_df,
            feature_names=features_df.columns.tolist(),
            metadata=stage_metadata,
            success=True
        )
        
        if verbose:
            tprint_success(f"✅ [SHARED_FEATURES] Feature preparation completed: {len(features_df.columns)} features, {len(features_df)} observations")
        
        return result
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ [SHARED_FEATURES] Feature preparation failed: {e}")
        raise

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def _generate_basic_features(market_data: pd.DataFrame, feature_config: FeatureConfig, verbose: bool = False) -> pd.DataFrame:
    """
    Generate basic features as fallback when comprehensive feature generation fails.
    
    Args:
        market_data: Market data DataFrame
        feature_config: Feature configuration
        verbose: Whether to enable verbose logging
        
    Returns:
        DataFrame with basic features
    """
    if verbose:
        tprint_debug("🔧 [SHARED_FEATURES] Generating basic features as fallback")
    
    features_dict = {}
    
    # Always generate basic features regardless of categories
    if verbose:
        tprint_debug("📈 [SHARED_FEATURES] Calculating basic returns")
    features_dict['returns'] = market_data['close'].pct_change(feature_config.returns_lookback)
    
    if verbose:
        tprint_debug("📊 [SHARED_FEATURES] Calculating basic volatility")
    returns = market_data['close'].pct_change()
    features_dict['volatility'] = returns.rolling(window=feature_config.volatility_lookback).std()
    
    if verbose:
        tprint_debug("📈 [SHARED_FEATURES] Calculating basic moving averages")
    ma_short = market_data['close'].rolling(window=feature_config.ma_short_lookback).mean()
    ma_long = market_data['close'].rolling(window=feature_config.ma_long_lookback).mean()
    features_dict['ma_ratio'] = ma_short / ma_long
    features_dict['ma_spread'] = (ma_short - ma_long) / ma_long
    
    if verbose:
        tprint_debug("📊 [SHARED_FEATURES] Calculating basic volume features")
    volume_ma = market_data['volume'].rolling(window=feature_config.volume_lookback).mean()
    features_dict['volume_ratio'] = market_data['volume'] / volume_ma
    features_dict['volume_change'] = market_data['volume'].pct_change()
    
    if verbose:
        tprint_debug("📊 [SHARED_FEATURES] Calculating basic price action features")
    features_dict['hl_spread'] = (market_data['high'] - market_data['low']) / market_data['close']
    features_dict['oc_spread'] = (market_data['close'] - market_data['open']) / market_data['open']
    features_dict['body_size'] = abs(market_data['close'] - market_data['open']) / (market_data['high'] - market_data['low'])
    
    if verbose:
        tprint_debug("🚀 [SHARED_FEATURES] Calculating basic momentum indicators")
    # RSI-like momentum
    returns = market_data['close'].pct_change()
    positive_returns = returns.where(returns > 0, 0).rolling(window=feature_config.momentum_lookback).mean()
    negative_returns = (-returns.where(returns < 0, 0)).rolling(window=feature_config.momentum_lookback).mean()
    features_dict['momentum'] = positive_returns / (positive_returns + negative_returns)
    features_dict['price_momentum'] = market_data['close'].pct_change(feature_config.momentum_lookback)
    
    if verbose:
        tprint_debug("🔍 [SHARED_FEATURES] Calculating basic entropy features")
    # Simple entropy measure based on price changes
    returns = market_data['close'].pct_change()
    returns_abs = abs(returns)
    features_dict['entropy'] = returns_abs.rolling(window=feature_config.entropy_lookback).std()
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_dict, index=market_data.index)
    
    if verbose:
        tprint_debug(f"📊 [SHARED_FEATURES] Basic features DataFrame shape: {features_df.shape}")
        tprint_debug(f"📊 [SHARED_FEATURES] Feature columns: {list(features_df.columns)}")
    
    return features_df


def detect_data_leakage(features: np.ndarray, targets: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Detect potential data leakage by checking for suspiciously high correlations
    between features and target variables.

    Args:
        features: Feature matrix (n_samples, n_features)
        targets: Target variable array (n_samples,)
        threshold: Correlation threshold above which to flag as suspicious

    Returns:
        Dictionary with leakage detection results
    """
    results = {
        'has_leakage': False,
        'suspicious_features': [],
        'max_correlation': 0.0,
        'correlation_matrix': None
    }

    try:
        if features.shape[0] != len(targets):
            raise ValueError(f"Features and targets must have same number of samples. Got {features.shape[0]} vs {len(targets)}")

        # Calculate correlation between each feature and target
        correlations = []
        suspicious_indices = []

        for i in range(features.shape[1]):
            feature_col = features[:, i]
            # Handle NaN values
            valid_mask = ~(np.isnan(feature_col) | np.isnan(targets))
            if np.sum(valid_mask) < 10:  # Need minimum samples for correlation
                continue

            corr = np.corrcoef(feature_col[valid_mask], targets[valid_mask])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
                if abs(corr) > threshold:
                    suspicious_indices.append(i)

        if correlations:
            results['max_correlation'] = max(correlations)
            results['suspicious_features'] = suspicious_indices
            results['has_leakage'] = results['max_correlation'] > threshold

        return results

    except Exception as e:
        print(f"Error in data leakage detection: {e}")
        return results


def validate_features(features: np.ndarray, expected_shape: Optional[tuple] = None) -> bool:
    """
    Validate feature array quality.
    
    Args:
        features: Feature array to validate
        expected_shape: Expected shape (n_samples, n_features)
        
    Returns:
        True if features are valid, False otherwise
    """
    try:
        # Check if features is None or empty
        if features is None or features.size == 0:
            return False
        
        # Check shape if provided
        if expected_shape is not None:
            if features.shape != expected_shape:
                return False
        
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            return False
        
        # Check for constant features (all same value)
        if np.all(features == features[0]):
            return False
        
        return True
        
    except Exception:
        return False


def _enhanced_nan_handling(features_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Enhanced NaN handling with regime-aware imputation strategies.
    
    Args:
        features_df: DataFrame with potential NaN values
        verbose: Whether to enable verbose logging
        
    Returns:
        DataFrame with enhanced NaN handling
    """
    if verbose:
        tprint("🔧 [SHARED_FEATURES] Applying enhanced NaN handling...")
    
    # Strategy 1: Forward fill for time series continuity
    features_df = features_df.fillna(method='ffill')
    
    # Strategy 2: Backward fill for remaining NaNs
    features_df = features_df.fillna(method='bfill')
    
    # Strategy 3: Regime-aware imputation for remaining NaNs
    for col in features_df.columns:
        if features_df[col].isna().any():
            # Use median for robust imputation
            median_val = features_df[col].median()
            if not pd.isna(median_val):
                features_df[col] = features_df[col].fillna(median_val)
            else:
                # Fallback to 0 for completely missing columns
                features_df[col] = features_df[col].fillna(0.0)
    
    if verbose:
        remaining_nans = features_df.isnull().sum().sum()
        tprint(f"✅ [SHARED_FEATURES] Enhanced NaN handling completed: {remaining_nans} NaNs remaining")
    
    return features_df


def _analyze_nan_patterns(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze NaN patterns in features for better handling strategies.
    
    Args:
        features_df: DataFrame to analyze
        
    Returns:
        Dictionary with NaN analysis results
    """
    total_cells = features_df.size
    total_nans = features_df.isnull().sum().sum()
    nan_percentage = (total_nans / total_cells) * 100
    
    # Analyze by feature
    feature_nan_counts = features_df.isnull().sum()
    features_with_nans = feature_nan_counts[feature_nan_counts > 0]
    
    # Analyze by row
    row_nan_counts = features_df.isnull().sum(axis=1)
    rows_with_nans = row_nan_counts[row_nan_counts > 0]
    
    return {
        'total_nans': int(total_nans),
        'nan_percentage': float(nan_percentage),
        'features_with_nans': len(features_with_nans),
        'rows_with_nans': len(rows_with_nans),
        'max_nans_per_feature': int(feature_nan_counts.max()) if len(feature_nan_counts) > 0 else 0,
        'max_nans_per_row': int(row_nan_counts.max()) if len(row_nan_counts) > 0 else 0
    }

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
