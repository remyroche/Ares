"""
HMM ML Integration Utilities

This module provides utilities for integrating HMM performance metrics with ML models,
including feature preparation, model training, and ensemble weighting.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Import HMM components
try:
    # enhanced_hmm_clustering module no longer exists - replaced by component system
    # from ...training.steps.market_analysis.hmm_clustering.enhanced_hmm_clustering import (
    #     EnhancedHMMClustering, HMMClusteringResult, run_hmm_clustering_analysis
    # )
    from ..categories.hmm_performance_metrics import (

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
        HMMPerformanceMetricsFeatureGenerator,
        create_hmm_performance_features_from_result,
        integrate_hmm_metrics_with_features
    )
    HMM_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HMM integration components not available: {e}")
    HMM_INTEGRATION_AVAILABLE = False


class HMMMLIntegrator:
    """
    Integrates HMM performance metrics with ML training pipelines.
    
    This class provides a complete pipeline for:
    1. Running HMM analysis
    2. Extracting performance metrics
    3. Converting metrics to ML features
    4. Integrating with existing feature pipelines
    5. Supporting ensemble model weighting
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize HMM ML Integrator.
        
        Args:
            cache_dir: Directory to cache HMM results for reuse
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.hmm_results_cache = {}
        self.feature_generator = None
        
        if HMM_INTEGRATION_AVAILABLE:
            self.feature_generator = HMMPerformanceMetricsFeatureGenerator()
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def run_hmm_analysis_with_caching(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        hmm_config: Optional[Any] = None,
        force_recompute: bool = False
    ) -> Optional[Any]:
        """
        Run HMM analysis with caching support.
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            interval: Time interval
            hmm_config: HMM configuration
            force_recompute: Force recomputation even if cached
            
        Returns:
            HMMClusteringResult or None if failed
        """
        if not HMM_INTEGRATION_AVAILABLE:
            logger.error("HMM integration not available")
            return None
        
        cache_key = f"{symbol}_{interval}_{hash(str(hmm_config))}"
        
        # Check cache first
        if not force_recompute and cache_key in self.hmm_results_cache:
            logger.info(f"Using cached HMM result for {symbol} {interval}")
            return self.hmm_results_cache[cache_key]
        
        # Check file cache
        if self.cache_dir and not force_recompute:
            cache_file = self.cache_dir / f"hmm_result_{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    logger.info(f"Loaded HMM result from file cache for {symbol} {interval}")
                    # Note: This is simplified - in practice you'd need to reconstruct the full result object
                    return cached_data
                except Exception as e:
                    logger.warning(f"Failed to load cached HMM result: {e}")
        
        # Run HMM analysis
        try:
            logger.info(f"Running HMM analysis for {symbol} {interval}")
            result = run_hmm_clustering_analysis(
                symbol=symbol,
                interval=interval,
                config=hmm_config,
                save_results=False  # We handle caching ourselves
            )
            
            if result is not None:
                # Cache in memory
                self.hmm_results_cache[cache_key] = result
                
                # Cache to file
                if self.cache_dir:
                    cache_file = self.cache_dir / f"hmm_result_{cache_key}.json"
                    try:
                        cache_data = {
                            'performance_metrics': result.performance_metrics,
                            'regime_labels': result.regime_labels.tolist(),
                            'regime_probabilities': result.regime_probabilities.tolist(),
                            'regime_characteristics': result.regime_characteristics,
                            'feature_importance': result.feature_importance,
                            'processing_time': result.processing_time,
                            'symbol': symbol,
                            'interval': interval,
                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                        with open(cache_file, 'w') as f:
                            json.dump(cache_data, f, indent=2)
                        logger.info(f"Cached HMM result to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to cache HMM result: {e}")
                
                return result
            else:
                logger.error(f"HMM analysis failed for {symbol} {interval}")
                return None
                
        except Exception as e:
            logger.error(f"Error running HMM analysis for {symbol} {interval}: {e}")
            return None
    
    def create_ml_features_from_hmm(
        self,
        data: pd.DataFrame,
        hmm_result: Any,
        lookback_window: int = 20,
        include_regime_features: bool = True,
        include_rolling_features: bool = True
    ) -> pd.DataFrame:
        """
        Create ML features from HMM results.
        
        Args:
            data: Market data DataFrame
            hmm_result: HMM clustering result
            lookback_window: Window size for rolling features
            include_regime_features: Include regime-based features
            include_rolling_features: Include rolling metrics features
            
        Returns:
            DataFrame with ML-ready features
        """
        if not HMM_INTEGRATION_AVAILABLE or self.feature_generator is None:
            logger.error("HMM feature generation not available")
            return pd.DataFrame(index=data.index)
        
        try:
            # Generate comprehensive HMM features
            hmm_features = self.feature_generator.generate_features(
                data,
                hmm_performance_metrics=hmm_result.performance_metrics,
                regime_labels=hmm_result.regime_labels if include_regime_features else None,
                regime_probabilities=hmm_result.regime_probabilities if include_regime_features else None
            )
            
            # Filter features based on options
            if not include_rolling_features:
                rolling_cols = [col for col in hmm_features.columns if 'rolling' in col]
                hmm_features = hmm_features.drop(columns=rolling_cols)
            
            logger.info(f"Generated {len(hmm_features.columns)} HMM features")
            return hmm_features
            
        except Exception as e:
            logger.error(f"Failed to create ML features from HMM result: {e}")
            return pd.DataFrame(index=data.index)
    
    def integrate_with_existing_features(
        self,
        base_features: pd.DataFrame,
        hmm_features: pd.DataFrame,
        feature_selection_method: str = 'correlation',
        max_correlation: float = 0.95,
        feature_importance_threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Integrate HMM features with existing feature set.
        
        Args:
            base_features: Existing features DataFrame
            hmm_features: HMM-derived features
            feature_selection_method: Method for feature selection
            max_correlation: Maximum correlation threshold for feature filtering
            feature_importance_threshold: Minimum importance threshold
            
        Returns:
            Tuple of (integrated_features, integration_info)
        """
        try:
            # Combine features
            combined_features = pd.concat([base_features, hmm_features], axis=1)
            
            # Remove duplicate columns
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            
            integration_info = {
                'original_features': len(base_features.columns),
                'hmm_features': len(hmm_features.columns),
                'combined_features': len(combined_features.columns),
                'feature_selection_method': feature_selection_method
            }
            
            # Feature selection based on method
            if feature_selection_method == 'correlation':
                # Remove highly correlated features
                correlation_matrix = combined_features.corr().abs()
                upper_triangle = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                
                high_corr_features = [
                    column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > max_correlation)
                ]
                
                # Prioritize HMM features over base features when removing correlations
                features_to_remove = []
                for feature in high_corr_features:
                    if not feature.startswith('hmm_'):
                        features_to_remove.append(feature)
                
                if features_to_remove:
                    combined_features = combined_features.drop(columns=features_to_remove)
                    integration_info['removed_correlated_features'] = len(features_to_remove)
            
            integration_info['final_features'] = len(combined_features.columns)
            
            logger.info(f"Integrated features: {integration_info}")
            return combined_features, integration_info
            
        except Exception as e:
            logger.error(f"Failed to integrate features: {e}")
            return base_features, {'error': str(e)}
    
    def create_ensemble_weights_from_hmm(
        self,
        hmm_results: List[Any],
        weighting_method: str = 'performance_based'
    ) -> np.ndarray:
        """
        Create ensemble weights based on HMM performance metrics.
        
        Args:
            hmm_results: List of HMM clustering results
            weighting_method: Method for calculating weights
            
        Returns:
            Array of normalized weights for ensemble models
        """
        if not hmm_results:
            return np.array([])
        
        try:
            weights = []
            
            for result in hmm_results:
                if weighting_method == 'performance_based':
                    # Weight based on multiple performance metrics
                    stability = result.performance_metrics.get('regime_stability', 0.5)
                    balance = result.performance_metrics.get('regime_balance', 0.5)
                    confidence = result.performance_metrics.get('avg_confidence', 0.5)
                    separation = result.performance_metrics.get('regime_separation_ratio', 0.5)
                    
                    # Composite score
                    weight = np.mean([stability, balance, confidence, separation])
                    weights.append(weight)
                
                elif weighting_method == 'confidence_based':
                    # Weight based on average confidence
                    confidence = result.performance_metrics.get('avg_confidence', 0.5)
                    weights.append(confidence)
                
                elif weighting_method == 'stability_based':
                    # Weight based on regime stability
                    stability = result.performance_metrics.get('regime_stability', 0.5)
                    weights.append(stability)
                
                else:
                    # Equal weighting
                    weights.append(1.0)
            
            # Normalize weights
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            logger.info(f"Created ensemble weights using {weighting_method}: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to create ensemble weights: {e}")
            return np.ones(len(hmm_results)) / len(hmm_results)
    
    def prepare_features_for_ml_training(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        base_feature_generator: Optional[Callable] = None,
        hmm_config: Optional[Any] = None,
        feature_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete pipeline for preparing features including HMM metrics for ML training.
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            interval: Time interval
            base_feature_generator: Function to generate base features
            hmm_config: HMM configuration
            feature_config: Feature generation configuration
            
        Returns:
            Tuple of (features_dataframe, metadata)
        """
        feature_config = feature_config or {}
        metadata = {
            'symbol': symbol,
            'interval': interval,
            'data_shape': data.shape,
            'processing_steps': []
        }
        
        try:
            # Step 1: Generate base features
            if base_feature_generator:
                logger.info("Generating base features")
                base_features = base_feature_generator(data)
                metadata['processing_steps'].append('base_features_generated')
                metadata['base_features_count'] = len(base_features.columns)
            else:
                # Create minimal base features
                base_features = pd.DataFrame({
                    'returns': data['close'].pct_change(),
                    'log_returns': np.log(data['close'] / data['close'].shift(1)),
                    'volatility': data['close'].pct_change().rolling(20).std()
                }, index=data.index)
                metadata['processing_steps'].append('minimal_base_features_created')
                metadata['base_features_count'] = len(base_features.columns)
            
            # Step 2: Run HMM analysis
            logger.info("Running HMM analysis")
            hmm_result = self.run_hmm_analysis_with_caching(
                data, symbol, interval, hmm_config
            )
            
            if hmm_result is not None:
                metadata['processing_steps'].append('hmm_analysis_completed')
                metadata['hmm_metrics_count'] = len(hmm_result.performance_metrics)
                
                # Step 3: Generate HMM features
                logger.info("Generating HMM features")
                hmm_features = self.create_ml_features_from_hmm(
                    data, hmm_result,
                    lookback_window=feature_config.get('lookback_window', 20),
                    include_regime_features=feature_config.get('include_regime_features', True),
                    include_rolling_features=feature_config.get('include_rolling_features', True)
                )
                metadata['processing_steps'].append('hmm_features_generated')
                metadata['hmm_features_count'] = len(hmm_features.columns)
                
                # Step 4: Integrate features
                logger.info("Integrating features")
                final_features, integration_info = self.integrate_with_existing_features(
                    base_features, hmm_features,
                    feature_selection_method=feature_config.get('feature_selection_method', 'correlation'),
                    max_correlation=feature_config.get('max_correlation', 0.95)
                )
                metadata['processing_steps'].append('features_integrated')
                metadata['integration_info'] = integration_info
                
            else:
                logger.warning("HMM analysis failed, using only base features")
                final_features = base_features
                metadata['processing_steps'].append('hmm_analysis_failed')
            
            # Step 5: Final preprocessing
            final_features = final_features.dropna()
            metadata['processing_steps'].append('final_preprocessing_completed')
            metadata['final_features_shape'] = final_features.shape
            
            logger.info(f"Feature preparation completed: {final_features.shape}")
            return final_features, metadata
            
        except Exception as e:
            logger.error(f"Failed to prepare features for ML training: {e}")
            metadata['error'] = str(e)
            return pd.DataFrame(index=data.index), metadata


# Convenience functions for easy integration

def quick_hmm_features_integration(
    data: pd.DataFrame,
    symbol: str,
    interval: str = "1h",
    hmm_config: Optional[Any] = None
) -> pd.DataFrame:
    """
    Quick integration of HMM features for immediate use.
    
    Args:
        data: Market data DataFrame
        symbol: Trading symbol
        interval: Time interval
        hmm_config: HMM configuration
        
    Returns:
        DataFrame with HMM features ready for ML
    """
    integrator = HMMMLIntegrator()
    features, _ = integrator.prepare_features_for_ml_training(
        data, symbol, interval, hmm_config=hmm_config
    )
    return features


def create_hmm_ensemble_pipeline(
    data_dict: Dict[str, pd.DataFrame],
    symbols: List[str],
    interval: str = "1h",
    hmm_configs: Optional[List[Any]] = None
) -> Tuple[Dict[str, pd.DataFrame], np.ndarray]:
    """
    Create ensemble pipeline with HMM-based weighting.
    
    Args:
        data_dict: Dictionary of {symbol: data} pairs
        symbols: List of symbols to analyze
        interval: Time interval
        hmm_configs: List of HMM configurations (one per symbol)
        
    Returns:
        Tuple of (features_dict, ensemble_weights)
    """
    integrator = HMMMLIntegrator()
    features_dict = {}
    hmm_results = []
    
    hmm_configs = hmm_configs or [None] * len(symbols)
    
    for symbol, config in zip(symbols, hmm_configs):
        if symbol in data_dict:
            features, metadata = integrator.prepare_features_for_ml_training(
                data_dict[symbol], symbol, interval, hmm_config=config
            )
            features_dict[symbol] = features
            
            # Get HMM result for ensemble weighting
            hmm_result = integrator.run_hmm_analysis_with_caching(
                data_dict[symbol], symbol, interval, config
            )
            if hmm_result:
                hmm_results.append(hmm_result)
    
    # Create ensemble weights
    ensemble_weights = integrator.create_ensemble_weights_from_hmm(
        hmm_results, weighting_method='performance_based'
    )
    
    return features_dict, ensemble_weights
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
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
