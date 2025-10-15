"""
import warnings
Temporal Feature Integration Module

This module provides hierarchical temporal analysis by combining:
1. Feature Lookback Optimization - Multiple periods per indicator on same timeframe
2. Cross Timeframe Analysis - Multiple timeframes with different granularities
3. Temporal Feature Integration - Intelligent combination and deduplication

The solution creates a hierarchical structure where:
- Lookback optimization feeds into cross-timeframe analysis
- Redundant features are intelligently removed
- Complementary temporal information is preserved
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Core imports
from src.utils.logger import system_logger
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt

# Feature optimization imports
try:
    from src.feature_generation.utils.optimization_config import OptimizationConfigManager
    from src.feature_generation.utils.optimization_validator import validate_optimization_results
    from src.feature_generation.utils.optimization_metrics import generate_optimization_report
    FEATURE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    FEATURE_OPTIMIZATION_AVAILABLE = False

# Cross timeframe analysis imports
try:
    from src.feature_generation.utils.optimized_cross_timeframe_analysis_integration import (

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

except ImportError:
    
    cp = None
        analyze_cross_timeframes_optimized, create_optimized_config
    )
    CROSS_TIMEFRAME_AVAILABLE = True
except ImportError:
    CROSS_TIMEFRAME_AVAILABLE = False

logger = system_logger.getChild('TemporalFeatureIntegration')

@dataclass
class TemporalFeatureConfig:
    """Configuration for temporal feature integration."""
    # Lookback optimization settings
    enable_lookback_optimization: bool = True
    lookback_optimization_config: Optional[Dict] = None
    
    # Cross timeframe analysis settings
    enable_cross_timeframe_analysis: bool = True
    cross_timeframe_config: Optional[Dict] = None
    
    # Integration settings
    correlation_threshold: float = 0.7  # Remove features with correlation > threshold
    information_threshold: float = 0.1  # Minimum information content to keep
    stability_threshold: float = 0.3    # Minimum stability score to keep
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0

@dataclass
class TemporalFeatureResult:
    """Result of temporal feature integration."""
    # Lookback optimization results
    lookback_results: Optional[Dict] = None
    lookback_features: Optional[Dict] = None
    
    # Cross timeframe analysis results
    cross_timeframe_results: Optional[Dict] = None
    cross_timeframe_features: Optional[Dict] = None
    
    # Integrated results
    integrated_features: Dict = field(default_factory=dict)
    deduplicated_features: Dict = field(default_factory=dict)
    feature_metadata: Dict = field(default_factory=dict)
    
    # Performance metrics
    integration_time: float = 0.0
    total_features_before: int = 0
    total_features_after: int = 0
    redundancy_removed: int = 0
    
    # Quality metrics
    average_correlation: float = 0.0
    average_information_content: float = 0.0
    average_stability: float = 0.0

class TemporalFeatureIntegration:
    """
    Hierarchical temporal feature integration system.
    
    This class combines feature lookback optimization and cross timeframe analysis
    to create a comprehensive temporal feature set without redundancy.
    """
    
    def __init__(self, config: Optional[TemporalFeatureConfig] = None):
        """Initialize temporal feature integration."""
        self.config = config or TemporalFeatureConfig()
        self.logger = logger.getChild('TemporalFeatureIntegration')
        
        # Initialize sub-systems
        self.lookback_optimizer = None
        self.cross_timeframe_analyzer = None
        
        self._initialize_subsystems()
    
    def _initialize_subsystems(self):
        """Initialize sub-systems for lookback optimization and cross timeframe analysis."""
        try:
            if FEATURE_OPTIMIZATION_AVAILABLE and self.config.enable_lookback_optimization:
                self.lookback_optimizer = OptimizationConfigManager()
                self.logger.info("✅ Lookback optimization system initialized")
            
            if CROSS_TIMEFRAME_AVAILABLE and self.config.enable_cross_timeframe_analysis:
                self.cross_timeframe_analyzer = create_optimized_config(
                    timeframes=['1m', '5m', '15m', '30m'],
                    enable_m1_optimizations=True,
                    enable_gpu_acceleration=True,
                    enable_advanced_feature_selection=True,
                    memory_limit_gb=self.config.memory_limit_gb,
                    max_workers=self.config.max_workers
                )
                self.logger.info("✅ Cross timeframe analysis system initialized")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Sub-system initialization warning: {e}")
    
    async def integrate_temporal_features(
        self, 
        data: pd.DataFrame,
        data_dir: Optional[str] = None,
        symbol: str = "BTCUSDT",
        exchange: str = "BINANCE"
    ) -> TemporalFeatureResult:
        """
        Integrate temporal features from both lookback optimization and cross timeframe analysis.
        
        Args:
            data: Input data DataFrame
            data_dir: Directory containing data files
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            TemporalFeatureResult with integrated features
        """
        start_time = time.time()
        result = TemporalFeatureResult()
        
        try:
            self.logger.info("🚀 Starting hierarchical temporal feature integration")
            
            # Step 1: Run lookback optimization
            if self.config.enable_lookback_optimization and self.lookback_optimizer:
                result.lookback_results, result.lookback_features = await self._run_lookback_optimization(data)
            
            # Step 2: Run cross timeframe analysis
            if self.config.enable_cross_timeframe_analysis and self.cross_timeframe_analyzer:
                result.cross_timeframe_results, result.cross_timeframe_features = await self._run_cross_timeframe_analysis(
                    data, data_dir, symbol, exchange
                )
            
            # Step 3: Integrate features intelligently
            result.integrated_features = await self._merge_temporal_features(
                result.lookback_features, 
                result.cross_timeframe_features
            )
            
            # Step 4: Remove redundant features
            result.deduplicated_features = await self._remove_temporal_redundancy(
                result.integrated_features
            )
            
            # Step 5: Generate metadata and metrics
            result.feature_metadata = await self._generate_feature_metadata(result.deduplicated_features)
            result = await self._calculate_quality_metrics(result)
            
            result.integration_time = time.time() - start_time
            result.total_features_before = len(result.integrated_features)
            result.total_features_after = len(result.deduplicated_features)
            result.redundancy_removed = result.total_features_before - result.total_features_after
            
            self.logger.info(f"✅ Temporal feature integration completed in {result.integration_time:.2f}s")
            self.logger.info(f"📊 Features: {result.total_features_before} → {result.total_features_after} (removed {result.redundancy_removed})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Temporal feature integration failed: {e}")
            result.integration_time = time.time() - start_time
            return result
    
    async def _run_lookback_optimization(self, data: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Run feature lookback optimization."""
        try:
            self.logger.info("🔍 Running lookback optimization...")
            
            # This would integrate with the actual lookback optimization pipeline
            # For now, we'll simulate the structure
            lookback_results = {
                'optimal_lookbacks': {
                    'rsi': 14, 'sma': 20, 'ema': 12, 'macd': 9,
                    'bollinger_bands': 20, 'stochastic': 21, 'atr': 14
                },
                'optimization_metrics': {
                    'method': 'enhanced_statistical_optimization',
                    'total_features_optimized': 7,
                    'average_performance_score': 0.72
                }
            }
            
            # Generate features with optimal lookbacks
            lookback_features = await self._generate_lookback_features(data, lookback_results['optimal_lookbacks'])
            
            return lookback_results, lookback_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Lookback optimization failed: {e}")
            return None, None
    
    async def _run_cross_timeframe_analysis(
        self, 
        data: pd.DataFrame, 
        data_dir: Optional[str], 
        symbol: str, 
        exchange: str
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Run cross timeframe analysis."""
        try:
            self.logger.info("⏰ Running cross timeframe analysis...")
            
            if not CROSS_TIMEFRAME_AVAILABLE:
                self.logger.warning("⚠️ Cross timeframe analysis not available")
                return None, None
            
            # Run cross timeframe analysis
            cross_timeframe_results = await analyze_cross_timeframes_optimized(
                data_dir=data_dir or "historical_data",
                symbol=symbol,
                exchange=exchange,
                config=self.cross_timeframe_analyzer
            )
            
            # Extract features from results
            cross_timeframe_features = cross_timeframe_results.get('features', {})
            
            return cross_timeframe_results, cross_timeframe_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cross timeframe analysis failed: {e}")
            return None, None
    
    async def _generate_lookback_features(self, data: pd.DataFrame, optimal_lookbacks: Dict[str, int]) -> Dict[str, pd.Series]:
        """Generate features using optimal lookback periods."""
        features = {}
        
        try:
            # RSI features with optimal period
            if 'rsi' in optimal_lookbacks:
                period = optimal_lookbacks['rsi']
                rsi = self._calculate_rsi(data['close'], period)
                features[f'rsi_{period}'] = rsi
                features[f'rsi_{period}_signal'] = (rsi > 70).astype(int) - (rsi < 30).astype(int)
            
            # SMA features with optimal period
            if 'sma' in optimal_lookbacks:
                period = optimal_lookbacks['sma']
                sma = data['close'].rolling(period).mean()
                features[f'sma_{period}'] = sma
                features[f'sma_{period}_ratio'] = data['close'] / sma
            
            # EMA features with optimal period
            if 'ema' in optimal_lookbacks:
                period = optimal_lookbacks['ema']
                ema = data['close'].ewm(span=period).mean()
                features[f'ema_{period}'] = ema
                features[f'ema_{period}_momentum'] = data['close'] - ema
            
            # MACD features with optimal period
            if 'macd' in optimal_lookbacks:
                period = optimal_lookbacks['macd']
                macd_line, signal_line, histogram = self._calculate_macd(data['close'], period, period*2, period//2)
                features[f'macd_{period}'] = macd_line
                features[f'macd_signal_{period}'] = signal_line
                features[f'macd_histogram_{period}'] = histogram
            
            # Bollinger Bands with optimal period
            if 'bollinger_bands' in optimal_lookbacks:
                period = optimal_lookbacks['bollinger_bands']
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'], period)
                features[f'bb_upper_{period}'] = bb_upper
                features[f'bb_middle_{period}'] = bb_middle
                features[f'bb_lower_{period}'] = bb_lower
                features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                features[f'bb_position_{period}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic with optimal period
            if 'stochastic' in optimal_lookbacks:
                period = optimal_lookbacks['stochastic']
                stoch_k, stoch_d = self._calculate_stochastic(data, period)
                features[f'stoch_k_{period}'] = stoch_k
                features[f'stoch_d_{period}'] = stoch_d
            
            # ATR with optimal period
            if 'atr' in optimal_lookbacks:
                period = optimal_lookbacks['atr']
                atr = self._calculate_atr(data, period)
                features[f'atr_{period}'] = atr
                features[f'atr_ratio_{period}'] = atr / data['close']
            
            self.logger.info(f"✅ Generated {len(features)} lookback features")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate lookback features: {e}")
            return {}
    
    async def _merge_temporal_features(
        self, 
        lookback_features: Optional[Dict[str, pd.Series]], 
        cross_timeframe_features: Optional[Dict[str, pd.Series]]
    ) -> Dict[str, pd.Series]:
        """Merge features from both temporal analysis approaches."""
        integrated_features = {}
        
        # Add lookback optimization features
        if lookback_features:
            for name, series in lookback_features.items():
                integrated_features[f'lookback_{name}'] = series
        
        # Add cross timeframe features
        if cross_timeframe_features:
            for name, series in cross_timeframe_features.items():
                integrated_features[f'cross_tf_{name}'] = series
        
        self.logger.info(f"✅ Merged {len(integrated_features)} temporal features")
        return integrated_features
    
    async def _remove_temporal_redundancy(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Remove redundant features based on correlation and information content."""
        if not features:
            return {}
        
        try:
            # Convert to DataFrame for correlation analysis
            feature_df = pd.DataFrame(features)
            feature_df = feature_df.dropna()
            
            if len(feature_df) < 10:  # Need minimum data for correlation
                self.logger.warning("⚠️ Insufficient data for redundancy removal")
                return features
            
            # Calculate correlation matrix
            corr_matrix = feature_df.corr().abs()
            
            # Find highly correlated pairs
            redundant_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.config.correlation_threshold:
                        redundant_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            # Remove redundant features (keep the one with higher information content)
            features_to_remove = set()
            for feat1, feat2, correlation in redundant_pairs:
                if feat1 in features_to_remove or feat2 in features_to_remove:
                    continue
                
                # Calculate information content (variance as proxy)
                info1 = features[feat1].var()
                info2 = features[feat2].var()
                
                # Remove the feature with lower information content
                if info1 > info2:
                    features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat1)
            
            # Create deduplicated features
            deduplicated_features = {
                name: series for name, series in features.items() 
                if name not in features_to_remove
            }
            
            self.logger.info(f"✅ Removed {len(features_to_remove)} redundant features")
            self.logger.info(f"📊 Features: {len(features)} → {len(deduplicated_features)}")
            
            return deduplicated_features
            
        except Exception as e:
            self.logger.error(f"❌ Redundancy removal failed: {e}")
            return features
    
    async def _generate_feature_metadata(self, features: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """Generate metadata for features."""
        metadata = {}
        
        for name, series in features.items():
            try:
                metadata[name] = {
                    'type': 'lookback' if name.startswith('lookback_') else 'cross_timeframe',
                    'length': len(series),
                    'non_null_count': series.count(),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'variance': series.var(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis()
                }
            except Exception as e:
                self.logger.debug(f"⚠️ Failed to generate metadata for {name}: {e}")
                metadata[name] = {'error': str(e)}
        
        return metadata
    
    async def _calculate_quality_metrics(self, result: TemporalFeatureResult) -> TemporalFeatureResult:
        """Calculate quality metrics for the integrated features."""
        try:
            if not result.deduplicated_features:
                return result
            
            # Convert to DataFrame for analysis
            feature_df = pd.DataFrame(result.deduplicated_features)
            feature_df = feature_df.dropna()
            
            if len(feature_df) < 2:
                return result
            
            # Calculate average correlation
            corr_matrix = feature_df.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            result.average_correlation = upper_triangle.stack().mean()
            
            # Calculate average information content (variance)
            variances = [series.var() for series in result.deduplicated_features.values()]
            result.average_information_content = np.mean(variances)
            
            # Calculate average stability (inverse of coefficient of variation)
            stabilities = []
            for series in result.deduplicated_features.values():
                if series.std() > 0:
                    cv = series.std() / abs(series.mean())
                    stability = 1 / (1 + cv)
                    stabilities.append(stability)
            result.average_stability = np.mean(stabilities) if stabilities else 0.0
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Quality metrics calculation failed: {e}")
            return result
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self._vectorbt_rolling_operation(prices, "mean", period)
        std = self._vectorbt_rolling_operation(prices, "std", period)
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = data['low'].rolling(window=period).min()
        high_max = data['high'].rolling(window=period).max()
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_percent = self._vectorbt_rolling_operation(k_percent, "mean", 3)
        return k_percent, d_percent
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = self._vectorbt_rolling_operation(true_range, "mean", period)
        return atr

# Convenience functions
async def integrate_temporal_features(
    data: pd.DataFrame,
    config: Optional[TemporalFeatureConfig] = None,
    data_dir: Optional[str] = None,
    symbol: str = "BTCUSDT",
    exchange: str = "BINANCE"
) -> TemporalFeatureResult:
    """
    Convenience function for temporal feature integration.
    
    Args:
        data: Input data DataFrame
        config: Configuration for integration
        data_dir: Directory containing data files
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        TemporalFeatureResult with integrated features
    """
    integrator = TemporalFeatureIntegration(config)
    return await integrator.integrate_temporal_features(data, data_dir, symbol, exchange)

def create_temporal_config(
    enable_lookback: bool = True,
    enable_cross_timeframe: bool = True,
    correlation_threshold: float = 0.7,
    parallel_processing: bool = True
) -> TemporalFeatureConfig:
    """
    Create a temporal feature configuration.
    
    Args:
        enable_lookback: Enable lookback optimization
        enable_cross_timeframe: Enable cross timeframe analysis
        correlation_threshold: Correlation threshold for redundancy removal
        parallel_processing: Enable parallel processing
        
    Returns:
        TemporalFeatureConfig instance
    """
    return TemporalFeatureConfig(
        enable_lookback_optimization=enable_lookback,
        enable_cross_timeframe_analysis=enable_cross_timeframe,
        correlation_threshold=correlation_threshold,
        parallel_processing=parallel_processing
    )
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
