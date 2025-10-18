"""
Enhanced VectorBT Integration for UnifiedDataDrivenPipeline

This module provides advanced VectorBT optimizations integrated from individual components
including DataDrivenPeriodSelector, DataDrivenInteractionGenerator, and FeatureLookbackOptimizationComponent.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None

# Import UnifiedVectorizationManager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OperationConfig
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OperationConfig = None
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

# Additional imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)

@dataclass
class VectorBTOptimizationConfig:
    """Configuration for VectorBT optimizations."""
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_efficient: bool = True
    batch_size: int = 1000
    max_workers: int = 4
    chunk_size: int = 10000
    use_cupy: bool = False  # GPU support removed
    optimization_level: str = "high"  # "low", "medium", "high", "maximum"

@dataclass
class VectorBTOptimizationResult:
    """Result from VectorBT feature optimization."""
    optimized_features: pd.DataFrame
    optimization_metrics: Dict[str, Any]
    performance_stats: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class EnhancedVectorBTOptimizer:
    """
    Enhanced VectorBT optimizer with advanced optimizations from individual components.

    Integrates optimizations from:
    - DataDrivenPeriodSelector: Period analysis and economic evaluation
    - DataDrivenInteractionGenerator: Interaction generation and feature selection
    - FeatureLookbackOptimizationComponent: Lookback optimization and matrix operations
    """

    def __init__(self, config: Optional[VectorBTOptimizationConfig] = None):
        """Initialize the enhanced VectorBT optimizer."""
        self.config = config or VectorBTOptimizationConfig()
        self.logger = logger

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'fallback_operations': 0,
            'gpu_operations': 0,
            'total_execution_time': 0.0,
            'memory_usage': 0.0
        }

        # Initialize UnifiedVectorizationManager if available
        self.vectorization_manager = None
        if UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_info("✅ UnifiedVectorizationManager initialized for VectorBT optimization")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize UnifiedVectorizationManager: {e}")
                self.vectorization_manager = None

        tprint_info("🚀 Enhanced VectorBT Optimizer initialized")
        tprint_debug(f"📊 Configuration: {self.config}")

    def optimize_period_analysis(self, data: pd.DataFrame, periods: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Optimize period analysis using VectorBT with advanced algorithms.

        Args:
            data: Input data with OHLCV columns
            periods: List of periods to analyze

        Returns:
            Dictionary mapping periods to analysis results
        """
        tprint_info(f"🔍 Optimizing period analysis for {len(periods)} periods")

        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._fallback_period_analysis(data, periods)

        try:
            start_time = time.time()
            results = {}

            # VectorBT-optimized period analysis
            for period in periods:
                period_result = self._analyze_period_vectorbt(data, period)
                results[period] = period_result

                # Update performance stats
                self.performance_stats['vectorbt_operations'] += 1

            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['total_operations'] += len(periods)

            tprint_success(f"✅ Period analysis completed in {execution_time:.3f}s")
            return results

        except Exception as e:
            tprint_error(f"❌ VectorBT period analysis failed: {e}")
            return self._fallback_period_analysis(data, periods)

    def _analyze_period_vectorbt(self, data: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Analyze a single period using VectorBT optimization."""
        try:
            if 'close' not in data.columns:
                return {'error': 'No close price data available'}

            close_prices = data['close']

            # VectorBT-optimized calculations
            sma = rolling_mean(close_prices, window=period)
            ema = close_prices.ewm(span=period).mean()
            volatility = rolling_std(close_prices, window=period)
            returns = close_prices.pct_change()

            # Advanced statistical measures
            sharpe_ratio = self._calculate_sharpe_ratio_vectorbt(returns, period)
            max_drawdown = self._calculate_max_drawdown_vectorbt(close_prices, period)
            win_rate = self._calculate_win_rate_vectorbt(returns, period)

            # Volatility clustering
            vol_clustering = self._calculate_volatility_clustering_vectorbt(returns, period)

            # Regime detection
            regime_stability = self._calculate_regime_stability_vectorbt(close_prices, period)

            return {
                'period': period,
                'sma': sma,
                'ema': ema,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility_clustering': vol_clustering,
                'regime_stability': regime_stability,
                'data_quality': self._assess_data_quality_vectorbt(close_prices, period)
            }

        except Exception as e:
            self.logger.error(f"VectorBT period analysis failed for period {period}: {e}")
            return {'error': str(e), 'period': period}

    def _calculate_sharpe_ratio_vectorbt(self, returns: pd.Series, period: int) -> float:
        """Calculate Sharpe ratio using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_sharpe_ratio(returns, period)

            # VectorBT-optimized Sharpe ratio calculation
            rolling_returns = rolling_mean(returns, window=period)
            rolling_vol = rolling_std(returns, window=period)

            # Avoid division by zero
            sharpe_ratios = rolling_returns / (rolling_vol + 1e-8)
            return float(sharpe_ratios.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT Sharpe ratio calculation failed: {e}")
            return self._fallback_sharpe_ratio(returns, period)

    def _calculate_max_drawdown_vectorbt(self, prices: pd.Series, period: int) -> float:
        """Calculate maximum drawdown using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_max_drawdown(prices, period)

            # VectorBT-optimized drawdown calculation
            rolling_max_prices = rolling_max(prices, window=period)
            drawdowns = (prices - rolling_max_prices) / rolling_max_prices
            return float(drawdowns.min())

        except Exception as e:
            self.logger.warning(f"VectorBT max drawdown calculation failed: {e}")
            return self._fallback_max_drawdown(prices, period)

    def _calculate_win_rate_vectorbt(self, returns: pd.Series, period: int) -> float:
        """Calculate win rate using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_win_rate(returns, period)

            # VectorBT-optimized win rate calculation
            positive_returns = (returns > 0).astype(int)
            rolling_win_rate = rolling_mean(positive_returns, window=period)
            return float(rolling_win_rate.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT win rate calculation failed: {e}")
            return self._fallback_win_rate(returns, period)

    def _calculate_volatility_clustering_vectorbt(self, returns: pd.Series, period: int) -> float:
        """Calculate volatility clustering using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_volatility_clustering(returns, period)

            # VectorBT-optimized volatility clustering
            squared_returns = returns ** 2
            rolling_vol = rolling_std(returns, window=period)
            rolling_squared_vol = rolling_mean(squared_returns, window=period)

            # Correlation between current volatility and past volatility
            vol_correlation = rolling_corr(rolling_vol, rolling_vol.shift(1), window=period)
            return float(vol_correlation.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT volatility clustering calculation failed: {e}")
            return self._fallback_volatility_clustering(returns, period)

    def _calculate_regime_stability_vectorbt(self, prices: pd.Series, period: int) -> float:
        """Calculate regime stability using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_regime_stability(prices, period)

            # VectorBT-optimized regime stability
            returns = prices.pct_change()
            rolling_vol = rolling_std(returns, window=period)
            vol_changes = rolling_vol.pct_change()

            # Stability is inverse of volatility of volatility
            stability = 1.0 / (rolling_std(vol_changes, window=period) + 1e-8)
            return float(stability.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT regime stability calculation failed: {e}")
            return self._fallback_regime_stability(prices, period)

    def _assess_data_quality_vectorbt(self, data: pd.Series, period: int) -> Dict[str, Any]:
        """Assess data quality using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_data_quality(data, period)

            # VectorBT-optimized data quality assessment
            missing_ratio = data.isna().sum() / len(data)
            outlier_ratio = self._calculate_outlier_ratio_vectorbt(data, period)
            stationarity = self._calculate_stationarity_vectorbt(data, period)

            return {
                'missing_ratio': float(missing_ratio),
                'outlier_ratio': float(outlier_ratio),
                'stationarity': float(stationarity),
                'data_length': len(data),
                'period_coverage': len(data) / period if period > 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"VectorBT data quality assessment failed: {e}")
            return self._fallback_data_quality(data, period)

    def _calculate_outlier_ratio_vectorbt(self, data: pd.Series, period: int) -> float:
        """Calculate outlier ratio using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_outlier_ratio(data, period)

            # VectorBT-optimized outlier detection
            rolling_mean_data = rolling_mean(data, window=period)
            rolling_std_data = rolling_std(data, window=period)

            # Z-score based outlier detection
            z_scores = (data - rolling_mean_data) / (rolling_std_data + 1e-8)
            outliers = (np.abs(z_scores) > 3).astype(int)
            outlier_ratio = rolling_mean(outliers, window=period)

            return float(outlier_ratio.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT outlier ratio calculation failed: {e}")
            return self._fallback_outlier_ratio(data, period)

    def _calculate_stationarity_vectorbt(self, data: pd.Series, period: int) -> float:
        """Calculate stationarity using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_stationarity(data, period)

            # VectorBT-optimized stationarity test
            returns = data.pct_change()
            rolling_vol = rolling_std(returns, window=period)

            # Stationarity as inverse of volatility of volatility
            vol_of_vol = rolling_std(rolling_vol, window=period)
            stationarity = 1.0 / (vol_of_vol + 1e-8)

            return float(stationarity.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT stationarity calculation failed: {e}")
            return self._fallback_stationarity(data, period)

    def optimize_interaction_generation(self, features: pd.DataFrame, targets: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """
        Optimize interaction generation using VectorBT with advanced algorithms.

        Args:
            features: Feature DataFrame
            targets: Optional target series for utility scoring

        Returns:
            List of generated interactions
        """
        tprint_info(f"⚡ Optimizing interaction generation for {len(features.columns)} features")

        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._fallback_interaction_generation(features, targets)

        try:
            start_time = time.time()
            interactions = []

            # VectorBT-optimized interaction generation
            feature_names = list(features.columns)

            # Generate different types of interactions
            interaction_types = [
                'product', 'ratio', 'difference', 'sum', 'log_product',
                'log_ratio', 'polynomial', 'conditional', 'rolling_mean',
                'rolling_std', 'correlation', 'zscore', 'rank'
            ]

            for interaction_type in interaction_types:
                type_interactions = self._generate_interactions_by_type_vectorbt(
                    features, targets, interaction_type
                )
                interactions.extend(type_interactions)

            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['vectorbt_operations'] += 1

            tprint_success(f"✅ Generated {len(interactions)} interactions in {execution_time:.3f}s")
            return interactions

        except Exception as e:
            tprint_error(f"❌ VectorBT interaction generation failed: {e}")
            return self._fallback_interaction_generation(features, targets)

    def _generate_interactions_by_type_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series],
                                              interaction_type: str) -> List[Dict[str, Any]]:
        """Generate interactions of a specific type using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        try:
            if interaction_type == 'product':
                interactions = self._generate_product_interactions_vectorbt(features, targets)
            elif interaction_type == 'ratio':
                interactions = self._generate_ratio_interactions_vectorbt(features, targets)
            elif interaction_type == 'difference':
                interactions = self._generate_difference_interactions_vectorbt(features, targets)
            elif interaction_type == 'sum':
                interactions = self._generate_sum_interactions_vectorbt(features, targets)
            elif interaction_type == 'log_product':
                interactions = self._generate_log_product_interactions_vectorbt(features, targets)
            elif interaction_type == 'log_ratio':
                interactions = self._generate_log_ratio_interactions_vectorbt(features, targets)
            elif interaction_type == 'polynomial':
                interactions = self._generate_polynomial_interactions_vectorbt(features, targets)
            elif interaction_type == 'conditional':
                interactions = self._generate_conditional_interactions_vectorbt(features, targets)
            elif interaction_type == 'rolling_mean':
                interactions = self._generate_rolling_interactions_vectorbt(features, targets, 'mean')
            elif interaction_type == 'rolling_std':
                interactions = self._generate_rolling_interactions_vectorbt(features, targets, 'std')
            elif interaction_type == 'correlation':
                interactions = self._generate_correlation_interactions_vectorbt(features, targets)
            elif interaction_type == 'zscore':
                interactions = self._generate_zscore_interactions_vectorbt(features, targets)
            elif interaction_type == 'rank':
                interactions = self._generate_rank_interactions_vectorbt(features, targets)

        except Exception as e:
            self.logger.warning(f"VectorBT {interaction_type} interaction generation failed: {e}")

        return interactions

    def _generate_product_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate product interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized product calculation
                    product = features[feat1] * features[feat2]

                    if not product.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(product, targets)

                        interactions.append({
                            'name': f"product_{feat1}_{feat2}",
                            'feature_series': product,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'product',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Product interaction failed for {feat1} x {feat2}: {e}")
                    continue

        return interactions

    def _generate_ratio_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate ratio interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized ratio calculation
                    ratio = features[feat1] / (features[feat2] + 1e-8)

                    if not ratio.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(ratio, targets)

                        interactions.append({
                            'name': f"ratio_{feat1}_{feat2}",
                            'feature_series': ratio,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'ratio',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Ratio interaction failed for {feat1} / {feat2}: {e}")
                    continue

        return interactions

    def _generate_difference_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate difference interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized difference calculation
                    difference = features[feat1] - features[feat2]

                    if not difference.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(difference, targets)

                        interactions.append({
                            'name': f"difference_{feat1}_{feat2}",
                            'feature_series': difference,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'difference',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Difference interaction failed for {feat1} - {feat2}: {e}")
                    continue

        return interactions

    def _generate_sum_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate sum interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized sum calculation
                    sum_feature = features[feat1] + features[feat2]

                    if not sum_feature.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(sum_feature, targets)

                        interactions.append({
                            'name': f"sum_{feat1}_{feat2}",
                            'feature_series': sum_feature,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'sum',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Sum interaction failed for {feat1} + {feat2}: {e}")
                    continue

        return interactions

    def _generate_log_product_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate log product interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized log product calculation
                    feat1_safe = np.where(features[feat1] <= 0, np.abs(features[feat1]) + 1e-8, features[feat1])
                    feat2_safe = np.where(features[feat2] <= 0, np.abs(features[feat2]) + 1e-8, features[feat2])

                    log_product = np.log(feat1_safe) * np.log(feat2_safe)
                    log_product = pd.Series(log_product, index=features.index)

                    if not log_product.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(log_product, targets)

                        interactions.append({
                            'name': f"log_product_{feat1}_{feat2}",
                            'feature_series': log_product,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'log_product',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Log product interaction failed for {feat1} x {feat2}: {e}")
                    continue

        return interactions

    def _generate_log_ratio_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate log ratio interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized log ratio calculation
                    feat1_safe = np.where(features[feat1] <= 0, np.abs(features[feat1]) + 1e-8, features[feat1])
                    feat2_safe = np.where(features[feat2] <= 0, np.abs(features[feat2]) + 1e-8, features[feat2])

                    log_ratio = np.log(feat1_safe) - np.log(feat2_safe)
                    log_ratio = pd.Series(log_ratio, index=features.index)

                    if not log_ratio.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(log_ratio, targets)

                        interactions.append({
                            'name': f"log_ratio_{feat1}_{feat2}",
                            'feature_series': log_ratio,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'log_ratio',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Log ratio interaction failed for {feat1} / {feat2}: {e}")
                    continue

        return interactions

    def _generate_polynomial_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate polynomial interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for feat in feature_names:
            try:
                # VectorBT-optimized polynomial calculation
                polynomial = features[feat] ** 2

                if not polynomial.isna().all():
                    utility_score = self._calculate_utility_score_vectorbt(polynomial, targets)

                    interactions.append({
                        'name': f"polynomial_{feat}",
                        'feature_series': polynomial,
                        'parent_features': [feat],
                        'interaction_type': 'polynomial',
                        'utility_score': utility_score,
                        'vectorbt_optimized': True
                    })

            except Exception as e:
                self.logger.debug(f"Polynomial interaction failed for {feat}: {e}")
                continue

        return interactions

    def _generate_conditional_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate conditional interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized conditional calculation
                    threshold = features[feat2].median()
                    conditional = features[feat1] * (features[feat2] > threshold).astype(int)

                    if not conditional.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(conditional, targets)

                        interactions.append({
                            'name': f"conditional_{feat1}_{feat2}",
                            'feature_series': conditional,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'conditional',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Conditional interaction failed for {feat1} | {feat2}: {e}")
                    continue

        return interactions

    def _generate_rolling_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series],
                                             operation: str) -> List[Dict[str, Any]]:
        """Generate rolling interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)
        window = 20  # Default window size

        for feat in feature_names:
            try:
                # VectorBT-optimized rolling calculation
                if operation == 'mean':
                    rolling_feature = rolling_mean(features[feat], window=window)
                elif operation == 'std':
                    rolling_feature = rolling_std(features[feat], window=window)
                else:
                    continue

                if not rolling_feature.isna().all():
                    utility_score = self._calculate_utility_score_vectorbt(rolling_feature, targets)

                    interactions.append({
                        'name': f"rolling_{operation}_{feat}",
                        'feature_series': rolling_feature,
                        'parent_features': [feat],
                        'interaction_type': f'rolling_{operation}',
                        'utility_score': utility_score,
                        'vectorbt_optimized': True
                    })

            except Exception as e:
                self.logger.debug(f"Rolling {operation} interaction failed for {feat}: {e}")
                continue

        return interactions

    def _generate_correlation_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate correlation interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)
        window = 20  # Default window size

        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                try:
                    # VectorBT-optimized correlation calculation
                    correlation = rolling_corr(features[feat1], features[feat2], window=window)

                    if not correlation.isna().all():
                        utility_score = self._calculate_utility_score_vectorbt(correlation, targets)

                        interactions.append({
                            'name': f"correlation_{feat1}_{feat2}",
                            'feature_series': correlation,
                            'parent_features': [feat1, feat2],
                            'interaction_type': 'correlation',
                            'utility_score': utility_score,
                            'vectorbt_optimized': True
                        })

                except Exception as e:
                    self.logger.debug(f"Correlation interaction failed for {feat1} x {feat2}: {e}")
                    continue

        return interactions

    def _generate_zscore_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate z-score interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for feat in feature_names:
            try:
                # VectorBT-optimized z-score calculation
                zscore_feature = zscore(features[feat])

                if not zscore_feature.isna().all():
                    utility_score = self._calculate_utility_score_vectorbt(zscore_feature, targets)

                    interactions.append({
                        'name': f"zscore_{feat}",
                        'feature_series': zscore_feature,
                        'parent_features': [feat],
                        'interaction_type': 'zscore',
                        'utility_score': utility_score,
                        'vectorbt_optimized': True
                    })

            except Exception as e:
                self.logger.debug(f"Z-score interaction failed for {feat}: {e}")
                continue

        return interactions

    def _generate_rank_interactions_vectorbt(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate rank interactions using VectorBT optimization."""
        interactions = []
        feature_names = list(features.columns)

        for feat in feature_names:
            try:
                # VectorBT-optimized rank calculation
                rank_feature = rank(features[feat])

                if not rank_feature.isna().all():
                    utility_score = self._calculate_utility_score_vectorbt(rank_feature, targets)

                    interactions.append({
                        'name': f"rank_{feat}",
                        'feature_series': rank_feature,
                        'parent_features': [feat],
                        'interaction_type': 'rank',
                        'utility_score': utility_score,
                        'vectorbt_optimized': True
                    })

            except Exception as e:
                self.logger.debug(f"Rank interaction failed for {feat}: {e}")
                continue

        return interactions

    def _calculate_utility_score_vectorbt(self, feature_series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate utility score using VectorBT optimization."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(feature_series.var())

            # Calculate correlation with targets
            correlation = feature_series.corr(targets)
            if pd.isna(correlation):
                return 0.0

            # Use absolute correlation as utility score
            return abs(correlation)

        except Exception as e:
            self.logger.debug(f"Utility score calculation failed: {e}")
            return 0.0

    def optimize_lookback_analysis(self, data: pd.DataFrame, features: List[str],
                                 lookback_periods: List[int]) -> Dict[str, Dict[int, Any]]:
        """
        Optimize lookback analysis using VectorBT with advanced algorithms.

        Args:
            data: Input data
            features: List of features to analyze
            lookback_periods: List of lookback periods to test

        Returns:
            Dictionary mapping features to lookback analysis results
        """
        tprint_info(f"🔍 Optimizing lookback analysis for {len(features)} features")

        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._fallback_lookback_analysis(data, features, lookback_periods)

        try:
            start_time = time.time()
            results = {}

            # VectorBT-optimized lookback analysis
            for feature in features:
                if feature not in data.columns:
                    continue

                feature_results = {}
                for lookback in lookback_periods:
                    lookback_result = self._analyze_lookback_period_vectorbt(
                        data, feature, lookback
                    )
                    feature_results[lookback] = lookback_result

                results[feature] = feature_results

                # Update performance stats
                self.performance_stats['vectorbt_operations'] += 1

            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['total_operations'] += len(features)

            tprint_success(f"✅ Lookback analysis completed in {execution_time:.3f}s")
            return results

        except Exception as e:
            tprint_error(f"❌ VectorBT lookback analysis failed: {e}")
            return self._fallback_lookback_analysis(data, features, lookback_periods)

    def _analyze_lookback_period_vectorbt(self, data: pd.DataFrame, feature: str, lookback: int) -> Dict[str, Any]:
        """Analyze a single lookback period using VectorBT optimization."""
        try:
            if feature not in data.columns:
                return {'error': f'Feature {feature} not found in data'}

            feature_series = data[feature]

            # VectorBT-optimized lookback analysis
            rolling_mean_feature = rolling_mean(feature_series, window=lookback)
            rolling_std_feature = rolling_std(feature_series, window=lookback)
            rolling_min_feature = rolling_min(feature_series, window=lookback)
            rolling_max_feature = rolling_max(feature_series, window=lookback)

            # Advanced statistical measures
            stability = self._calculate_stability_vectorbt(feature_series, lookback)
            predictability = self._calculate_predictability_vectorbt(feature_series, lookback)
            information_content = self._calculate_information_content_vectorbt(feature_series, lookback)

            return {
                'lookback': lookback,
                'rolling_mean': rolling_mean_feature,
                'rolling_std': rolling_std_feature,
                'rolling_min': rolling_min_feature,
                'rolling_max': rolling_max_feature,
                'stability': stability,
                'predictability': predictability,
                'information_content': information_content,
                'data_quality': self._assess_lookback_data_quality_vectorbt(feature_series, lookback)
            }

        except Exception as e:
            self.logger.error(f"VectorBT lookback analysis failed for {feature} period {lookback}: {e}")
            return {'error': str(e), 'lookback': lookback}

    def _calculate_stability_vectorbt(self, feature_series: pd.Series, lookback: int) -> float:
        """Calculate stability using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_stability(feature_series, lookback)

            # VectorBT-optimized stability calculation
            rolling_std_feature = rolling_std(feature_series, window=lookback)
            stability = 1.0 / (rolling_std_feature + 1e-8)
            return float(stability.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT stability calculation failed: {e}")
            return self._fallback_stability(feature_series, lookback)

    def _calculate_predictability_vectorbt(self, feature_series: pd.Series, lookback: int) -> float:
        """Calculate predictability using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_predictability(feature_series, lookback)

            # VectorBT-optimized predictability calculation
            rolling_mean_feature = rolling_mean(feature_series, window=lookback)
            prediction_error = np.abs(feature_series - rolling_mean_feature)
            predictability = 1.0 / (prediction_error + 1e-8)
            return float(predictability.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT predictability calculation failed: {e}")
            return self._fallback_predictability(feature_series, lookback)

    def _calculate_information_content_vectorbt(self, feature_series: pd.Series, lookback: int) -> float:
        """Calculate information content using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_information_content(feature_series, lookback)

            # VectorBT-optimized information content calculation
            rolling_std_feature = rolling_std(feature_series, window=lookback)
            information_content = rolling_std_feature / (feature_series + 1e-8)
            return float(information_content.mean())

        except Exception as e:
            self.logger.warning(f"VectorBT information content calculation failed: {e}")
            return self._fallback_information_content(feature_series, lookback)

    def _assess_lookback_data_quality_vectorbt(self, feature_series: pd.Series, lookback: int) -> Dict[str, Any]:
        """Assess lookback data quality using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._fallback_lookback_data_quality(feature_series, lookback)

            # VectorBT-optimized lookback data quality assessment
            missing_ratio = feature_series.isna().sum() / len(feature_series)
            outlier_ratio = self._calculate_outlier_ratio_vectorbt(feature_series, lookback)
            stationarity = self._calculate_stationarity_vectorbt(feature_series, lookback)

            return {
                'missing_ratio': float(missing_ratio),
                'outlier_ratio': float(outlier_ratio),
                'stationarity': float(stationarity),
                'data_length': len(feature_series),
                'lookback_coverage': len(feature_series) / lookback if lookback > 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"VectorBT lookback data quality assessment failed: {e}")
            return self._fallback_lookback_data_quality(feature_series, lookback)

    # Fallback methods for when VectorBT is not available
    def _fallback_period_analysis(self, data: pd.DataFrame, periods: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fallback period analysis when VectorBT is not available."""
        results = {}
        for period in periods:
            results[period] = {
                'period': period,
                'error': 'VectorBT not available',
                'fallback': True
            }
        return results

    def optimize_features(self, data: pd.DataFrame, targets: pd.Series) -> 'VectorBTOptimizationResult':
        """
        Optimize features using VectorBT with advanced algorithms.
        
        Args:
            data: Input data with features
            targets: Target values for optimization
            
        Returns:
            VectorBTOptimizationResult with optimized features
        """
        start_time = time.time()
        
        try:
            tprint_info("⚡ Starting VectorBT feature optimization")
            tprint_debug(f"📊 Input data shape: {data.shape}")
            tprint_debug(f"📊 Target data shape: {targets.shape if targets is not None else 'None'}")
            tprint_debug(f"📊 Available columns: {list(data.columns)}")
            
            # Skip UnifiedVectorizationManager for now due to missing performance_monitoring method
            # Use standard VectorBT optimization instead
            tprint_info("🔄 Using standard VectorBT optimization")
            
            # Fallback to comprehensive VectorBT optimization
            tprint_info("🔄 Starting comprehensive VectorBT optimization")
            optimized_features = data.copy()
            optimization_metrics = {}
            
            # Perform period analysis optimization
            if 'close' in data.columns:
                tprint_info("📈 Performing period analysis optimization")
                periods = [5, 10, 20, 50, 100, 200]
                tprint_debug(f"📊 Analyzing periods: {periods}")
                period_results = self.optimize_period_analysis(data, periods)
                optimization_metrics['period_analysis'] = period_results
                tprint_success(f"✅ Period analysis completed: {len(period_results)} periods analyzed")
            else:
                tprint_warning("⚠️ No close price data available for period analysis")
            
            # Perform interaction generation optimization
            feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            if feature_columns:
                tprint_info(f"🔗 Performing interaction generation optimization for {len(feature_columns)} features")
                tprint_debug(f"📊 Feature columns: {feature_columns}")
                interaction_results = self.optimize_interaction_generation(
                    data[feature_columns], targets
                )
                optimization_metrics['interaction_generation'] = interaction_results
                tprint_success(f"✅ Interaction generation completed: {len(interaction_results)} interactions generated")
                
                # Add generated interactions to optimized features
                added_interactions = 0
                for interaction in interaction_results:
                    if 'feature_name' in interaction and 'values' in interaction:
                        optimized_features[interaction['feature_name']] = interaction['values']
                        added_interactions += 1
                tprint_info(f"📊 Added {added_interactions} interactions to optimized features")
            else:
                tprint_warning("⚠️ No feature columns available for interaction generation")
            
            # Perform lookback analysis optimization
            if feature_columns:
                tprint_info("🔍 Performing lookback analysis optimization")
                lookback_periods = [5, 10, 20, 50]
                tprint_debug(f"📊 Lookback periods: {lookback_periods}")
                lookback_results = self.optimize_lookback_analysis(
                    data, feature_columns, lookback_periods
                )
                optimization_metrics['lookback_analysis'] = lookback_results
                tprint_success(f"✅ Lookback analysis completed: {len(lookback_results)} features analyzed")
            else:
                tprint_warning("⚠️ No feature columns available for lookback analysis")
            
            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['vectorbt_operations'] += 1
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_success(f"✅ VectorBT optimization completed in {execution_time:.3f}s")
            tprint_info(f"📊 Final optimized features shape: {optimized_features.shape}")
            
            return VectorBTOptimizationResult(
                optimized_features=optimized_features,
                optimization_metrics=optimization_metrics,
                performance_stats=self.performance_stats,
                success=True
            )
                
        except Exception as e:
            tprint_error(f"❌ Feature optimization failed: {e}")
            tprint_debug(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            return VectorBTOptimizationResult(
                optimized_features=data,
                optimization_metrics={},
                performance_stats=self.performance_stats,
                success=False,
                error_message=str(e)
            )
        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            tprint_debug(f"⏱️ VectorBT optimization completed in {execution_time:.3f}s")

    def _fallback_interaction_generation(self, features: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Fallback interaction generation when VectorBT is not available."""
        return []

    def _fallback_lookback_analysis(self, data: pd.DataFrame, features: List[str],
                                  lookback_periods: List[int]) -> Dict[str, Dict[int, Any]]:
        """Fallback lookback analysis when VectorBT is not available."""
        results = {}
        for feature in features:
            results[feature] = {}
            for lookback in lookback_periods:
                results[feature][lookback] = {
                    'lookback': lookback,
                    'error': 'VectorBT not available',
                    'fallback': True
                }
        return results

    # Additional fallback methods for individual calculations
    def _fallback_sharpe_ratio(self, returns: pd.Series, period: int) -> float:
        """Fallback Sharpe ratio calculation."""
        try:
            rolling_returns = returns.rolling(window=period).mean()
            rolling_vol = returns.rolling(window=period).std()
            sharpe_ratios = rolling_returns / (rolling_vol + 1e-8)
            return float(sharpe_ratios.mean())
        except:
            return 0.0

    def _fallback_max_drawdown(self, prices: pd.Series, period: int) -> float:
        """Fallback max drawdown calculation."""
        try:
            rolling_max_prices = prices.rolling(window=period).max()
            drawdowns = (prices - rolling_max_prices) / rolling_max_prices
            return float(drawdowns.min())
        except:
            return 0.0

    def _fallback_win_rate(self, returns: pd.Series, period: int) -> float:
        """Fallback win rate calculation."""
        try:
            positive_returns = (returns > 0).astype(int)
            rolling_win_rate = positive_returns.rolling(window=period).mean()
            return float(rolling_win_rate.mean())
        except:
            return 0.0

    def _fallback_volatility_clustering(self, returns: pd.Series, period: int) -> float:
        """Fallback volatility clustering calculation."""
        try:
            rolling_vol = returns.rolling(window=period).std()
            vol_correlation = rolling_vol.corr(rolling_vol.shift(1))
            return float(vol_correlation) if not pd.isna(vol_correlation) else 0.0
        except:
            return 0.0

    def _fallback_regime_stability(self, prices: pd.Series, period: int) -> float:
        """Fallback regime stability calculation."""
        try:
            returns = prices.pct_change()
            rolling_vol = returns.rolling(window=period).std()
            vol_changes = rolling_vol.pct_change()
            stability = 1.0 / (vol_changes.rolling(window=period).std() + 1e-8)
            return float(stability.mean())
        except:
            return 0.0

    def _fallback_data_quality(self, data: pd.Series, period: int) -> Dict[str, Any]:
        """Fallback data quality assessment."""
        try:
            missing_ratio = data.isna().sum() / len(data)
            outlier_ratio = 0.0  # Simplified
            stationarity = 0.0  # Simplified

            return {
                'missing_ratio': float(missing_ratio),
                'outlier_ratio': float(outlier_ratio),
                'stationarity': float(stationarity),
                'data_length': len(data),
                'period_coverage': len(data) / period if period > 0 else 0
            }
        except:
            return {'error': 'Data quality assessment failed'}

    def _fallback_outlier_ratio(self, data: pd.Series, period: int) -> float:
        """Fallback outlier ratio calculation."""
        try:
            rolling_mean_data = data.rolling(window=period).mean()
            rolling_std_data = data.rolling(window=period).std()
            z_scores = (data - rolling_mean_data) / (rolling_std_data + 1e-8)
            outliers = (np.abs(z_scores) > 3).astype(int)
            return float(outliers.mean())
        except:
            return 0.0

    def _fallback_stationarity(self, data: pd.Series, period: int) -> float:
        """Fallback stationarity calculation."""
        try:
            returns = data.pct_change()
            rolling_vol = returns.rolling(window=period).std()
            vol_of_vol = rolling_vol.rolling(window=period).std()
            stationarity = 1.0 / (vol_of_vol + 1e-8)
            return float(stationarity.mean())
        except:
            return 0.0

    def _fallback_stability(self, feature_series: pd.Series, lookback: int) -> float:
        """Fallback stability calculation."""
        try:
            rolling_std_feature = feature_series.rolling(window=lookback).std()
            stability = 1.0 / (rolling_std_feature + 1e-8)
            return float(stability.mean())
        except:
            return 0.0

    def _fallback_predictability(self, feature_series: pd.Series, lookback: int) -> float:
        """Fallback predictability calculation."""
        try:
            rolling_mean_feature = feature_series.rolling(window=lookback).mean()
            prediction_error = np.abs(feature_series - rolling_mean_feature)
            predictability = 1.0 / (prediction_error + 1e-8)
            return float(predictability.mean())
        except:
            return 0.0

    def _fallback_information_content(self, feature_series: pd.Series, lookback: int) -> float:
        """Fallback information content calculation."""
        try:
            rolling_std_feature = feature_series.rolling(window=lookback).std()
            information_content = rolling_std_feature / (feature_series + 1e-8)
            return float(information_content.mean())
        except:
            return 0.0

    def _fallback_lookback_data_quality(self, feature_series: pd.Series, lookback: int) -> Dict[str, Any]:
        """Fallback lookback data quality assessment."""
        try:
            missing_ratio = feature_series.isna().sum() / len(feature_series)
            outlier_ratio = 0.0  # Simplified
            stationarity = 0.0  # Simplified

            return {
                'missing_ratio': float(missing_ratio),
                'outlier_ratio': float(outlier_ratio),
                'stationarity': float(stationarity),
                'data_length': len(feature_series),
                'lookback_coverage': len(feature_series) / lookback if lookback > 0 else 0
            }
        except:
            return {'error': 'Lookback data quality assessment failed'}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'fallback_operations': 0,
            'gpu_operations': 0,
            'total_execution_time': 0.0,
            'memory_usage': 0.0
        }

def create_enhanced_vectorbt_optimizer(config: Optional[VectorBTOptimizationConfig] = None) -> EnhancedVectorBTOptimizer:
    """Create an enhanced VectorBT optimizer with default configuration."""
    return EnhancedVectorBTOptimizer(config)
